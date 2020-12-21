"""Train a text classifier using the TextCategorizer component. 
The textcat model is trained and added to spacy.pipeline, 
and predictions are available via `doc.cats`.
Code adapted from:
https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py
For more details, see the documentation:
* Training: https://spacy.io/usage/training
Compatible with: spaCy v2.0.0+
"""
import gc
import time
import plac
import numpy
import random
from pathlib import Path
from tqdm.auto import tqdm
from itertools import chain

import scispacy
import spacy
from spacy.util import minibatch, compounding, decaying
from sklearn.metrics import classification_report

ARCHS = {"ensemble","simple_cnn","bow"}

def gen_dict_label(label, label_set):
    """Helper function that creates a dict of booleans specifying which label is
    True for the observation."""
    d = dict()
    for l in label_set:
        if label == l:
            d[str(l)] = True
        if label != l:
            d[str(l)] = False
    return d


def gen_dict_labels(labels):
    """Helper function that takes a list of labels and creates a list of dicts
    specifying which label is True for a given observation."""
    label_set = set(labels)
    return [gen_dict_label(label, label_set) for label in labels]


def load_ft_file(file, label_text_sep=" "):
    """Load and parse fasttext formatted file"""
    if type(file) != Path:
        file = Path(file).absolute()
    if file.is_file():
        labels = []
        texts  = []
        with open(file, "r") as f:
            line = f.readline()
            while line:
                labels.append(line[:line.index(label_text_sep)].replace('__label__',''))
                texts.append(line[line.index(label_text_sep) + 1:].strip())
                line = f.readline()
        labels = gen_dict_labels(labels)
        return (texts, labels)
    else:
        raise FileNotFoundError("Could not find file:\n-{}".format(file))


def prep_training_data(train_texts, train_labels, parser=lambda x:x):
    """Helper function to normalize texts and transform labels ("cats") into a format suitable for 
    spacy's TextCategorizer and it's training requirements."""
    return list(zip([parser(txt) for txt in train_texts], 
                    [{"cats": labels} for labels in train_labels]))


def mk_model(model=None, arch="simple_cnn", base_model=None, model_labels=None):
    """Create or load a spacy pipeline and initialize TextCategorizer component"""
#     if base_model is not None:
#         print(f"Loading base_model {base_model}")
#         nlp = spacy.load(base_model)
    if model is not None:
        if 'spacy' in str(type(model)):
            nlp = model # use existing spaCy model
            print("Using model '%s'" % '_'.join([nlp.meta['lang'],nlp.meta['name']]))
        else:
            print("Loading model '%s'" % model)
            nlp = spacy.load(model)  # load existing (sci)spaCy model
    else:
        print("Creating blank 'en' model")
        nlp = spacy.blank("en")  # create blank Language class
        
    # load textcat checkpoint to continue training
    if base_model is not None:
        # possible todo: was having errors with nlp.from_disk(base_model), but spacy.load works for now.
        print(f"Loading textcat from {base_model}")
        textcat = spacy.pipeline.TextCategorizer(nlp.vocab)
        textcat.from_disk(base_model)
        nlp.add_pipe(textcat)
#         nlp.add_pipe(spacy.load(base_model).get_pipe('textcat'))

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        print("Creating TextCategorizer")
        # spacy textcat architectures https://spacy.io/api/textcategorizer#architectures
        if arch in ARCHS:
            textcat = nlp.create_pipe(
                "textcat",
                config={
                    "exclusive_classes": True,
                    "architecture": arch, #todo can add extras such as attr and ngram_size if arch is 'ensemble' or 'bow'
                }
            )
            nlp.add_pipe(textcat, last=True)
        else:
            raise ValueError("Architecture %s is not supported. Please use architecture from %s" %
                             (arch, ", ".join(ARCHS)))
    
    # store model labels in textcategorizer if they don't already exist
    if model_labels:
        current_labels = set(nlp.get_pipe("textcat").labels)
        for l in model_labels:
            if l not in current_labels:
                nlp.get_pipe("textcat").add_label(str(l))
    return nlp


def evaluate(textcat, docs, labels):
    """Evaluate TextCategorizer performance"""
    preds = [max(doc.cats, key=doc.cats.get) for doc in textcat.pipe(docs)]
    labels = [str(gold) for label in labels for gold,istrue in label.items() if istrue]
    scores = classification_report(y_true=labels, y_pred=preds, output_dict=True)
    return scores, set(chain(labels, preds))


def train_textcat(nlp, train_data, init_tok2vec=None, continue_training=False,
                  epochs=10, seed=0, dropout_rates=(0.6, 0.2, 1e-4), minibatch_sizes=(1.0, 64.0, 1.001),
                  valid_docs=None, valid_labels=None, output_dir=None, use_tqdm=False):
    """Train, evaluate, and store TextCategorizer model."""
    if "textcat" in nlp.pipe_names:
        # set all seeds for reproducability
        random.seed(seed)
        numpy.random.seed(seed)
        if spacy.prefer_gpu():
            import cupy
            cupy.random.seed(seed)
        
        train_eval_time = time.time()
        
        if valid_docs is not None or init_tok2vec is not None:
            textcat = nlp.get_pipe("textcat")
        
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
        with nlp.disable_pipes(*other_pipes):  # only train textcat
            # if base
            if continue_training:
                # Start with an existing model, use default optimizer
                optimizer = nlp.resume_training()
            else:
                optimizer = nlp.begin_training()
            
            # load pretrained LMAO weights
            if init_tok2vec is not None:
                with init_tok2vec.open("rb") as file_:
                    print("Loading LMAO weights...")
                    textcat.model.tok2vec.from_bytes(file_.read())
            
            print("Training the model...")
            print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
            
            # create batch sizes
            min_batch_size, max_batch_size, update_by = minibatch_sizes
            batch_sizes = compounding(min_batch_size, max_batch_size, update_by)
            
            # create decaying dropout
            starting_dropout, ending_dropout, decay_rate = dropout_rates
            dropouts = decaying(starting_dropout, ending_dropout, decay_rate)
            
            best_avg_f1 = 0
            for i in range(epochs):
                print("Epoch:", i)
                losses = {}
                
                # batch up the examples using spaCy's minibatch
                random.shuffle(train_data)
                if use_tqdm:
                    train_data = tqdm(train_data, leave=False)
                batches = minibatch(train_data, size=batch_sizes)
                for batch, dropout in zip(batches, dropouts):
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)
                
                # evaluate model on validatation set
                if valid_docs is not None and valid_labels is not None:
                    with textcat.model.use_params(optimizer.averages):
                        scores, valid_label_set = evaluate(textcat, valid_docs, valid_labels)
                    print("{0:.3f}\t{1:}\t{2:}\t{3:}".format(losses["textcat"],"_____","_____","_____"))
                    avg_f1 = 0
                    for vc in valid_label_set:
                        print(
                            "{0:}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print as a table
                                vc,
                                scores[vc]["precision"],
                                scores[vc]["recall"],
                                scores[vc]["f1-score"],
                            )
                        )
                        avg_f1 += scores[vc]["f1-score"]
                    print("Accuracy:", scores["accuracy"])
                    print("_____________________________")
                    
                    # assign best model, score, and epoch
                    avg_f1 = avg_f1 / len(valid_label_set)
                    if avg_f1 > best_avg_f1:
                        best_avg_f1 = avg_f1
                        # overwrite the weak with the strong
                        store_model(output_dir, nlp, optimizer)
                
                if use_tqdm:
                    # train_data was put into tqdm object and won't shuffle properly due to indexing
                    # put train_data back to it's original type
                    train_data = train_data.iterable
            
            # store final model if no evaluation performed
            if valid_docs is None:
                store_model(output_dir, nlp, optimizer)
        
        print("Finished after: {0:.2f} minutes".format((time.time()-train_eval_time)/60))            
    else:
        raise NameError("Pipe 'textcat' is not in the nlp pipeline. Be sure to run mk_model() before training.")
    
    return nlp

def store_model(output_dir, nlp, optimizer):
    # method is designed to overwrite existing models in output_dir
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Model saved to {}".format(output_dir))
        
        
@plac.annotations(
    train_path=("Path to Fasttext formatted file for training.", "positional", None, Path),
    valid_path=("Path to Fasttext formatted file for validation. Defaults to None.", "option", "valid", Path),
    model=("Model name or path to custom spacy model. Defaults to blank 'en' model.", "option", "spacy_model", str),
    textcat_arch=("TextCategorizer model architecture. Defaults to 'simple_cnn'. Ignored if checkpoint is used.", "option", "arch", str, ["ensemble","simple_cnn","bow"]),
    base_model=("Path to trained textcat model to continue training.", "option", "base_model", Path),
    output_dir=("Optional output directory. Defaults to None.", "option", "outdir", Path),
    epochs=("Number of training epochs. Defaults to 10.", "option", "epochs", int),
    seed=("Seed to use for random state. Defaults to 13.", "option", "seed", int),
    dropout_rates=("Decaying % of neurons to randomly drop during backprop. Defaults to '0.6, 0.2, 1e-4'.", "option", "dropout_rates", str),
    minibatch_sizes=("Tuple comprised of minimum batch size, maximum batch size, and increment-by. Defaults to '1.0, 64.0, 1.001'", "option", "minibatch_sizes", str),
    init_tok2vec=("Pretrained tok2vec weights. Defaults to None.", "option", "t2v", Path),
    use_tqdm=("Print a progress bar during training. Defaults to False.", "flag", "tqdm")
)
def main(train_path, valid_path=None, model=None, 
         textcat_arch="simple_cnn", base_model=None, output_dir=None, epochs=10, 
         seed=13, dropout_rates=(0.6, 0.2, 1e-4), minibatch_sizes=(1.0, 64.0, 1.001), 
         init_tok2vec=None, use_tqdm=False):    
    
    # load train data
    train_texts, train_labels = load_ft_file(train_path)
    model_labels = set(train_labels[0].keys())
    
    # load spacy pipeline and initialize textcategorizer
    nlp = mk_model(model, 
                   arch=textcat_arch,
                   base_model=base_model,
                   model_labels=model_labels)
    
    # prepare train data 
    # passing in nlp.tokenizer as parser will return list of spacy docs
    # this will save time downstream during training 
    # (https://github.com/explosion/spaCy/blob/master/spacy/language.py#L438)
    print("Prepping training data")
    train_data = prep_training_data(train_texts, train_labels, parser=nlp.tokenizer)
    
    # save memory at all costs!
    del train_texts
    gc.collect()
    
    # load and create spacy docs out of validation data to save time for evaluation function
    if valid_path is not None:
        print("Prepping validation data")
        valid_docs, valid_labels = load_ft_file(valid_path)
        valid_docs = [nlp.tokenizer(text) for text in valid_docs]
        
    # convert dropout_rates to floats
    if type(dropout_rates) == str:
        dropout_rates = [float(i.strip()) for i in dropout_rates.split(',')]
    
    # convert minibatch_sizes to floats
    if type(minibatch_sizes) == str:
        minibatch_sizes = [float(i.strip()) for i in minibatch_sizes.split(',')]
    
    # train classifier
    train_textcat(nlp,
                  train_data,
                  init_tok2vec=init_tok2vec,
                  continue_training=False if base_model is None else True,
                  epochs=epochs,
                  seed=seed,
                  dropout_rates=dropout_rates,
                  minibatch_sizes=minibatch_sizes,
                  valid_docs=valid_docs,
                  valid_labels=valid_labels,
                  output_dir=output_dir,
                  use_tqdm=use_tqdm)
    
    
if __name__ == "__main__":
    plac.call(main)