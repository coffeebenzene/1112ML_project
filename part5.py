import argparse
import collections
import itertools
import os.path
import time

import numpy as np
from scipy.optimize import minimize

import crf.crf_feature as crf_feature
import crf.forward_backward as forward_backward

# For debugging
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

all_y = ("O", "B-positive", "B-neutral", "B-negative", "I-positive", "I-neutral", "I-negative")
all_y_ss = all_y + ("START", "STOP")
unk_threshold = 3 # k. If word appears less than this frequency, then word is treated as unknown.

def sentence_gen(f):
    """Generates a list of strings for a sentence.
       Each string is either a "word", or a "word tag" pair (each line of the data file).
    """
    sentence = []
    in_sentence = False
    for line in f:
        line = line.strip()
        if line:
            sentence.append(line)
            in_sentence=True
        elif in_sentence and not line:
            yield sentence
            sentence = []
            in_sentence=False
        else:
            in_sentence=False
    if sentence:
        yield sentence



def preprocess_train(training_file):
    """Preprocess data and return
       1. train data: [(sentence, tags), ...]
       2. wordcount: {word: count}
    """
    train_data = []
    wordcount = collections.Counter() # To identify UNK words
    
    # Split tags and count words
    for sentence_data in sentence_gen(training_file):
        sentence = []
        tags = []
        for pair in sentence_data:
            x, y = pair.rsplit(" ", 1) # word, tag
            sentence.append(x)
            tags.append(y)
        wordcount.update(sentence)
        train_data.append((sentence, tags))
    
    # Replace UNK words
    unk_words = set()
    for word, count in wordcount.items():
        if count < unk_threshold:
            unk_words.add(count)
    for sentence, tags in train_data:
        for i, word in enumerate(sentence):
            if word in unk_words:
                sentence[i] = "#UNK#"
    
    return train_data, wordcount

def train(training_file):
    train_data, wordcount = preprocess_train(training_file)
    
    hmm_features = crf_feature.generate_hmm_features(all_y_ss, wordcount.keys())
    vfeature_func = crf_feature.make_vfeature_func(hmm_features)
    
    regularizer = 0.5 # Larger is more regualrization
    
    def loss_grad(w):
        loss = 0
        grad = np.zeros(len(w))
        for sentence, tags in train_data:
            potential = forward_backward.potential_func(w, vfeature_func, sentence)
            alpha_table, beta_table = forward_backward.forward_backward(potential, sentence, all_y)
            z, marginals = forward_backward.calc_z_marginals(alpha_table, beta_table, potential, all_y)
            
            # Fixed per sentence calculation
            loss += np.log(z)
            loss += np.dot(w,w)*regularizer
            grad += w*regularizer*2
            # Per word calculation
            prev_tag = "START"
            for i, tag in enumerate(tags):
                feature = vfeature_func(prev_tag, tag, i, sentence)
                loss -= np.dot(w, feature)
                grad -= feature
                prev_tag = tag
            feature = vfeature_func(prev_tag, "STOP", None, sentence)
            loss -= np.dot(w, feature)
            grad -= feature
            # hard part of crf
            for i in range(len(tags)):
                for u,v in itertools.product(all_y, all_y):
                    grad += marginals[(i, u, v)]/z*vfeature_func(u, v, i, sentence)
        print(w) # DEBUG
        return loss, grad
    
    result = minimize(loss_grad, np.ones(len(hmm_features)),
                      jac=True, method="L-BFGS-B")
                      #options={"ftol":1*np.finfo(float).eps, "gtol":1e-15})
    
    print(result.x)
    print(result.fun)
    print(result.jac)
    print(result.message)
    return None

def predict(e, t, in_file):
    """Generate list of tuples: [(sentence, prediction), ...]"""
    predictions = []
    
    return predictions


def main(args):
    train_path = os.path.join(args.folder, args.train)
    with open(train_path, encoding="utf-8") as training_file:
        train(training_file)
    """
    if debug:
        pprint.pprint(e)
        pprint.pprint(t)
    
    infile_path = os.path.join(args.folder, args.infile)
    with open(infile_path, encoding="utf-8") as in_file:
        predictions = predict(e, t, in_file)
    
    outfile_path = os.path.join(args.folder, args.outfile)
    with open(outfile_path, "w", encoding="utf-8") as out_file:
        first = True
        for sentence_pair in predictions:
            if first:
                first = False
            else:
                out_file.write("\n\n")
            str_predictions = [" ".join(pair) for pair in zip(*sentence_pair)]
            str_predictions = "\n".join(str_predictions)
            out_file.write(str_predictions)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=str, default="train", help='Training dataset file')
    parser.add_argument('-i', '--infile', type=str, default="dev.in", help='Input (to be decoded) dataset file')
    parser.add_argument('-o', '--outfile', type=str, default="dev.p5.out", help='Output (the predictions) file')
    parser.add_argument('-f', '--folder', type=str, default=".", help='Folder containing files (prepended to files).')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    debug = args.debug
    if debug:
        import pprint
    import pprint
    
    start_time = time.time()
    main(args)
    print("{}s".format(time.time()-start_time))