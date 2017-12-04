import argparse
import collections
import itertools
import os.path
import time

import numpy as np
from scipy.optimize import minimize

import crf.crf_feature as crf_feature
import crf.forward_backward as fb

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
            x = x.casefold()
            sentence.append(x)
            tags.append(y)
        wordcount.update(sentence)
        train_data.append((sentence, tags))
    
    # Replace UNK words
    unk_words = set()
    unk_count = 0
    for word, count in wordcount.items():
        if count < unk_threshold:
            unk_words.add(count)
            unk_count += count
    for word in unk_words:
        del wordcount[word]
    wordcount["#UNK#"] = unk_count
    for sentence, tags in train_data:
        for i, word in enumerate(sentence):
            if word in unk_words:
                sentence[i] = "#UNK#"
    
    return train_data, wordcount

def train(training_file):
    train_data, wordcount = preprocess_train(training_file)
    
    if debug: #DEBUG
        print(len(train_data))
        print("-"*10)
    
    vfeature_func, feature_len = crf_feature.make_vfeature_func(all_y_ss, wordcount.keys())
    
    regularizer = 5 # Larger is more regularization
    
    def loss_grad(w):
        loss = 0
        grad = np.zeros(len(w))
        if debug: # DEBUG
            print("-"*10)
            print(w)
            print("max:{} | min:{}".format(np.max(w), np.min(w)))
            start = time.time()
        # regulariser
        loss += np.dot(w,w)*regularizer
        grad += w*regularizer*2
        # Per sentence
        for sentence, tags in train_data:
            potential = fb.potential_func(w, vfeature_func, sentence)
            alpha_table, beta_table = fb.forward_backward(potential, sentence, all_y)
            z, marginals = fb.calc_z_marginals(alpha_table, beta_table, potential, all_y)
            
            # Fixed per sentence calculation
            loss += np.log(z)
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
            for k in range(len(tags)):
                for (i,u),(j,v) in itertools.product(enumerate(all_y), repeat=2):
                    grad += marginals[k,i,j]*vfeature_func(u, v, k, sentence)
        if debug: # DEBUG
            print("loss:{}".format(loss))
            print("grad:{}".format(grad))
            print(time.time()-start)
            print("-"*10)
        return loss, grad
    
    bounds = [[None,None] for i in range(feature_len)]
    bounds[0] = [1,1]
    result = minimize(loss_grad, np.ones(feature_len),
                      jac=True, method="L-BFGS-B", bounds = bounds,
                      options = {"ftol":1e-7, "gtol":10, "disp":debug})
                      #options = {"ftol":1*np.finfo(float).eps, "gtol":1e-15, "disp":True})
                      # use default accuracy.
    
    if debug:
        print(result)
    model = {"vfeature_func": vfeature_func,
             "feature_len": feature_len,
             "weights": result.x,
             "wordcount": wordcount,
    }
    return model



def predict(model, in_file):
    """Generate list of tuples: [(sentence, prediction), ...]"""
    wordcount = model["wordcount"]
    w = model["weights"]
    vfeature_func = model["vfeature_func"]
    
    predictions = []
    
    for original_sentence in sentence_gen(in_file):
        # Filter unknown words.
        sentence = [word.casefold() for word in original_sentence]
        sentence = [word if (word in wordcount) else "#UNK#" for word in sentence]
        
        # row: index of sentence, col: state/tag.
        # Value is max log probability for that word to be that tag.
        viterbi_table = np.empty((len(sentence), len(all_y)))
        # Backtracking pointer. Holds the previous state the resulted in cell's value. 
        viterbi_backtrack = np.empty((len(sentence), len(all_y)), dtype=np.int32)
        # Initial step.
        i = 0
        for j, u in enumerate(all_y):
            viterbi_table[i,j] = np.dot(w, vfeature_func("START", u, 0, sentence))
        # subsequent steps
        for i in range(1, len(sentence)):
            for j, u in enumerate(all_y):
                possibilities = [viterbi_table[i-1,k]
                                 + np.dot(w, vfeature_func(v, u, i, sentence))
                                 for k, v in enumerate(all_y)
                                ]
                viterbi_table[i,j] = np.max(possibilities)
                viterbi_backtrack[i,j] = np.argmax(possibilities)
        # last step.
        i = len(sentence)
        final_possibilities = [viterbi_table[i-1,k]
                               + np.dot(w, vfeature_func(v, "STOP", None, sentence))
                               for k, v in enumerate(all_y)
                              ]
        max_state_index = np.argmax(final_possibilities)
        
        # backtracking.
        predicted_y = [None for word in sentence]
        predicted_y[-1] = max_state_index
        for i in range(len(sentence)-2, -1, -1):
            next_state_idx = predicted_y[i+1]
            curr_max_state_idx = viterbi_backtrack[i+1, next_state_idx]
            predicted_y[i] = curr_max_state_idx
        predicted_y = [all_y[idx] for idx in predicted_y]
        predicted_y.reverse()
        
        if debug:
            pprint.pprint(original_sentence)
            pprint.pprint(sentence)
            pprint.pprint(predicted_y)
        
        predictions.append((original_sentence, predicted_y))
    
    return predictions


def main(args):
    train_path = os.path.join(args.folder, args.train)
    with open(train_path, encoding="utf-8") as training_file:
        model = train(training_file)
    
    infile_path = os.path.join(args.folder, args.infile)
    with open(infile_path, encoding="utf-8") as in_file:
        predictions = predict(model, in_file)
    
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
    import pprint # DEBUG
    
    start_time = time.time()
    main(args)
    print("{}s".format(time.time()-start_time))