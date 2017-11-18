import argparse
import os.path
from fractions import Fraction

all_y = ("O", "B-positive", "B-neutral", "B-negative", "I-positive", "I-neutral", "I-negative")
all_y_ss = all_y + ("START", "STOP")
unk_threshold = 3 # k. If word appears less than this frequency, then word is treated as unknown.

def sentence_gen(f):
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

def train(training_file):
    counts_e = {}
    counts_t = {}
    counts_y = {y:0 for y in all_y_ss}
    in_sentence = False
    prev_tag = None
    for line in training_file:
        line = line.strip()
        if not line: # next line is blank
            # if in_sentence: # previous line is part of a sentence
                #calc stop
            in_sentence = False
            continue

        x, y = line.rsplit(" ", 1) # word, tag
        if x not in counts_e:
            counts_e[x] = {y:0 for y in all_y}
        counts_e[x][y] += 1
        counts_y[y] += 1
        prev_tag = y

    
    to_unk = []
    counts_unk = {y:0 for y in all_y}
    for x, y_to_x in counts_e.items():
        if sum(y_to_x.values()) < unk_threshold:
            to_unk.append(x)
        #addup to counts_unk
    return None, None
    # should return emission parameters and transition parameters.



def main(args):
    train_path = os.path.join(args.folder, args.train)
    with open(train_path, encoding="utf-8") as training_file:
        e = train(training_file)
    
    infile_path = os.path.join(args.folder, args.infile)
    with open(infile_path, encoding="utf-8") as in_file:
        predictions = predict(e, in_file)
    
    outfile_path = os.path.join(args.folder, args.outfile)
    with open(outfile_path, "w", encoding="utf-8") as out_file:
        str_predictions = [(" ".join(pair) if pair is not None else "") for pair in predictions]
        str_predictions = "\n".join(str_predictions)
        out_file.write(str_predictions)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=str, default="train", help='Training dataset file')
    parser.add_argument('-i', '--infile', type=str, default="dev.in", help='Input (to be decoded) dataset file')
    parser.add_argument('-o', '--outfile', type=str, default="dev.prediction", help='Input (to be decoded) dataset file')
    parser.add_argument('-f', '--folder', type=str, default=".", help='Folder containing files (prepended to files).')
    args = parser.parse_args()
    
    main(args)