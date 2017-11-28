import argparse
import os.path
from fractions import Fraction

all_y = ("O", "B-positive", "B-neutral", "B-negative", "I-positive", "I-neutral", "I-negative")
all_y_ss = all_y + ("START", "STOP")
unk_threshold = 3 # k. If word appears less than this frequency, then word is treated as unknown.

def train(training_file):
    # counts_e[x][y] is emission of (tag y)->(word x)
    counts_e = {}
    # raw count of tags
    counts_y = {y:0 for y in all_y_ss}
    # count emissions by line (word)
    for line in training_file:
        line = line.strip()
        if not line:
            continue
        x, y = line.rsplit(" ", 1) # word, tag
        if x not in counts_e:
            counts_e[x] = {y:0 for y in all_y}
        counts_e[x][y] += 1
        counts_y[y] += 1
    
    # Accumulate low frequency words into #UNK#.
    to_unk = []
    counts_unk = {y:0 for y in all_y}
    for x, y_to_x in counts_e.items():
        if sum(y_to_x.values()) < unk_threshold:
            to_unk.append(x)
            for y, count in y_to_x.items(): # Add counts to #UNK#
                counts_unk[y] += count
    for x in to_unk:
        del counts_e[x]
    counts_e["#UNK#"] = counts_unk
    
    # Emission probabilities
    e = {}
    for x, y_to_x in counts_e.items():
        e[x] = {}
        for y, count in y_to_x.items():
            e[x][y] = Fraction(count, counts_y[y])
    
    return e

def predict(e, in_file):
    """Generate list of tuples: [(word, prediction), ...]"""
    predictions = []
    for line in in_file:
        x = line.strip()
        if not x:
            predictions.append(None) # Account for empty lines between sentences
            continue
        if x in e:
            max_y_prob = max(e[x].items(), key=lambda y_prob:y_prob[1])
        else:
            max_y_prob = max(e["#UNK#"].items(), key=lambda y_prob:y_prob[1])
        y = max_y_prob[0]
        predictions.append((x,y))
    return predictions

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
    parser.add_argument('-o', '--outfile', type=str, default="dev.p2.out", help='Output (the predictions) file')
    parser.add_argument('-f', '--folder', type=str, default=".", help='Folder containing files (prepended to files).')
    args = parser.parse_args()
    
    main(args)