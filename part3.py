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
    # counts_e[x][y] is emission of (tag y)->(word x)
    counts_e = {}
    # counts_t[next][prev] is transition from (prev)->(next)
    counts_t = {n : {p:0 for p in all_y+("START",)}
                for n in all_y+("STOP",)}
    # raw count of tags
    counts_y = {y:0 for y in all_y_ss}
    
    for sentence in sentence_gen(training_file):
        counts_y["START"] += 1
        prev_y = "START"
        for pair in sentence:
            x, y = pair.rsplit(" ", 1) # word, tag
            if x not in counts_e:
                counts_e[x] = {y:0 for y in all_y}
            counts_e[x][y] += 1
            counts_t[y][prev_y] += 1
            counts_y[y] += 1
            prev_y = y
        counts_y["STOP"] += 1
        counts_t["STOP"][prev_y] += 1
    
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
    
    e = {}
    for x, y_to_x in counts_e.items():
        e[x] = {}
        for y, count in y_to_x.items():
            e[x][y] = Fraction(count, counts_y[y])
    
    t = {}
    for y_next, prev_to_next in counts_t.items():
        t[y_next] = {}
        for y_prev, count in prev_to_next.items():
            t[y_next][y_prev] = Fraction(count, counts_y[y_prev])
    
    return e, t
    # should return emission parameters and transition parameters.

def predict(e, t, in_file):
    predictions = []
    
    for sentence in sentence_gen(in_file):
        # insert viterbi algorithm here
        pass
    
    return predictions


def main(args):
    train_path = os.path.join(args.folder, args.train)
    with open(train_path, encoding="utf-8") as training_file:
        e, t = train(training_file)
    
    # DEBUG
    import pprint
    pprint.pprint(e)
    pprint.pprint(t)
    
    infile_path = os.path.join(args.folder, args.infile)
    with open(infile_path, encoding="utf-8") as in_file:
        predictions = predict(e, t, in_file)
    
    """
    outfile_path = os.path.join(args.folder, args.outfile)
    with open(outfile_path, "w", encoding="utf-8") as out_file:
        str_predictions = [(" ".join(pair) if pair is not None else "") for pair in predictions]
        str_predictions = "\n".join(str_predictions)
        out_file.write(str_predictions)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=str, default="train", help='Training dataset file')
    parser.add_argument('-i', '--infile', type=str, default="dev.in", help='Input (to be decoded) dataset file')
    parser.add_argument('-o', '--outfile', type=str, default="dev.prediction", help='Input (to be decoded) dataset file')
    parser.add_argument('-f', '--folder', type=str, default=".", help='Folder containing files (prepended to files).')
    args = parser.parse_args()
    
    main(args)