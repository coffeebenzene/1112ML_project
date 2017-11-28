import argparse
import os.path
from fractions import Fraction

debug = False
if debug:
    import pprint

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
    # Transition probabilities
    t = {}
    for y_next, prev_to_next in counts_t.items():
        t[y_next] = {}
        for y_prev, count in prev_to_next.items():
            t[y_next][y_prev] = Fraction(count, counts_y[y_prev])
    
    return e, t
    # should return emission parameters and transition parameters.

def predict(e, t, in_file):
    """Generate list of tuples: [(sentence, prediction), ...]"""
    predictions = []
    
    for sentence in sentence_gen(in_file):
        viterbi_table = [] # by index, each element is dict of {tags : probabilities}
        # Initial step.
        d = {}
        word_emissions = e.get(sentence[0])
        if word_emissions is None:
            word_emissions = e["#UNK#"]
        for v in all_y:
            d[v] = t[v]["START"] * word_emissions[v]
        viterbi_table.append(d)
        # subsequent steps
        for i, word in enumerate(sentence[1:], 1):
            d = {}
            word_emissions = e.get(word)
            if word_emissions is None:
                word_emissions = e["#UNK#"]
            for v in all_y:
                d[v] = max( viterbi_table[i-1][u] * t[v][u] for u in all_y ) * word_emissions[v]
            viterbi_table.append(d)
        
        if debug:
            pprint.pprint(viterbi_table)
        
        predicted_y = [] # reversed order first
        # backtracking y prediction
        next_y = "STOP"
        for i in range(len(sentence)-1, -1, -1):
            # Using tuples with tag in 2nd index.
            # break even by alphabetical order of tags ("O" is prioritized, then "I", then "B")
            prob, v = max( (viterbi_table[i][v] * t[next_y][v], v) for v in all_y )
            next_y = v
            predicted_y.append(v)
        predicted_y.reverse()
        
        if debug:
            pprint.pprint(sentence)
            pprint.pprint(predicted_y)
        
        predictions.append((sentence, predicted_y))
    
    return predictions


def main(args):
    train_path = os.path.join(args.folder, args.train)
    with open(train_path, encoding="utf-8") as training_file:
        e, t = train(training_file)
    
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=str, default="train", help='Training dataset file')
    parser.add_argument('-i', '--infile', type=str, default="dev.in", help='Input (to be decoded) dataset file')
    parser.add_argument('-o', '--outfile', type=str, default="dev.p3.out", help='Output (the predictions) file')
    parser.add_argument('-f', '--folder', type=str, default=".", help='Folder containing files (prepended to files).')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    debug = args.debug
    if debug:
        import pprint
    
    main(args)