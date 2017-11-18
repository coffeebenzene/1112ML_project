import argparse

all_y = ("O", "B-positive", "B-neutral", "B-negative", "I-positive", "I-neutral", "I-negative")
all_y_ss = all_y + ("START", "STOP")
unk_threshold = 3 # k. If word appears less than this frequency, then word is treated as unknown.

def train(training_file):
    counts_e = {}
    counts_y = {y:0 for y in all_y_ss}
    for line in training_file:
        if not line:
            continue
        x, y = line.split(" ") # word, tag
        if x not in counts_e:
            counts_e[x] = {y:0 for y in all_y}
        counts_e[x][y] += 1
        counts_y[y] += 1
    
    to_unk = []
    counts_unk = {y:0 for y in all_y}
    for x, y_to_x in counts_e.items():
        if sum(y_to_x.values()) < unk_threshold:
            to_unk.append(x)
        #addup to counts_unk
    return None, None
    # should return emission parameters and transition parameters.



def main(args):
    with open(args.train) as training_file:
        e, t = train(training_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=str, default="train", help='Training dataset file')
    parser.add_argument('-i', '--infile', type=str, default="dev.in", help='Input (to be decoded) dataset file')
    parser.add_argument('-o', '--outfile', type=str, default="dev.prediction", help='Input (to be decoded) dataset file')
    args = parser.parse_args()
    
    main(args)