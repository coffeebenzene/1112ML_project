import argparse

def train(training_file):
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