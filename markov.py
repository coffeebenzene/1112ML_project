import argparse

def train(training_file):
    pass
    # should return emission parameters and transition parameters.

def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('TRAIN', type=str, default="train", help='Training dataset file')
    parser.add_argument('INFILE', type=str, default="dev.in", help='Input (to be decoded) dataset file')
    parser.add_argument('OUTFILE', type=str, default="dev.prediction", help='Input (to be decoded) dataset file')
    args parser.parse_args()
    
    main(args)