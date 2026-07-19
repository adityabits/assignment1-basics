import os
import argparse
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import pickle

import numpy.typing as npt
import regex as re
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.tokenizer import train_bpe
from cs336_basics.utils import DATA_PATH, OUTPUT_PATH

DATASETS = {
    "owt_valid": "owt_valid.txt",
    "owt_train": "owt_train.txt",
    "ts_valid": "TinyStoriesV2-GPT4-valid.txt",
    "ts_train": "TinyStoriesV2-GPT4-train.txt",
}

def main(args):
    if args.dataset is None:
        dataset = "ts_valid"
    else:
        dataset = args.dataset

    path = DATA_PATH / DATASETS[dataset]
    print(path)

    vocab, merges = train_bpe(path, 1000, ['<|endoftext|>'])

    vocab_out = OUTPUT_PATH / (dataset + "vocab")
    with open(vocab_out, 'wb') as f:
        pickle.dump(vocab, f)
    
    merges_out = OUTPUT_PATH / (dataset + "merges")
    with open(merges_out, 'wb') as f:
        pickle.dump(merges, f)
    

    # test
    with open(vocab_out, 'rb') as f:
        read_vocab = pickle.load(f)
        print("Read vocab", read_vocab)


def build_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    return parser

if __name__ == "__main__":
    args = build_parse().parse_args()
    main(args)