import os
from collections.abc import Iterable, Iterator
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import pickle
import regex as re
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from typing import BinaryIO
from tqdm import tqdm

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_chunk(text: str, special_tokens: list[str]) -> dict[tuple[bytes,...], int]:

    special_token_reg = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
    #special_token_reg = "|".join(special_tokens)
    pre_tokens = {}
    for chunk in re.split(special_token_reg, text):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(PAT, chunk):
            # convert a str into tuple[bytes]
            t = tuple(bytes([b]) for b in match.group().encode('utf-8'))
            pre_tokens[t] = pre_tokens.get(t, 0) + 1
    return pre_tokens


def pretokenize(input_path: str | os.PathLike, 
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    vocab_set = {bytes([i]) for i in range(256)} | {t.encode('utf-8') for t in special_tokens}
    pre_tokens = {tuple([b]): 0 for b in vocab_set} # dict[tuple(bytes, ...), int]
    
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            pre_chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pt = pretokenize_chunk(pre_chunk, special_tokens)
            for t, count in pt:
                pre_tokens[t] = pre_tokens.get(t, 0) + count

    return pre_tokens

def merge_pt(pt: tuple[bytes, ...], max_merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_pt_list = []
    new_token = b''.join(list(max_merge))
    found = 0
    i = 0
    while i < len(pt):
        if i < len(pt) - 1 and max_merge == (pt[i], pt[i+1]):
            found = 1
            new_pt_list.append(new_token)
            i = i + 1
        else:
            new_pt_list.append(pt[i])
        i = i + 1 
    return tuple(new_pt_list)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Pre-tokenization
    vocab = {}
    vocab_set = {bytes([i]) for i in range(256)} | {t.encode('utf-8') for t in special_tokens}
    for i, t in enumerate(vocab_set):
        vocab[i] = t

    merges = []
    pre_tokens = pretokenize(input_path, special_tokens)
    
    count_map = {} # dict[tuple(bytes, bytes), int]
    inv_index = {} # dict[tuple(bytes, bytes), set(tuple(bytes, ...)))]
    for pt in pre_tokens.keys():
        pt_list = list(pt)
        for pt_merges in zip(pt_list[:-1], pt_list[1:]):
            count_map[pt_merges] = count_map.get(pt_merges, 0) + pre_tokens[pt]
            inv_index.setdefault(pt_merges, set()).add(pt)
    print("Index", inv_index[(b' ', b',')])
    deleted_pre_tokens = []
    add_pt = {}

    #print("Inv index", len(inv_index))

    
    # Merge
    num_merges = vocab_size - len(vocab.keys()) 
    for _ in tqdm(range(num_merges), desc="merges"):
        # Find most common merge based on alphabetical order
        max_merge = max(count_map, key=lambda k: (count_map[k], k)) # tuple(bytes, bytes)
        print("Max merge", max_merge)
        new_token = b''.join(list(max_merge))
        merges.append(max_merge)

        pt_updates = {}   
        for t in vocab.values():
            old_pair = (max_merge[1], t)
            for pt in inv_index.get(old_pair, []):
                pt_updates[pt] = merge_pt(pt, max_merge)
            
            old_pair = (t, max_merge[0])
            for pt in inv_index.get(old_pair, []):
                pt_updates[pt] = merge_pt(pt, max_merge)
            
            for pt in inv_index.get(max_merge, []):
                pt_updates[pt] = merge_pt(pt, max_merge)
        
        #print("Index contents", inv_index[(b' ', b',')])
        #print("Udates", len(pt_updates))
        # Update pre_tokens first
        for old_pt, new_pt in pt_updates.items():
            val = pre_tokens[old_pt]
            #print("Updating", val, max_merge, old_pt, new_pt)
            # update count map, inv index
            for pair in zip(old_pt[:-1], old_pt[1:]):
                count_map[pair] = count_map.get(pair, 0) - val
                if pair in inv_index:
                    inv_index[pair].discard(old_pt)
            for pair in zip(new_pt[:-1], new_pt[1:]):
                count_map[pair] = count_map.get(pair, 0) + val 
                inv_index.setdefault(pair, set()).add(new_pt)
            # update inv index
        
            del pre_tokens[old_pt]
            pre_tokens[new_pt] = val
        # add to vocab
        vocab[len(vocab.keys())] = new_token

    return (vocab, merges)




class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = []) -> None:
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges 
        self.special_tokens = special_tokens if special_tokens else []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=[]) -> None:
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    
    def encode(self, text:str) -> list[int]:
        encoded = []
        # Pretokenize
        special_token_reg = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        if len(self.special_tokens) > 0:
            parts = re.split(special_token_reg, text)
        else:
            parts = [text]
        pt_list = [] # list[tuple[bytes, ...]]
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        for piece in parts:
            if piece in self.special_tokens:
                pt = (piece.encode('utf-8'), )
                pt_list.append(pt)

            else:
                for match in re.finditer(PAT, piece):
                    # convert a str into tuple[bytes]
                    pt = tuple(bytes([b]) for b in match.group().encode('utf-8'))
                    pt_list.append(pt)

        # brute force: O(len(merges) X len(text))
        for pt in pt_list:
            # go through merges and iteratively merge pt
            for merge in self.merges:
                pt = merge_pt(pt, merge)
            # encode fully merged pt
            for t in pt:
                encoded.append(self.inv_vocab[t])
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)

    def decode(self, tokens: list[int]) -> str:
        bytelist = [self.vocab[t] for t in tokens] # list[bytes]
        return "".join([b.decode('utf-8', errors='ignore') for b in bytelist])