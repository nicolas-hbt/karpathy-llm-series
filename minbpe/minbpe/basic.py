"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Implements the BasicTokenizer, the simplest implementation of the BPE algorithm that runs directly on text.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids) # we need to repeat this every time since the ids are changing after every merge
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            # because lowest merge index means it was merged earlier, which means it is a more fundamental merge
            stats = get_stats(ids)
            # explanation for the line below, pair = ...
            # How the key function works
            # The key=lambda p: self.merges.get(p, float("inf")) parameter tells min() how to score each pair:
                # 1. For each pair p in stats, the lambda function looks it up in self.merges
                # 2. If the pair exists in self.merges, it returns its assigned token ID (a positive integer)
                # 3. If the pair does not exist in self.merges, it returns float("inf") (infinity) as a fallback
            # The selection logic
            # The min() function then compares these scores across all pairs and returns the pair with the lowest score. This creates a **ranking preference**
            # Pairs already in self.merges get their token IDs as scores (typically 256 and above)
            # Pairs not yet merged get float("inf") as their score
            # The KEY here is to observe that token IDs can be seen as scores
            # i.e., among the already-merged pairs, it selects the one with the smallest token ID—meaning the pair that was merged earliest in the training process.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids