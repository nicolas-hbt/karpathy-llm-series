"""
Implements the Tokenizer class, which is the base class. 
It contains the train, encode, and decode stubs, save/load functionality, and there are also a few common utility functions. 
This class is not meant to be used directly, but rather to be inherited from.

It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

def get_stats(ids, counts=None):
    """
    Count consecutive token pairs in an integer sequence.

    Args:
        ids: Sequence of token ids.
        counts: Optional dictionary to update in-place, mapping
            `(id_i, id_{i+1}) -> frequency`.

    Returns:
        A dictionary mapping consecutive id pairs to their counts.

    Example:
        [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    Replace every consecutive occurrence of a pair with a merged token id.

    Args:
        ids: Sequence of token ids.
        pair: Consecutive id pair to merge, as `(left_id, right_id)`.
        idx: Token id to emit when `pair` is found.

    Returns:
        A new list of token ids where each non-overlapping occurrence of
        `pair` is replaced with `idx`.

    Example:
        ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# first two helper functions...
def replace_control_characters(s: str) -> str:
    """
    Escape Unicode control characters in a string for readable display.

    Args:
        s: Input string that may contain control characters.

    Returns:
        A string where control characters are replaced by `\\uXXXX` escapes.
    """
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    """
    Render a byte token as a printable string.

    Args:
        t: Raw token bytes.

    Returns:
        A UTF-8 decoded string (with replacement for invalid byte sequences)
        where control characters are escaped.
    """
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """
    Base class for tokenizers.

    Subclasses are expected to implement training and text/id conversion, i.e. encode/decode.
    """

    def __init__(self):
        """Initialize tokenizer state with byte-level defaults."""
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        """
        Train tokenizer merges from text.

        Args:
            text: Training corpus as a string.
            vocab_size: Target vocabulary size after training.
            verbose: If `True`, print training progress.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        """
        Encode text into token ids.

        Args:
            text: Input string.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        """
        Decode token ids back into text.

        Args:
            ids: Sequence of token ids.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        """
        Build the vocabulary mapping from byte tokens, merges, and specials.

        Returns:
            A dictionary mapping token id to token bytes.
        """
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Save tokenizer state to disk as model and vocab files.

        Args:
            file_prefix: Output prefix. Writes `{file_prefix}.model` and
                `{file_prefix}.vocab`.

        Notes:
            The `.model` file is the source of truth for `load()`.
            The `.vocab` file is for human inspection only.
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """
        Load tokenizer state from a `.model` file.

        Args:
            model_file: Path to a file produced by `save()` with suffix
                `.model`.

        Notes:
            This restores merges, special tokens, and the derived vocabulary.
        """
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()