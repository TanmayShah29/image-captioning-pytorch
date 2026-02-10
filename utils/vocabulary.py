"""
Vocabulary Builder for Image Captioning

Builds a word↔index mapping from captions with frequency-based filtering.

Special tokens:
    <pad>   = 0   Padding for fixed-length batches
    <start> = 1   Beginning of caption
    <end>   = 2   End of caption
    <unk>   = 3   Unknown / rare word
"""

import pickle
from collections import Counter


class Vocabulary:
    """Builds and manages vocabulary for image captions."""

    SPECIAL_TOKENS = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}

    def __init__(self, freq_threshold=5):
        """
        Args:
            freq_threshold: minimum word count to be included (else → <unk>).
        """
        self.freq_threshold = freq_threshold
        self.word2idx = dict(self.SPECIAL_TOKENS)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx = len(self.SPECIAL_TOKENS)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.word2idx)

    def build_vocabulary(self, caption_list):
        """Build vocab from a list of caption strings.

        Raises:
            ValueError: if no words meet the frequency threshold.
        """
        if not caption_list:
            raise ValueError("Cannot build vocabulary from an empty caption list.")

        frequencies = Counter()
        for caption in caption_list:
            frequencies.update(self.tokenize(caption))

        added = 0
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
                added += 1

        # ---------- post-build assertions ----------
        assert "<start>" in self.word2idx, "Missing <start> token"
        assert "<end>" in self.word2idx, "Missing <end> token"
        assert "<unk>" in self.word2idx, "Missing <unk> token"
        assert len(self) > len(self.SPECIAL_TOKENS), (
            f"Vocabulary only contains special tokens ({len(self)} words). "
            f"Lower freq_threshold (currently {self.freq_threshold}) or add more captions."
        )

        print(f"Vocabulary built: {len(self)} words  "
              f"(freq_threshold={self.freq_threshold}, {added} content words)")

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize(text):
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        for ch in ",.!?;:\"'()[]{}":
            text = text.replace(ch, "")
        return text.split()

    def numericalize(self, text):
        """Convert text → list of indices (unknown words → <unk>)."""
        unk = self.word2idx["<unk>"]
        return [self.word2idx.get(tok, unk) for tok in self.tokenize(text)]

    def denumericalize(self, indices):
        """Convert list of indices → string."""
        return " ".join(self.idx2word.get(i, "<unk>") for i in indices)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_vocabulary(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Vocabulary saved → {filepath}")

    @staticmethod
    def load_vocabulary(filepath):
        with open(filepath, "rb") as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded ← {filepath}  ({len(vocab)} words)")
        return vocab


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    captions = [
        "a dog is running in the park",
        "a cat is sitting on the mat",
        "a dog is playing with a ball",
        "children are playing in the park",
    ]
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(captions)

    test = "a dog is running"
    idx = vocab.numericalize(test)
    print(f"\n'{test}' → {idx} → '{vocab.denumericalize(idx)}'")
    print(f"Vocab size: {len(vocab)}")
