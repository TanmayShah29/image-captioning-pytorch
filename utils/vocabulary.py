"""
Vocabulary Builder for Image Captioning

This module creates a vocabulary from captions and handles word-to-index conversions.

Key Concepts:
- Vocabulary: Set of all unique words in the dataset
- Special Tokens: <pad>, <start>, <end>, <unk>
- Word-to-Index: Maps words to numerical indices for neural network
- Index-to-Word: Maps indices back to words for caption generation

Why needed: Neural networks work with numbers, not text. We need to convert
words to indices before feeding them to the model.
"""

import pickle
from collections import Counter


class Vocabulary:
    """
    Builds and manages vocabulary for image captions.
    
    Attributes:
        word2idx (dict): Maps words to indices
        idx2word (dict): Maps indices to words
        idx (int): Current index counter
    """
    
    def __init__(self, freq_threshold=5):
        """
        Initialize vocabulary with special tokens.
        
        Args:
            freq_threshold (int): Minimum word frequency to include in vocabulary.
                                 Words appearing less than this are treated as <unk>
        
        Why freq_threshold: Rare words are replaced with <unk> to reduce vocabulary
        size and prevent overfitting on uncommon words.
        """
        self.freq_threshold = freq_threshold
        
        # Special tokens explained:
        # <pad>: Padding token to make all captions same length
        # <start>: Marks beginning of caption
        # <end>: Marks end of caption
        # <unk>: Unknown word (not in vocabulary)
        self.word2idx = {
            "<pad>": 0,
            "<start>": 1,
            "<end>": 2,
            "<unk>": 3
        }
        
        # Reverse mapping: index to word
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Start index after special tokens
        self.idx = 4
    
    def __len__(self):
        """
        Returns the size of vocabulary.
        
        Returns:
            int: Total number of words in vocabulary
        """
        return len(self.word2idx)
    
    def build_vocabulary(self, caption_list):
        """
        Build vocabulary from list of captions.
        
        Args:
            caption_list (list): List of caption strings
            
        Process:
            1. Tokenize all captions (split into words)
            2. Count word frequencies
            3. Add words that appear >= freq_threshold times
            4. Ignore rare words (they'll be treated as <unk>)
        
        Example:
            captions = ["a dog running", "a cat sitting", "a dog playing"]
            vocab.build_vocabulary(captions)
            # "a" and "dog" appear frequently, added to vocab
            # Other words added if they meet threshold
        """
        frequencies = Counter()
        
        # Count word frequencies across all captions
        for caption in caption_list:
            # Tokenize: split caption into words and convert to lowercase
            tokens = self.tokenize(caption)
            frequencies.update(tokens)
        
        # Add words that meet frequency threshold
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
        
        print(f"Vocabulary built with {len(self.word2idx)} words")
        print(f"Frequency threshold: {self.freq_threshold}")
    
    def tokenize(self, text):
        """
        Convert text to lowercase tokens (words).
        
        Args:
            text (str): Input caption
            
        Returns:
            list: List of lowercase words
            
        Example:
            tokenize("A Dog Running") -> ["a", "dog", "running"]
        """
        # Simple tokenization: lowercase and split by spaces
        # Remove punctuation for cleaner tokens
        text = text.lower().replace(',', '').replace('.', '').replace('!', '').replace('?', '')
        return text.split()
    
    def numericalize(self, text):
        """
        Convert text to list of indices.
        
        Args:
            text (str): Input caption
            
        Returns:
            list: List of word indices
            
        Process:
            1. Tokenize text into words
            2. Look up each word in word2idx
            3. If word not found, use <unk> index
            
        Example:
            vocab.word2idx = {"<unk>": 3, "a": 4, "dog": 5}
            numericalize("a dog") -> [4, 5]
            numericalize("a cat") -> [4, 3]  # "cat" not in vocab, becomes <unk>
        """
        tokens = self.tokenize(text)
        
        # Convert each word to its index
        # Use <unk> index if word not in vocabulary
        return [
            self.word2idx.get(token, self.word2idx["<unk>"])
            for token in tokens
        ]
    
    def denumericalize(self, indices):
        """
        Convert list of indices back to text.
        
        Args:
            indices (list): List of word indices
            
        Returns:
            str: Caption text
            
        Used during inference to convert model predictions to readable text.
        
        Example:
            vocab.idx2word = {4: "a", 5: "dog"}
            denumericalize([4, 5]) -> "a dog"
        """
        # Look up each index and join words with spaces
        words = [self.idx2word.get(idx, "<unk>") for idx in indices]
        return " ".join(words)
    
    def save_vocabulary(self, filepath):
        """
        Save vocabulary to file for later use.
        
        Args:
            filepath (str): Path to save vocabulary
            
        Why needed: After building vocabulary during training, we need to save it
        to use the same vocabulary during inference.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Vocabulary saved to {filepath}")
    
    @staticmethod
    def load_vocabulary(filepath):
        """
        Load vocabulary from file.
        
        Args:
            filepath (str): Path to vocabulary file
            
        Returns:
            Vocabulary: Loaded vocabulary object
        """
        with open(filepath, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded from {filepath}")
        return vocab


# Example usage (for testing)
if __name__ == "__main__":
    # Sample captions
    sample_captions = [
        "a dog is running in the park",
        "a cat is sitting on the mat",
        "a dog is playing with a ball",
        "children are playing in the park"
    ]
    
    # Create vocabulary
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(sample_captions)
    
    # Test numericalization
    test_caption = "a dog is running"
    indices = vocab.numericalize(test_caption)
    print(f"\nOriginal: {test_caption}")
    print(f"Indices: {indices}")
    print(f"Reconstructed: {vocab.denumericalize(indices)}")
    
    # Show vocabulary
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Sample words: {list(vocab.word2idx.keys())[:10]}")
