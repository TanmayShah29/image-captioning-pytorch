"""
LSTM Decoder for Image Captioning

This module generates captions word-by-word from image features.

Key Concepts:
- LSTM: Long Short-Term Memory network for sequence generation
- Word Embeddings: Dense vector representations of words
- Sequential Generation: Predict one word at a time
- Teacher Forcing: Use actual previous word during training

Architecture:
    Image Features (embed_size)
        ↓
    Word Embeddings (vocab_size → embed_size)
        ↓
    LSTM (embed_size → hidden_size)
        ↓
    Fully Connected (hidden_size → vocab_size)
        ↓
    Softmax → Predicted Word

Why LSTM?
- Designed for sequential data (sentences)
- Remembers long-term dependencies
- Avoids vanishing gradient problem
- Better than simple RNN for language tasks
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    LSTM-based caption decoder.
    
    What does this do?
    - Takes image features as input
    - Generates caption word-by-word
    - Uses LSTM to maintain context
    
    Attributes:
        embed (nn.Embedding): Word embedding layer
        lstm (nn.LSTM): LSTM network
        linear (nn.Linear): Output layer
        dropout (nn.Dropout): Regularization
    """
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        """
        Initialize decoder.
        
        Args:
            embed_size (int): Dimension of word embeddings
                             Should match encoder output size
            hidden_size (int): LSTM hidden state size
                              Controls model capacity
            vocab_size (int): Size of vocabulary
                            Number of possible output words
            num_layers (int): Number of LSTM layers
                            Default: 1 (simple for beginners)
            dropout (float): Dropout probability for regularization
                           Prevents overfitting
        
        Layer Dimensions:
            Embedding: vocab_size → embed_size
            LSTM: embed_size → hidden_size
            Linear: hidden_size → vocab_size
        
        Example:
            vocab_size=5000, embed_size=256, hidden_size=512
            - Embedding: 5000 → 256
            - LSTM: 256 → 512
            - Linear: 512 → 5000
        """
        super(Decoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Word embedding layer
        # Converts word indices to dense vectors
        # Why? Neural networks work better with dense representations
        # Example: word index 42 → 256-dimensional vector
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        # Input: embed_size (word embeddings)
        # Output: hidden_size (hidden states)
        # batch_first=True: input shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout for regularization
        # Randomly sets some activations to zero during training
        # Prevents overfitting
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        # Maps LSTM output to vocabulary size
        # Output: probability distribution over all words
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights for better training
        self.init_weights()
        
        print(f"Decoder initialized:")
        print(f"  Embed size: {embed_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Num layers: {num_layers}")
    
    def init_weights(self):
        """
        Initialize weights for better training.
        
        Why initialize?
        - Random initialization can lead to slow training
        - Proper initialization helps model converge faster
        
        Initialization strategy:
        - Embedding: uniform distribution [-0.1, 0.1]
        - Linear: uniform distribution [-0.1, 0.1]
        """
        # Initialize embedding weights
        self.embed.weight.data.uniform_(-0.1, 0.1)
        
        # Initialize linear layer weights and bias
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
    
    def forward(self, features, captions):
        """
        Forward pass during training (with teacher forcing).
        
        Args:
            features (Tensor): Image features from encoder
                             Shape: (batch_size, embed_size)
            captions (Tensor): Target captions (word indices)
                             Shape: (batch_size, max_length)
        
        Returns:
            outputs (Tensor): Predicted word scores
                            Shape: (batch_size, max_length, vocab_size)
        
        Teacher Forcing Explained:
        - During training, we feed the ACTUAL previous word
        - Not the predicted word
        - This helps model learn faster
        - Example: If actual caption is "a dog running"
          - Step 1: Input <start> → Predict "a"
          - Step 2: Input "a" (actual) → Predict "dog"
          - Step 3: Input "dog" (actual) → Predict "running"
        
        Process:
            1. Remove last word from captions (don't need to predict after <end>)
            2. Convert words to embeddings
            3. Concatenate image features with word embeddings
            4. Pass through LSTM
            5. Apply dropout
            6. Project to vocabulary size
        
        Shape Transformations:
            features: (batch, embed_size)
            captions: (batch, max_length)
            embeddings: (batch, max_length-1, embed_size)
            features_unsqueezed: (batch, 1, embed_size)
            lstm_input: (batch, max_length, embed_size)
            lstm_output: (batch, max_length, hidden_size)
            outputs: (batch, max_length, vocab_size)
        """
        # Remove last word from captions
        # We predict words 1 to N given words 0 to N-1
        # Example: Given [<start>, a, dog], predict [a, dog, <end>]
        captions = captions[:, :-1]
        
        # Convert word indices to embeddings
        # Shape: (batch_size, caption_length, embed_size)
        embeddings = self.embed(captions)
        
        # Reshape features to match embedding dimensions
        # (batch_size, embed_size) → (batch_size, 1, embed_size)
        # This allows concatenation with word embeddings
        features = features.unsqueeze(1)
        
        # Concatenate image features with word embeddings
        # Image features act as the first "word" in the sequence
        # Shape: (batch_size, caption_length + 1, embed_size)
        # Example: [image_features, word1, word2, word3, ...]
        lstm_input = torch.cat((features, embeddings), dim=1)
        
        # Pass through LSTM
        # lstm_output: (batch_size, seq_length, hidden_size)
        # hidden: final hidden state (not used here)
        lstm_output, _ = self.lstm(lstm_input)
        
        # Apply dropout for regularization
        lstm_output = self.dropout(lstm_output)
        
        # Project to vocabulary size
        # (batch_size, seq_length, hidden_size) → (batch_size, seq_length, vocab_size)
        outputs = self.linear(lstm_output)
        
        # Remove first position (image features position) to align with targets
        # After concatenation, lstm_output has shape (batch, max_length, hidden_size)
        # where position 0 is image features, positions 1+ are word predictions
        # Targets are captions[:, 1:] which has length max_length-1
        # So we need to remove position 0 from outputs
        outputs = outputs[:, 1:, :]  # (batch, max_length-1, vocab_size)
        
        return outputs
    
    def generate_caption(self, features, vocab, max_length=50):
        """
        Generate caption for a single image (inference mode).
        
        Args:
            features (Tensor): Image features from encoder
                             Shape: (1, embed_size)
            vocab (Vocabulary): Vocabulary object for word conversion
            max_length (int): Maximum caption length
        
        Returns:
            caption (list): List of words in generated caption
        
        Greedy Decoding Explained:
        - Start with <start> token
        - At each step:
          1. Feed current word to LSTM
          2. Get probability distribution over vocabulary
          3. Pick word with highest probability (greedy)
          4. Use it as input for next step
        - Stop when <end> token is generated or max_length reached
        
        Example:
            Step 1: Input <start> → Predict "a" (highest prob)
            Step 2: Input "a" → Predict "dog" (highest prob)
            Step 3: Input "dog" → Predict "running" (highest prob)
            Step 4: Input "running" → Predict <end> (stop)
            Result: "a dog running"
        
        Why greedy?
        - Simple and fast
        - Good enough for most cases
        - Alternatives: beam search (more complex)
        """
        caption = []
        hidden = None
        
        # FIRST STEP: Concatenate image features with <start> token
        # This matches the training setup where image features are prepended
        start_token = torch.tensor([vocab.word2idx["<start>"]]).unsqueeze(0)
        start_token = start_token.to(features.device)
        start_embedding = self.embed(start_token)
        
        # Initial LSTM input: [image_features, <start>]
        # Shape: (1, 2, embed_size)
        lstm_input = torch.cat((features.unsqueeze(1), start_embedding), dim=1)
        
        # Pass through LSTM
        lstm_output, hidden = self.lstm(lstm_input, hidden)
        
        # Get prediction from last position (after <start>)
        # Shape: (1, hidden_size)
        last_output = lstm_output[:, -1, :]
        
        # Project to vocabulary size
        scores = self.linear(last_output)
        
        # Get word with highest score
        predicted_idx = scores.argmax(dim=1)
        predicted_word = vocab.idx2word[predicted_idx.item()]
        
        # Add to caption if not <end>
        if predicted_word != "<end>":
            caption.append(predicted_word)
        else:
            return caption
        
        # SUBSEQUENT STEPS: Use only word embeddings with hidden state
        # No need to concatenate image features again
        input_word = predicted_idx.unsqueeze(1)
        
        for _ in range(max_length - 1):
            # Convert word index to embedding
            # Shape: (1, 1, embed_size)
            word_embedding = self.embed(input_word)
            
            # Pass through LSTM with previous hidden state
            # Shape: (1, 1, hidden_size)
            lstm_output, hidden = self.lstm(word_embedding, hidden)
            
            # Get last output
            last_output = lstm_output[:, -1, :]
            
            # Project to vocabulary size
            scores = self.linear(last_output)
            
            # Get word with highest score (greedy decoding)
            predicted_idx = scores.argmax(dim=1)
            predicted_word = vocab.idx2word[predicted_idx.item()]
            
            # Stop if <end> token is generated
            if predicted_word == "<end>":
                break
            
            # Add word to caption
            caption.append(predicted_word)
            
            # Use predicted word as input for next step
            input_word = predicted_idx.unsqueeze(1)
        
        return caption



# Make models a package
class __init__:
    pass


# Example usage (for testing)
if __name__ == "__main__":
    # Parameters
    embed_size = 256
    hidden_size = 512
    vocab_size = 5000
    batch_size = 4
    max_length = 20
    
    # Create decoder
    decoder = Decoder(embed_size, hidden_size, vocab_size)
    
    # Create dummy inputs
    features = torch.randn(batch_size, embed_size)
    captions = torch.randint(0, vocab_size, (batch_size, max_length))
    
    # Forward pass
    outputs = decoder(features, captions)
    
    print(f"\nInput shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Captions: {captions.shape}")
    print(f"\nOutput shape: {outputs.shape}")
    print(f"Expected: ({batch_size}, {max_length}, {vocab_size})")
    
    # Count parameters
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params:,}")
    print("\n✓ Decoder test passed!")
