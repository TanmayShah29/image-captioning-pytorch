"""
Training Script for Image Captioning

This script trains the LSTM decoder on Flickr8k dataset.

Training Process Overview:
1. Load and preprocess Flickr8k dataset
2. Build vocabulary from captions
3. Initialize encoder (frozen ResNet50) and decoder (trainable LSTM)
4. Train decoder using teacher forcing
5. Save trained model and vocabulary

Key Concepts Explained:
- Teacher Forcing: Feed actual previous word during training
- Loss Function: CrossEntropyLoss measures prediction accuracy
- Backpropagation: Update only decoder weights (encoder frozen)
- Optimizer: Adam optimizer for efficient training
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocabulary import Vocabulary
from utils.dataset import FlickrDataset, get_transform


def train_model(
    data_dir="data",
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
    embed_size=256,
    hidden_size=512,
    num_layers=1,
    save_dir="saved_models"
):
    """
    Main training function.
    
    Args:
        data_dir (str): Path to data folder containing Flickr8k
        num_epochs (int): Number of training epochs
        batch_size (int): Number of samples per batch
        learning_rate (float): Learning rate for optimizer
        embed_size (int): Dimension of embeddings
        hidden_size (int): LSTM hidden state size
        num_layers (int): Number of LSTM layers
        save_dir (str): Directory to save models
    
    Training Loop:
        For each epoch:
            For each batch:
                1. Extract image features (encoder)
                2. Generate caption predictions (decoder)
                3. Calculate loss
                4. Backpropagate
                5. Update decoder weights
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========== STEP 1: Load Dataset ==========
    print("\n" + "="*50)
    print("STEP 1: Loading Dataset")
    print("="*50)
    
    # Paths to Flickr8k data
    img_dir = os.path.join(data_dir, "Flickr8k_Dataset")
    captions_file = os.path.join(data_dir, "Flickr8k_text", "Flickr8k.token.txt")
    
    # Check if data exists
    if not os.path.exists(img_dir):
        print(f"\nERROR: Image directory not found: {img_dir}")
        print("\nPlease download Flickr8k dataset:")
        print("1. Download from: https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print("2. Extract to: data/Flickr8k_Dataset/")
        return
    
    if not os.path.exists(captions_file):
        print(f"\nERROR: Captions file not found: {captions_file}")
        print("\nPlease ensure Flickr8k.token.txt is in: data/Flickr8k_text/")
        return
    
    # ========== STEP 2: Build Vocabulary ==========
    print("\n" + "="*50)
    print("STEP 2: Building Vocabulary")
    print("="*50)
    
    # Load all captions for vocabulary building
    all_captions = []
    with open(captions_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                all_captions.append(parts[1])
    
    print(f"Total captions: {len(all_captions)}")
    
    # Build vocabulary
    # freq_threshold=5: Only include words appearing 5+ times
    # This reduces vocabulary size and prevents overfitting on rare words
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_captions)
    
    # Save vocabulary for later use (inference)
    vocab_path = os.path.join(save_dir, "vocabulary.pkl")
    vocab.save_vocabulary(vocab_path)
    
    # ========== STEP 3: Create DataLoader ==========
    print("\n" + "="*50)
    print("STEP 3: Creating DataLoader")
    print("="*50)
    
    # Create dataset
    dataset = FlickrDataset(
        root_dir=img_dir,
        captions_file=captions_file,
        vocab=vocab,
        transform=get_transform(train=True)
    )
    
    # Create dataloader for batching
    # shuffle=True: Randomize order each epoch
    # num_workers=2: Parallel data loading
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # Faster GPU transfer
    )
    
    print(f"Batches per epoch: {len(dataloader)}")
    
    # ========== STEP 4: Initialize Models ==========
    print("\n" + "="*50)
    print("STEP 4: Initializing Models")
    print("="*50)
    
    # Create encoder (frozen ResNet50)
    encoder = Encoder(embed_size=embed_size).to(device)
    encoder.eval()  # Set to evaluation mode (frozen)
    
    # Create decoder (trainable LSTM)
    decoder = Decoder(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        num_layers=num_layers
    ).to(device)
    decoder.train()  # Set to training mode
    
    # ========== STEP 5: Setup Training ==========
    print("\n" + "="*50)
    print("STEP 5: Setting Up Training")
    print("="*50)
    
    # Loss function: CrossEntropyLoss
    # Measures how well predicted words match actual words
    # Ignores <pad> tokens in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    
    # Optimizer: Adam
    # Only update decoder parameters (encoder is frozen)
    # Why Adam? Adaptive learning rates, works well for most tasks
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    print(f"Loss function: CrossEntropyLoss")
    print(f"Optimizer: Adam (lr={learning_rate})")
    print(f"Only training decoder parameters (encoder frozen)")
    
    # ========== STEP 6: Training Loop ==========
    print("\n" + "="*50)
    print("STEP 6: Training")
    print("="*50)
    
    # Track loss history for plotting
    loss_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        # Progress bar for batches
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, captions) in enumerate(pbar):
            # Move data to device (GPU/CPU)
            images = images.to(device)
            captions = captions.to(device)
            
            # ===== Forward Pass =====
            
            # 1. Extract image features using encoder
            # Shape: (batch_size, embed_size)
            with torch.no_grad():  # No gradients for encoder
                features = encoder(images)
            
            # 2. Generate predictions using decoder
            # Shape: (batch_size, max_length, vocab_size)
            outputs = decoder(features, captions)
            
            # 3. Prepare targets
            # Remove first word (<start>) from captions
            # We predict words 1 to N given words 0 to N-1
            targets = captions[:, 1:]
            
            # Reshape for loss calculation
            # outputs: (batch_size * max_length, vocab_size)
            # targets: (batch_size * max_length)
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = targets.reshape(-1)
            
            # 4. Calculate loss
            # How different are predictions from actual words?
            loss = criterion(outputs, targets)
            
            # ===== Backward Pass =====
            
            # 5. Clear previous gradients
            optimizer.zero_grad()
            
            # 6. Backpropagation
            # Calculate gradients for all parameters
            loss.backward()
            
            # 7. Update weights
            # Only decoder weights are updated (encoder frozen)
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Average loss for epoch
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'vocab_size': len(vocab),
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        }
        
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    # ========== STEP 7: Save Final Model ==========
    print("\n" + "="*50)
    print("STEP 7: Saving Final Model")
    print("="*50)
    
    final_model_path = os.path.join(save_dir, "final_model.pth")
    torch.save(checkpoint, final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # ========== STEP 8: Plot Training Loss ==========
    print("\n" + "="*50)
    print("STEP 8: Plotting Training Loss")
    print("="*50)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    
    plot_path = os.path.join(save_dir, "training_loss.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved: {plot_path}")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"\nSaved files:")
    print(f"  - Model: {final_model_path}")
    print(f"  - Vocabulary: {vocab_path}")
    print(f"  - Loss plot: {plot_path}")


if __name__ == "__main__":
    """
    Main entry point for training.
    
    To run:
        python train.py
    
    Requirements:
        1. Flickr8k dataset in data/Flickr8k_Dataset/
        2. Captions file in data/Flickr8k_text/Flickr8k.token.txt
    
    Output:
        - Trained model in saved_models/
        - Vocabulary in saved_models/vocabulary.pkl
        - Training loss plot
    
    Training Tips:
        - Start with 5-10 epochs for testing
        - Monitor loss - should decrease over time
        - Training on CPU: ~30 min per epoch
        - Training on GPU: ~5 min per epoch
        - Use Google Colab for free GPU access
    """
    
    print("="*50)
    print("IMAGE CAPTIONING - TRAINING SCRIPT")
    print("="*50)
    print("\nThis script trains an LSTM decoder on Flickr8k dataset.")
    print("The CNN encoder (ResNet50) is pre-trained and frozen.")
    print("Only the LSTM decoder is trained from scratch.")
    print("\n" + "="*50)
    
    # Training configuration
    config = {
        "data_dir": "data",
        "num_epochs": 10,          # Increase for better results
        "batch_size": 32,          # Reduce if out of memory
        "learning_rate": 0.001,
        "embed_size": 256,
        "hidden_size": 512,
        "num_layers": 1,
        "save_dir": "saved_models"
    }
    
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Start training
    train_model(**config)
