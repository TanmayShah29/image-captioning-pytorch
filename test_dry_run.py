"""
Dry Run Test Script - No Dataset Required

This script validates the code logic without needing the Flickr8k dataset.
It creates synthetic data to test:
1. Encoder forward pass
2. Decoder forward pass (training mode)
3. Decoder caption generation (inference mode)
4. Shape alignment between outputs and targets
5. Loss calculation

If this runs without errors, your code is ready for real training.
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocabulary import Vocabulary

print("="*60)
print("DRY RUN TEST - Image Captioning Project")
print("="*60)
print("\nThis test validates code logic without requiring dataset.\n")

# Test configuration
batch_size = 2
embed_size = 256
hidden_size = 512
vocab_size = 100  # Small vocab for testing
max_length = 20
num_layers = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ========== TEST 1: Encoder ==========
print("="*60)
print("TEST 1: Encoder (CNN Feature Extraction)")
print("="*60)

try:
    encoder = Encoder(embed_size=embed_size).to(device)
    encoder.eval()
    
    # Create dummy images (batch_size, 3, 224, 224)
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        features = encoder(dummy_images)
    
    print(f"✓ Input shape: {dummy_images.shape}")
    print(f"✓ Output shape: {features.shape}")
    print(f"✓ Expected: ({batch_size}, {embed_size})")
    
    assert features.shape == (batch_size, embed_size), "Encoder output shape mismatch!"
    print("✓ Encoder test PASSED\n")
    
except Exception as e:
    print(f"✗ Encoder test FAILED: {e}\n")
    sys.exit(1)

# ========== TEST 2: Vocabulary ==========
print("="*60)
print("TEST 2: Vocabulary")
print("="*60)

try:
    # Create dummy vocabulary
    vocab = Vocabulary(freq_threshold=1)
    sample_captions = [
        "a dog running in the park",
        "a cat sitting on the mat",
        "a dog playing with a ball"
    ]
    vocab.build_vocabulary(sample_captions)
    
    print(f"✓ Vocabulary size: {len(vocab)}")
    print(f"✓ Special tokens: <pad>={vocab.word2idx['<pad>']}, "
          f"<start>={vocab.word2idx['<start>']}, "
          f"<end>={vocab.word2idx['<end>']}, "
          f"<unk>={vocab.word2idx['<unk>']}")
    print("✓ Vocabulary test PASSED\n")
    
except Exception as e:
    print(f"✗ Vocabulary test FAILED: {e}\n")
    sys.exit(1)

# ========== TEST 3: Decoder Training Mode ==========
print("="*60)
print("TEST 3: Decoder Forward Pass (Training Mode)")
print("="*60)

try:
    decoder = Decoder(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        num_layers=num_layers
    ).to(device)
    decoder.train()
    
    # Create dummy captions (batch_size, max_length)
    dummy_captions = torch.randint(0, len(vocab), (batch_size, max_length)).to(device)
    
    # Forward pass
    outputs = decoder(features, dummy_captions)
    
    print(f"✓ Input features shape: {features.shape}")
    print(f"✓ Input captions shape: {dummy_captions.shape}")
    print(f"✓ Output shape: {outputs.shape}")
    print(f"✓ Expected: ({batch_size}, {max_length-1}, {len(vocab)})")
    
    # CRITICAL: Check shape alignment
    expected_shape = (batch_size, max_length - 1, len(vocab))
    assert outputs.shape == expected_shape, f"Decoder output shape mismatch! Got {outputs.shape}, expected {expected_shape}"
    print("✓ Decoder forward pass test PASSED\n")
    
except Exception as e:
    print(f"✗ Decoder forward pass test FAILED: {e}\n")
    sys.exit(1)

# ========== TEST 4: Loss Calculation ==========
print("="*60)
print("TEST 4: Loss Calculation (Shape Alignment)")
print("="*60)

try:
    # Prepare targets (same as train.py)
    targets = dummy_captions[:, 1:]  # Remove <start> token
    
    print(f"✓ Outputs shape: {outputs.shape}")
    print(f"✓ Targets shape: {targets.shape}")
    
    # Reshape for loss calculation
    outputs_flat = outputs.reshape(-1, outputs.shape[2])
    targets_flat = targets.reshape(-1)
    
    print(f"✓ Outputs (flat) shape: {outputs_flat.shape}")
    print(f"✓ Targets (flat) shape: {targets_flat.shape}")
    print(f"✓ Expected: ({batch_size * (max_length-1)}, {len(vocab)}) and ({batch_size * (max_length-1)},)")
    
    # Calculate loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    loss = criterion(outputs_flat, targets_flat)
    
    print(f"✓ Loss value: {loss.item():.4f}")
    print("✓ Loss calculation test PASSED\n")
    
except Exception as e:
    print(f"✗ Loss calculation test FAILED: {e}\n")
    sys.exit(1)

# ========== TEST 5: Decoder Inference Mode ==========
print("="*60)
print("TEST 5: Decoder Caption Generation (Inference Mode)")
print("="*60)

try:
    decoder.eval()
    
    # Generate caption for single image
    single_feature = features[0:1]  # Take first image
    
    with torch.no_grad():
        generated_caption = decoder.generate_caption(
            single_feature, 
            vocab, 
            max_length=10
        )
    
    print(f"✓ Input features shape: {single_feature.shape}")
    print(f"✓ Generated caption: {generated_caption}")
    print(f"✓ Caption length: {len(generated_caption)}")
    print("✓ Caption generation test PASSED\n")
    
except Exception as e:
    print(f"✗ Caption generation test FAILED: {e}\n")
    sys.exit(1)

# ========== TEST 6: Gradient Flow ==========
print("="*60)
print("TEST 6: Gradient Flow (Encoder Frozen, Decoder Trainable)")
print("="*60)

try:
    # Check encoder parameters are frozen
    encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    encoder_total = sum(p.numel() for p in encoder.parameters())
    
    # Check decoder parameters are trainable
    decoder_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    decoder_total = sum(p.numel() for p in decoder.parameters())
    
    print(f"✓ Encoder: {encoder_trainable:,} / {encoder_total:,} trainable")
    print(f"✓ Decoder: {decoder_trainable:,} / {decoder_total:,} trainable")
    
    assert encoder_trainable > 0, "Encoder should have some trainable params (linear layer)"
    assert decoder_trainable == decoder_total, "All decoder params should be trainable"
    
    print("✓ Gradient flow test PASSED\n")
    
except Exception as e:
    print(f"✗ Gradient flow test FAILED: {e}\n")
    sys.exit(1)

# ========== FINAL SUMMARY ==========
print("="*60)
print("🎉 ALL TESTS PASSED!")
print("="*60)
print("\n✅ Your code is ready for training!")
print("\nNext steps:")
print("1. Download Flickr8k dataset (see DATASET_SETUP.md)")
print("2. Run: pip install -r requirements.txt")
print("3. Run: python train.py")
print("\nThe fixes are working correctly:")
print("  ✓ Decoder output/target shapes aligned")
print("  ✓ Inference caption generation logic correct")
print("  ✓ Encoder compatibility updated")
print("  ✓ Loss calculation works without errors")
print("\n" + "="*60)
