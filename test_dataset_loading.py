"""
Diagnostic Test for Flickr8k Dataset Loading

This script tests the caption loading functionality and provides
detailed diagnostics to help identify any issues.

Run this before training to verify your dataset is loaded correctly.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.vocabulary import Vocabulary
from utils.dataset import FlickrDataset, get_transform

print("=" * 70)
print("FLICKR8K DATASET LOADING DIAGNOSTIC TEST")
print("=" * 70)

# Configuration
data_dir = "data"
img_dir = os.path.join(data_dir, "Flickr8k_Dataset")
captions_file = os.path.join(data_dir, "Flickr8k_text", "Flickr8k.token.txt")

print("\n" + "=" * 70)
print("STEP 1: Checking File Paths")
print("=" * 70)

print(f"\nImage directory: {img_dir}")
print(f"  Exists: {os.path.exists(img_dir)}")
if os.path.exists(img_dir):
    num_images = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    print(f"  Number of .jpg files: {num_images}")

print(f"\nCaptions file: {captions_file}")
print(f"  Exists: {os.path.exists(captions_file)}")
if os.path.exists(captions_file):
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"  Total lines: {len(lines)}")
    print(f"\n  First 3 lines:")
    for i, line in enumerate(lines[:3]):
        # Show with visible tab character
        display_line = line.strip().replace('\t', '<TAB>')
        print(f"    {i+1}: {display_line[:100]}")

if not os.path.exists(img_dir):
    print("\n" + "=" * 70)
    print("ERROR: Image directory not found!")
    print("=" * 70)
    print("\nPlease ensure images are in: data/Flickr8k_Dataset/")
    print("Download from: https://www.kaggle.com/datasets/adityajn105/flickr8k")
    sys.exit(1)

if not os.path.exists(captions_file):
    print("\n" + "=" * 70)
    print("ERROR: Captions file not found!")
    print("=" * 70)
    print("\nPlease ensure captions file is in: data/Flickr8k_text/Flickr8k.token.txt")
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 2: Building Vocabulary")
print("=" * 70)

# Load all captions for vocabulary
all_captions = []
with open(captions_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            all_captions.append(parts[1])

print(f"\nCaptions extracted for vocabulary: {len(all_captions)}")

if len(all_captions) == 0:
    print("\n" + "=" * 70)
    print("ERROR: No captions extracted!")
    print("=" * 70)
    print("\nThe file format may be incorrect.")
    print("Expected format: image.jpg#0<TAB>caption text")
    print("\nPlease check the file format and encoding.")
    sys.exit(1)

# Build vocabulary
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(all_captions)

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample words: {list(vocab.word2idx.keys())[:10]}")

print("\n" + "=" * 70)
print("STEP 3: Creating Dataset")
print("=" * 70)

try:
    dataset = FlickrDataset(
        root_dir=img_dir,
        captions_file=captions_file,
        vocab=vocab,
        transform=get_transform(train=True)
    )
    
    print(f"\n✓ Dataset created successfully!")
    print(f"✓ Dataset length: {len(dataset)}")
    
except ValueError as e:
    print(f"\n✗ Dataset creation failed!")
    print(str(e))
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 4: Testing Sample Loading")
print("=" * 70)

try:
    # Load first sample
    image, caption = dataset[0]
    
    print(f"\n✓ Successfully loaded sample!")
    print(f"  Image shape: {image.shape}")
    print(f"  Caption shape: {caption.shape}")
    print(f"  Caption (indices): {caption[:15].tolist()}...")
    
    # Decode caption
    decoded_words = []
    for idx in caption:
        idx_val = idx.item()
        if idx_val in vocab.idx2word:
            word = vocab.idx2word[idx_val]
            if word == '<end>':
                break
            if word not in ['<start>', '<pad>']:
                decoded_words.append(word)
    
    print(f"  Caption (text): {' '.join(decoded_words)}")
    
except Exception as e:
    print(f"\n✗ Sample loading failed!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 5: Validation Summary")
print("=" * 70)

print(f"\n✓ All checks passed!")
print(f"\nDataset Statistics:")
print(f"  Total captions: {len(all_captions)}")
print(f"  Dataset size: {len(dataset)}")
print(f"  Vocabulary size: {len(vocab)}")
print(f"  Expected captions: ~40,456 (8,091 images × 5 captions)")

if len(dataset) > 40000:
    print(f"\n✓ Dataset size looks correct!")
elif len(dataset) > 0:
    print(f"\n⚠ Dataset size is lower than expected.")
    print(f"  This might be okay if you're using a subset.")
else:
    print(f"\n✗ Dataset is empty!")

print("\n" + "=" * 70)
print("🎉 DIAGNOSTIC TEST COMPLETE!")
print("=" * 70)
print("\nYour dataset is ready for training!")
print("\nNext steps:")
print("  1. Run: python train.py")
print("  2. Monitor the training loss")
print("  3. Use saved model for inference")
print("\n" + "=" * 70)
