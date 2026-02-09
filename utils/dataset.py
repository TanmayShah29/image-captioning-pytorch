"""
Dataset Loader for Flickr8k Image Captioning

This module handles loading images and captions from Flickr8k dataset.

Key Responsibilities:
1. Read image-caption pairs from Flickr8k files
2. Preprocess images (resize, normalize)
3. Tokenize and pad captions
4. Create batches for training

Dataset Structure:
- Flickr8k_Dataset/: Contains all images
- Flickr8k_text/Flickr8k.token.txt: Contains all captions
- Each image has 5 different captions
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class FlickrDataset(Dataset):
    """
    PyTorch Dataset for Flickr8k.
    
    What is a Dataset?
    - PyTorch Dataset is a class that provides data samples
    - Must implement __len__ and __getitem__ methods
    - Works with DataLoader to create batches
    
    Attributes:
        root_dir (str): Path to Flickr8k_Dataset folder
        captions_file (str): Path to Flickr8k.token.txt
        transform: Image preprocessing pipeline
        vocab: Vocabulary object for text processing
    """
    
    def __init__(self, root_dir, captions_file, vocab, transform=None, max_length=50):
        """
        Initialize dataset.
        
        Args:
            root_dir (str): Path to image folder (Flickr8k_Dataset/)
            captions_file (str): Path to captions file (Flickr8k.token.txt)
            vocab (Vocabulary): Vocabulary object for text conversion
            transform: Image transformations (resize, normalize, etc.)
            max_length (int): Maximum caption length (for padding)
            
        Why max_length: All captions must be same length for batch processing.
        Shorter captions are padded, longer ones are truncated.
        """
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length
        
        # Load all image-caption pairs
        self.imgs, self.captions = self.load_captions(captions_file)
        
        # Validate that captions were loaded
        if len(self.captions) == 0:
            raise ValueError(
                f"\n{'='*60}\n"
                f"ERROR: No captions were loaded from {captions_file}\n"
                f"{'='*60}\n"
                f"Possible causes:\n"
                f"1. File is empty or doesn't exist\n"
                f"2. File format is incorrect (should be: image.jpg#0<TAB>caption)\n"
                f"3. File encoding issue (should be UTF-8)\n"
                f"\nExpected format:\n"
                f"  1000268201_693b08cb0e.jpg#0<TAB>A child in a pink dress climbing stairs\n"
                f"  1000268201_693b08cb0e.jpg#1<TAB>A girl going into a wooden building\n"
                f"\nPlease check the file and try again.\n"
                f"{'='*60}\n"
            )
        
        print(f"Dataset loaded: {len(self.imgs)} image-caption pairs")
        print(f"Unique images: {len(set(self.imgs))}")

    
    def load_captions(self, captions_file):
        """
        Load captions from Flickr8k.token.txt file.
        
        Args:
            captions_file (str): Path to captions file
            
        Returns:
            imgs (list): List of image filenames
            captions (list): List of corresponding captions
            
        File Format:
            1000268201_693b08cb0e.jpg#0    A child in a pink dress climbing stairs
            1000268201_693b08cb0e.jpg#1    A girl going into a wooden building
            
        Note: Each image has 5 captions (#0 to #4)
        """
        all_imgs = []
        all_captions = []
        
        print(f"\n{'='*60}")
        print(f"LOADING DATASET")
        print(f"{'='*60}")
        print(f"Captions file: {captions_file}")
        print(f"Image directory: {self.root_dir}")
        print(f"File exists: {os.path.exists(captions_file)}")
        
        if not os.path.exists(captions_file):
            print(f"[ERROR] Captions file not found: {captions_file}")
            return all_imgs, all_captions
        
        line_count = 0
        skipped_format = 0
        
        # First pass: Load ALL captions from file
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split line into image_id and caption using TAB
                # Format: "image.jpg#0\tCaption text"
                parts = line.strip().split('\t')
                
                if len(parts) != 2:
                    skipped_format += 1
                    if skipped_format <= 3:  # Show first few problematic lines
                        print(f"[WARNING] Line {line_count} has {len(parts)} parts (expected 2): {line[:80]}")
                    continue
                
                # Extract image filename (remove #0, #1, etc.)
                img_id = parts[0].split('#')[0]
                caption = parts[1].strip()
                
                # Add to lists
                all_imgs.append(img_id)
                all_captions.append(caption)
        
        print(f"\n--- Caption Loading Results ---")
        print(f"Total lines read: {line_count}")
        print(f"Lines with format issues: {skipped_format}")
        print(f"Total captions loaded: {len(all_captions)}")
        print(f"Unique images referenced: {len(set(all_imgs))}")
        
        if len(all_captions) == 0:
            print("\n[ERROR] No captions were loaded!")
            print("Check that file format is: image.jpg#0<TAB>caption text")
            return all_imgs, all_captions
        
        # Second pass: Filter by image existence
        imgs = []
        captions = []
        missing_images = set()
        
        for img_name, caption in zip(all_imgs, all_captions):
            img_path = os.path.join(self.root_dir, img_name)
            if os.path.exists(img_path):
                imgs.append(img_name)
                captions.append(caption)
            else:
                missing_images.add(img_name)
        
        print(f"\n--- Image Verification Results ---")
        print(f"Images found: {len(set(imgs))}")
        print(f"Images missing: {len(missing_images)}")
        print(f"Final dataset size: {len(imgs)} image-caption pairs")
        
        if len(missing_images) > 0 and len(missing_images) <= 5:
            print(f"Missing images: {list(missing_images)[:5]}")
        
        if len(imgs) > 0:
            print(f"\n--- Sample Data ---")
            print(f"First image: {imgs[0]}")
            print(f"First caption: {captions[0][:80]}...")
        
        print(f"{'='*60}\n")
        
        return imgs, captions
    
    def __len__(self):
        """
        Return total number of samples.
        
        Returns:
            int: Number of image-caption pairs
            
        Why needed: DataLoader uses this to know dataset size.
        """
        return len(self.imgs)
    
    def __getitem__(self, idx):
        """
        Get one sample (image + caption) by index.
        
        Args:
            idx (int): Index of sample to retrieve
            
        Returns:
            image (Tensor): Preprocessed image tensor
            caption (Tensor): Numericalized and padded caption
            
        Process:
            1. Load image from disk
            2. Apply transformations (resize, normalize)
            3. Convert caption to indices
            4. Add <start> and <end> tokens
            5. Pad to max_length
            
        Why needed: DataLoader calls this method to get batches.
        """
        # Get image filename and caption
        img_name = self.imgs[idx]
        caption = self.captions[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Convert caption to numerical indices
        # Example: "a dog running" -> [4, 5, 6]
        numericalized_caption = self.vocab.numericalize(caption)
        
        # Add <start> and <end> tokens
        # [4, 5, 6] -> [1, 4, 5, 6, 2]
        # where 1=<start>, 2=<end>
        numericalized_caption = [self.vocab.word2idx["<start>"]] + \
                               numericalized_caption + \
                               [self.vocab.word2idx["<end>"]]
        
        # Pad or truncate to max_length
        # Why: All captions in a batch must have same length
        if len(numericalized_caption) < self.max_length:
            # Pad with <pad> token (index 0)
            numericalized_caption += [self.vocab.word2idx["<pad>"]] * \
                                    (self.max_length - len(numericalized_caption))
        else:
            # Truncate if too long
            numericalized_caption = numericalized_caption[:self.max_length]
        
        # Convert to PyTorch tensor
        caption_tensor = torch.tensor(numericalized_caption, dtype=torch.long)
        
        return image, caption_tensor


def get_transform(train=True):
    """
    Get image preprocessing transformations.
    
    Args:
        train (bool): Whether for training or testing
        
    Returns:
        transforms.Compose: Composition of transformations
        
    Transformations explained:
    1. Resize: Make all images 224x224 (ResNet input size)
    2. ToTensor: Convert PIL Image to PyTorch tensor (0-1 range)
    3. Normalize: Standardize using ImageNet mean and std
       - Why: ResNet was trained on ImageNet with these values
       - Mean: [0.485, 0.456, 0.406] for RGB channels
       - Std: [0.229, 0.224, 0.225] for RGB channels
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224
            transforms.ToTensor(),           # Convert to tensor (H,W,C) -> (C,H,W)
            transforms.Normalize(            # Normalize using ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Same transformations for test/validation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def get_data_loader(root_dir, captions_file, vocab, batch_size=32, 
                   shuffle=True, num_workers=2):
    """
    Create DataLoader for batching.
    
    Args:
        root_dir (str): Path to images
        captions_file (str): Path to captions
        vocab (Vocabulary): Vocabulary object
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of parallel workers for loading
        
    Returns:
        DataLoader: PyTorch DataLoader object
        
    What is DataLoader?
    - Automatically creates batches from Dataset
    - Handles shuffling and parallel loading
    - Makes training loop much simpler
    
    Batch Shape:
    - Images: (batch_size, 3, 224, 224)
    - Captions: (batch_size, max_length)
    
    Example:
        batch_size=32, max_length=50
        -> Images: (32, 3, 224, 224)
        -> Captions: (32, 50)
    """
    # Get transformations
    transform = get_transform(train=shuffle)
    
    # Create dataset
    dataset = FlickrDataset(
        root_dir=root_dir,
        captions_file=captions_file,
        vocab=vocab,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    return dataloader


# Example usage (for testing)
if __name__ == "__main__":
    from vocabulary import Vocabulary
    
    # Paths (update these to your actual paths)
    root_dir = "data/Flickr8k_Dataset"
    captions_file = "data/Flickr8k_text/Flickr8k.token.txt"
    
    # Check if files exist
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found")
        print("Please download Flickr8k dataset first")
    elif not os.path.exists(captions_file):
        print(f"Error: {captions_file} not found")
        print("Please download Flickr8k captions first")
    else:
        # Create dummy vocabulary for testing
        vocab = Vocabulary(freq_threshold=1)
        
        # Load dataset
        dataset = FlickrDataset(
            root_dir=root_dir,
            captions_file=captions_file,
            vocab=vocab,
            transform=get_transform()
        )
        
        # Test loading one sample
        if len(dataset) > 0:
            image, caption = dataset[0]
            print(f"\nSample loaded successfully!")
            print(f"Image shape: {image.shape}")
            print(f"Caption shape: {caption.shape}")
        else:
            print("Dataset is empty")
