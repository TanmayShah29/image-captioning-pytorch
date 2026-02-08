"""
Inference Script for Image Captioning

This script generates captions for new images using a trained model.

Process:
1. Load trained model and vocabulary
2. Load and preprocess input image
3. Extract features using encoder
4. Generate caption using decoder (greedy decoding)
5. Display image with generated caption

Greedy Decoding:
- Start with <start> token
- At each step, pick word with highest probability
- Feed predicted word as input for next step
- Stop at <end> token or max length
"""

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocabulary import Vocabulary
from utils.dataset import get_transform


def load_model(model_path, vocab_path, device):
    """
    Load trained model and vocabulary.
    
    Args:
        model_path (str): Path to saved model checkpoint
        vocab_path (str): Path to saved vocabulary
        device: torch device (CPU/GPU)
    
    Returns:
        encoder: Loaded encoder model
        decoder: Loaded decoder model
        vocab: Loaded vocabulary
    
    Why load both?
    - Encoder: Extract features from new images
    - Decoder: Generate captions from features
    - Vocabulary: Convert word indices to text
    """
    print("Loading model and vocabulary...")
    
    # Load vocabulary
    vocab = Vocabulary.load_vocabulary(vocab_path)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model parameters from checkpoint
    embed_size = checkpoint['embed_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    
    # Initialize encoder
    encoder = Encoder(embed_size=embed_size).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()  # Set to evaluation mode
    
    # Initialize decoder
    decoder = Decoder(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        num_layers=num_layers
    ).to(device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()  # Set to evaluation mode
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Vocabulary size: {len(vocab)}")
    
    return encoder, decoder, vocab


def generate_caption(image_path, encoder, decoder, vocab, device, max_length=50):
    """
    Generate caption for a single image.
    
    Args:
        image_path (str): Path to input image
        encoder: Trained encoder model
        decoder: Trained decoder model
        vocab: Vocabulary object
        device: torch device
        max_length (int): Maximum caption length
    
    Returns:
        caption (str): Generated caption
        image (PIL.Image): Original image
    
    Process:
        1. Load and preprocess image
        2. Extract features using encoder
        3. Generate caption using decoder
        4. Convert indices to words
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess image
    # Same transformations as training
    transform = get_transform(train=False)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Extract features
    with torch.no_grad():
        features = encoder(image_tensor)
    
    # Generate caption
    # decoder.generate_caption uses greedy decoding
    caption_words = decoder.generate_caption(features, vocab, max_length)
    
    # Join words into sentence
    caption = " ".join(caption_words)
    
    return caption, image


def display_result(image, caption, save_path=None):
    """
    Display image with generated caption.
    
    Args:
        image (PIL.Image): Input image
        caption (str): Generated caption
        save_path (str): Optional path to save result
    
    Why visualize?
    - Easy to evaluate caption quality
    - Good for presentations and viva
    - Shows model working end-to-end
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated Caption:\n{caption}", fontsize=14, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Result saved to: {save_path}")
    
    plt.show()


def main():
    """
    Main inference function.
    
    Usage:
        python inference.py --image path/to/image.jpg
        python inference.py --image path/to/image.jpg --save output.png
    
    Arguments:
        --image: Path to input image (required)
        --model: Path to model checkpoint (default: saved_models/final_model.pth)
        --vocab: Path to vocabulary (default: saved_models/vocabulary.pkl)
        --save: Path to save result (optional)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate caption for an image")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--model", type=str, 
                       default="saved_models/final_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str,
                       default="saved_models/vocabulary.pkl",
                       help="Path to vocabulary file")
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save result image")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Please train the model first using train.py")
        return
    
    if not os.path.exists(args.vocab):
        print(f"Error: Vocabulary not found: {args.vocab}")
        print("Please train the model first using train.py")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    encoder, decoder, vocab = load_model(args.model, args.vocab, device)
    
    # Generate caption
    print(f"\nGenerating caption for: {args.image}")
    caption, image = generate_caption(
        args.image, encoder, decoder, vocab, device
    )
    
    # Display result
    print(f"\nGenerated Caption: {caption}")
    display_result(image, caption, args.save)


def test_on_sample_images(encoder, decoder, vocab, device, sample_dir="data/Flickr8k_Dataset"):
    """
    Test model on multiple sample images.
    
    Args:
        encoder: Trained encoder
        decoder: Trained decoder
        vocab: Vocabulary
        device: torch device
        sample_dir: Directory containing sample images
    
    Why useful?
    - Evaluate model on multiple images
    - Good for viva demonstration
    - Shows model generalization
    """
    # Get first 5 images from dataset
    image_files = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')][:5]
    
    print(f"\nTesting on {len(image_files)} sample images...")
    
    for img_file in image_files:
        img_path = os.path.join(sample_dir, img_file)
        caption, image = generate_caption(img_path, encoder, decoder, vocab, device)
        
        print(f"\nImage: {img_file}")
        print(f"Caption: {caption}")
        
        # Display
        display_result(image, caption)


if __name__ == "__main__":
    """
    Entry point for inference.
    
    Two modes:
    1. Single image: python inference.py --image path/to/image.jpg
    2. Batch testing: Uncomment test_on_sample_images() below
    
    Examples:
        # Generate caption for one image
        python inference.py --image data/Flickr8k_Dataset/1000268201_693b08cb0e.jpg
        
        # Generate and save result
        python inference.py --image test.jpg --save result.png
        
        # Use custom model
        python inference.py --image test.jpg --model saved_models/checkpoint_epoch_5.pth
    """
    
    print("="*50)
    print("IMAGE CAPTIONING - INFERENCE SCRIPT")
    print("="*50)
    print("\nThis script generates captions for images using a trained model.")
    print("="*50 + "\n")
    
    main()
    
    # Uncomment below to test on multiple sample images
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder, decoder, vocab = load_model(
    #     "saved_models/final_model.pth",
    #     "saved_models/vocabulary.pkl",
    #     device
    # )
    # test_on_sample_images(encoder, decoder, vocab, device)
