"""
Inference Script for Image Captioning
======================================
Generate captions for images using a trained model.

Supports:
  • Greedy decoding (fast, default)
  • Beam search decoding (higher quality, --beam_size > 1)
"""

import argparse
import os

import torch
from PIL import Image

from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocabulary import Vocabulary
from utils.dataset import get_transform


# ======================================================================
# Model loading
# ======================================================================

def load_model(model_path, vocab_path, device):
    """Load encoder, decoder, and vocabulary from disk.

    Returns
    -------
    encoder, decoder, vocab
    """
    vocab = Vocabulary.load_vocabulary(vocab_path)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    embed_size = ckpt["embed_size"]
    hidden_size = ckpt["hidden_size"]
    num_layers = ckpt["num_layers"]

    encoder = Encoder(embed_size=embed_size).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    decoder = Decoder(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        num_layers=num_layers,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()

    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")
    return encoder, decoder, vocab


# ======================================================================
# Caption generation
# ======================================================================

def generate_caption(image_path, encoder, decoder, vocab, device,
                     beam_size=1, max_length=50):
    """Generate a caption for a single image.

    Args:
        beam_size: 1 = greedy decoding; >1 = beam search.

    Returns
    -------
    caption : str
    image   : PIL.Image
    """
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(tensor)
        if beam_size > 1:
            words = decoder.beam_search(features, vocab,
                                        beam_size=beam_size,
                                        max_length=max_length)
        else:
            words = decoder.generate_caption(features, vocab, max_length)

    # Filter any remaining special tokens
    clean = [w for w in words if w not in ("<start>", "<end>", "<pad>")]
    caption = " ".join(clean) if clean else "(empty caption)"
    return caption, image


# ======================================================================
# Display
# ======================================================================

def display_result(image, caption, save_path=None):
    """Show / save image with its caption."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"Generated Caption:\n{caption}", fontsize=14, pad=20)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Result saved → {save_path}")
        plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"  (Could not display image: {e})")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate image captions")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--model", type=str, default="saved_models/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, default="saved_models/vocabulary.pkl",
                        help="Path to vocabulary file")
    parser.add_argument("--beam_size", type=int, default=3,
                        help="Beam size (1 = greedy)")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--save", type=str, default=None,
                        help="Save result image to this path")
    args = parser.parse_args()

    for label, path in [("Image", args.image), ("Model", args.model),
                        ("Vocab", args.vocab)]:
        if not os.path.exists(path):
            print(f"Error: {label} not found → {path}")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder, decoder, vocab = load_model(args.model, args.vocab, device)

    mode = f"beam-{args.beam_size}" if args.beam_size > 1 else "greedy"
    print(f"\nGenerating caption ({mode}): {args.image}")

    caption, image = generate_caption(
        args.image, encoder, decoder, vocab, device,
        beam_size=args.beam_size, max_length=args.max_length,
    )

    print(f"\n  Caption: {caption}\n")
    display_result(image, caption, args.save)


if __name__ == "__main__":
    print("=" * 60)
    print("IMAGE CAPTIONING — INFERENCE")
    print("=" * 60)
    main()
