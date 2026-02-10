"""
Evaluation Metrics for Image Captioning
========================================
Provides BLEU-1 through BLEU-4 scoring using NLTK.
"""

import torch
from tqdm import tqdm

# NLTK is imported lazily to give a clear error message
try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
except ImportError:
    corpus_bleu = None
    SmoothingFunction = None


def ensure_nltk():
    """Download NLTK data if needed and verify import."""
    if corpus_bleu is None:
        raise ImportError(
            "nltk is required for BLEU evaluation.\n"
            "Install it:  pip install nltk"
        )
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


def evaluate_bleu(encoder, decoder, dataset, vocab, device,
                  max_samples=500, max_length=50):
    """Compute corpus-level BLEU-1 … BLEU-4 on *dataset*.

    Parameters
    ----------
    encoder, decoder : nn.Module
    dataset : FlickrDataset
    vocab : Vocabulary
    device : torch.device
    max_samples : int – cap the number of unique images evaluated
    max_length : int

    Returns
    -------
    dict  {"bleu1": float, "bleu2": float, "bleu3": float, "bleu4": float}
    """
    ensure_nltk()
    smoothie = SmoothingFunction().method1

    encoder.eval()
    decoder.eval()

    from utils.dataset import get_transform
    transform = get_transform(train=False)

    # Group captions by image
    from collections import defaultdict
    from PIL import Image
    import os

    img_caps = defaultdict(list)
    for img_name, cap in dataset.samples:
        img_caps[img_name].append(vocab.tokenize(cap))

    references_all = []
    hypotheses_all = []

    imgs = list(img_caps.keys())[:max_samples]

    with torch.no_grad():
        for img_name in tqdm(imgs, desc="Evaluating BLEU", leave=False):
            img_path = os.path.join(dataset.img_dir, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            image_tensor = transform(image).unsqueeze(0).to(device)
            features = encoder(image_tensor)
            hypothesis = decoder.generate_caption(features, vocab, max_length)

            # References: list of tokenised reference captions for this image
            refs = img_caps[img_name]
            references_all.append(refs)
            hypotheses_all.append(hypothesis)

    if not hypotheses_all:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    bleu1 = corpus_bleu(references_all, hypotheses_all,
                        weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(references_all, hypotheses_all,
                        weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = corpus_bleu(references_all, hypotheses_all,
                        weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(references_all, hypotheses_all,
                        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return {"bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4}


def print_bleu(scores, prefix=""):
    """Pretty-print BLEU scores."""
    tag = f"[{prefix}] " if prefix else ""
    print(f"  {tag}BLEU-1: {scores['bleu1']:.4f}  "
          f"BLEU-2: {scores['bleu2']:.4f}  "
          f"BLEU-3: {scores['bleu3']:.4f}  "
          f"BLEU-4: {scores['bleu4']:.4f}")
