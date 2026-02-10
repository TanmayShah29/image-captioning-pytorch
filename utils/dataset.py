"""
Dataset Loader — Reads ONLY the canonical format
=================================================

Canonical contract (produced by prepare_dataset.py):

    data/images/<image>.jpg
    data/captions.txt     →  image_name<TAB>caption  (no header)

This module provides:
  • FlickrDataset      — PyTorch Dataset for training/evaluation
  • get_transform      — image transforms (train vs eval)
  • load_caption_map   — parse canonical captions.txt
  • split_dataset      — reproducible train/val/test split by image ID
"""

import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


# ======================================================================
# Caption loading  (canonical format ONLY)
# ======================================================================

def load_caption_map(captions_file):
    """Load canonical captions.txt → dict[image_name] = [captions].

    Format per line:  ``image_name<TAB>caption``

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If zero valid lines are found.
    """
    if not os.path.isfile(captions_file):
        raise FileNotFoundError(
            f"Canonical caption file not found: '{captions_file}'\n"
            f"Run prepare_dataset() first, or: python -m utils.prepare_dataset"
        )

    cap_map = defaultdict(list)
    malformed = 0

    with open(captions_file, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                malformed += 1
                if malformed <= 3:
                    print(f"  ⚠ Line {lineno}: malformed → {line[:80]}")
                continue
            cap_map[parts[0]].append(parts[1])

    if malformed > 3:
        print(f"  ⚠ ({malformed} total malformed lines)")

    if not cap_map:
        raise RuntimeError(
            f"Zero valid captions in '{captions_file}'.\n"
            f"Expected format: image_name<TAB>caption (one per line, no header).\n"
            f"Re-run dataset preparation: python -m utils.prepare_dataset"
        )

    return dict(cap_map)


# ======================================================================
# Train / Val / Test split by unique image ID
# ======================================================================

def split_dataset(caption_map, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Split by **image ID** → (train_map, val_map, test_map).

    Guarantees no image appears in more than one split.
    """
    imgs = sorted(caption_map.keys())
    rng = random.Random(seed)
    rng.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_imgs = set(imgs[:n_train])
    val_imgs = set(imgs[n_train: n_train + n_val])
    test_imgs = set(imgs[n_train + n_val:])

    assert not (train_imgs & val_imgs), "Train/Val image overlap!"
    assert not (train_imgs & test_imgs), "Train/Test image overlap!"
    assert not (val_imgs & test_imgs), "Val/Test image overlap!"

    train_map = {k: caption_map[k] for k in train_imgs}
    val_map = {k: caption_map[k] for k in val_imgs}
    test_map = {k: caption_map[k] for k in test_imgs}

    print(f"  Split → train: {len(train_map)} imgs | "
          f"val: {len(val_map)} imgs | test: {len(test_map)} imgs")
    return train_map, val_map, test_map


# ======================================================================
# Transforms
# ======================================================================

def get_transform(train=True):
    """Image transforms.  Train: augmented.  Val/Test: deterministic."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])


# ======================================================================
# Dataset
# ======================================================================

class FlickrDataset(Dataset):
    """PyTorch Dataset that reads the canonical format.

    Parameters
    ----------
    img_dir      : str   – path to ``data/images/``
    caption_map  : dict  – {image_filename: [caption_strings]}
    vocab        : Vocabulary
    transform    : callable
    max_length   : int   – fixed caption tensor length
    """

    def __init__(self, img_dir, caption_map, vocab, transform=None, max_length=50):
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

        # Flatten to (image, caption) pairs, keep only files that exist
        self.samples = []
        missing = set()
        for img, caps in caption_map.items():
            if os.path.isfile(os.path.join(img_dir, img)):
                for cap in caps:
                    self.samples.append((img, cap))
            else:
                missing.add(img)

        if missing:
            print(f"  ⚠ {len(missing)} images in captions but missing on disk")

        n_imgs = len({s[0] for s in self.samples})
        n_caps = len(self.samples)
        if n_caps == 0:
            raise RuntimeError(
                f"Dataset is empty (0 pairs). img_dir={img_dir}, "
                f"caption_map had {len(caption_map)} entries, "
                f"{len(missing)} missing on disk."
            )
        print(f"  Dataset: {n_imgs} images, {n_caps} captions")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Safe image loading
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ⚠ Corrupted image {img_name}: {e}")
            return self[random.randint(0, len(self) - 1)]

        if self.transform:
            image = self.transform(image)

        # Numericalize caption
        tokens = self.vocab.numericalize(caption)
        tokens = ([self.vocab.word2idx["<start>"]]
                  + tokens
                  + [self.vocab.word2idx["<end>"]])

        pad_idx = self.vocab.word2idx["<pad>"]
        if len(tokens) < self.max_length:
            tokens += [pad_idx] * (self.max_length - len(tokens))
        else:
            tokens = tokens[: self.max_length - 1] + [self.vocab.word2idx["<end>"]]

        return image, torch.tensor(tokens, dtype=torch.long)
