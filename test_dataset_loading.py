"""
Dataset Dry-Run Test
====================
Validates the entire data pipeline without training:

  1. Run prepare_dataset() to normalize data
  2. Load canonical captions.txt
  3. Build vocabulary
  4. Load at least 5 image–caption pairs
  5. Open images with PIL
  6. Print sample captions

Exits 0 ONLY if everything is valid.
"""

import os
import sys
import torch
from PIL import Image

from utils.prepare_dataset import prepare_dataset, DatasetError
from utils.dataset import load_caption_map, get_transform, FlickrDataset
from utils.vocabulary import Vocabulary


def run_test(data_root="data", num_samples=5):
    """Run dry-run dataset loading test."""

    print("=" * 64)
    print("  DATASET DRY-RUN TEST")
    print("=" * 64)

    # ------------------------------------------------------------------
    # 1. Prepare dataset (auto-detect, normalize, validate)
    # ------------------------------------------------------------------
    print("\n✦ Step 1: Dataset preparation")
    try:
        stats = prepare_dataset(data_root)
    except DatasetError as e:
        print(str(e))
        return False

    img_dir = stats["images_dir"]
    captions_file = stats["captions_file"]

    # ------------------------------------------------------------------
    # 2. Load canonical captions
    # ------------------------------------------------------------------
    print("\n✦ Step 2: Load captions")
    caption_map = load_caption_map(captions_file)
    total_imgs = len(caption_map)
    total_caps = sum(len(v) for v in caption_map.values())
    print(f"  Loaded {total_imgs} images, {total_caps} captions")

    if total_caps == 0:
        print("  ❌ FAIL: Zero captions loaded")
        return False
    print("  ✅ PASS")

    # ------------------------------------------------------------------
    # 3. Build vocabulary
    # ------------------------------------------------------------------
    print("\n✦ Step 3: Build vocabulary")
    all_captions = [c for caps in caption_map.values() for c in caps]
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_captions)
    print(f"  Vocabulary size: {len(vocab)} words")

    for token in ["<pad>", "<start>", "<end>", "<unk>"]:
        assert token in vocab.word2idx, f"Missing required token: {token}"
    print("  ✅ PASS — All special tokens present")

    # ------------------------------------------------------------------
    # 4. Load image–caption pairs + open images with PIL
    # ------------------------------------------------------------------
    print(f"\n✦ Step 4: Load & validate {num_samples} samples")
    ds = FlickrDataset(img_dir, caption_map, vocab,
                       transform=get_transform(train=False))

    if len(ds) < num_samples:
        print(f"  ❌ FAIL: Dataset has only {len(ds)} samples (need {num_samples})")
        return False

    for i in range(num_samples):
        img_name, caption_text = ds.samples[i]
        img_path = os.path.join(img_dir, img_name)

        # Open with PIL (raw, no transform)
        try:
            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size
        except Exception as e:
            print(f"  ❌ FAIL: Cannot open {img_name}: {e}")
            return False

        # Also test the __getitem__ path (transform + numericalize)
        image_tensor, caption_tensor = ds[i]
        assert image_tensor.shape == (3, 224, 224), f"Bad image shape: {image_tensor.shape}"
        assert len(caption_tensor.shape) == 1, f"Bad caption shape: {caption_tensor.shape}"

        print(f"  [{i+1}] {img_name}")
        print(f"      Image  : {w}×{h} → tensor {tuple(image_tensor.shape)}")
        print(f"      Caption: {caption_text[:80]}")

    print(f"\n  ✅ PASS — All {num_samples} samples valid")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  ✅ DRY-RUN TEST PASSED")
    print("=" * 64)
    print(f"  Images         : {total_imgs}")
    print(f"  Captions       : {total_caps}")
    print(f"  Vocabulary     : {len(vocab)} words")
    print(f"  Dataset samples: {len(ds)}")
    print(f"  Canonical dir  : {img_dir}")
    print(f"  Canonical file : {captions_file}")
    print("=" * 64)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dry-run dataset test")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    ok = run_test(args.data_root, args.num_samples)
    sys.exit(0 if ok else 1)
