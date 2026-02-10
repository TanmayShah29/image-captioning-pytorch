"""
Verification Script
====================
End-to-end sanity check for the image captioning pipeline.

Validates:
  1. Dataset preparation (auto-detect, normalize, validate)
  2. Vocabulary build & assertions
  3. Train / val split (no overlap)
  4. 1-epoch training (loss is finite)
  5. Sample inference (greedy + beam)
  6. Checkpoint save & reload
"""

import math
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocabulary import Vocabulary
from utils.prepare_dataset import prepare_dataset, DatasetError
from utils.dataset import (
    load_caption_map,
    split_dataset,
    get_transform,
    FlickrDataset,
)


def run_verification(data_dir="data"):
    """Run all verification steps. Returns True if all pass."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    passed = 0
    total = 6

    # ------------------------------------------------------------------
    # 1. Dataset preparation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("CHECK 1/6 — Dataset Preparation")
    print("=" * 60)
    try:
        stats = prepare_dataset(data_dir)
        img_dir = stats["images_dir"]
        captions_file = stats["captions_file"]
        print(f"  ✅ {stats['num_images']} images, {stats['num_captions']} captions")
        passed += 1
    except DatasetError as e:
        print(f"  ❌ {e}")
        print("\n⛔ Cannot continue without dataset. Exiting.")
        return False

    # ------------------------------------------------------------------
    # 2. Vocabulary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 2/6 — Vocabulary")
    print("=" * 60)
    try:
        caption_map = load_caption_map(captions_file)
        all_caps = [c for caps in caption_map.values() for c in caps]
        vocab = Vocabulary(freq_threshold=5)
        vocab.build_vocabulary(all_caps)
        assert len(vocab) > 4, f"Vocab too small ({len(vocab)})"
        for tok in ("<start>", "<end>", "<unk>", "<pad>"):
            assert tok in vocab.word2idx, f"Missing {tok}"
        print(f"  ✅ Vocabulary size: {len(vocab)}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")
        return False

    # ------------------------------------------------------------------
    # 3. Train/Val split — no overlap
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 3/6 — Train / Val Split")
    print("=" * 60)
    try:
        train_map, val_map, test_map = split_dataset(caption_map, seed=42)
        assert not (set(train_map) & set(val_map)), "Train/Val overlap!"
        assert not (set(train_map) & set(test_map)), "Train/Test overlap!"
        assert not (set(val_map) & set(test_map)), "Val/Test overlap!"
        print(f"  ✅ No image overlap between splits")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")
        return False

    # ------------------------------------------------------------------
    # 4. 1-epoch training
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 4/6 — 1-Epoch Training (small subset)")
    print("=" * 60)
    try:
        small_train = dict(list(train_map.items())[:50])
        train_ds = FlickrDataset(img_dir, small_train, vocab,
                                 transform=get_transform(train=True), max_length=50)
        loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

        encoder = Encoder(embed_size=256).to(device)
        encoder.eval()
        decoder = Decoder(256, 512, len(vocab), num_layers=1).to(device)
        decoder.train()

        criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
        optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

        total_loss = 0
        for images, captions in loader:
            images, captions = images.to(device), captions.to(device)
            with torch.no_grad():
                features = encoder(images)
            outputs = decoder(features, captions)
            targets = captions[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        assert not math.isnan(avg), "NaN loss!"
        assert not math.isinf(avg), "Inf loss!"
        print(f"  ✅ 1-epoch loss: {avg:.4f} (finite)")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")
        import traceback; traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 5/6 — Sample Inference")
    print("=" * 60)
    try:
        decoder.eval()
        from PIL import Image
        sample_img = list(train_map.keys())[0]
        img = Image.open(os.path.join(img_dir, sample_img)).convert("RGB")
        tensor = get_transform(train=False)(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = encoder(tensor)
            greedy = decoder.generate_caption(features, vocab, max_length=30)
            beam = decoder.beam_search(features, vocab, beam_size=3, max_length=30)
        print(f"  ✅ Greedy : {' '.join(greedy) if greedy else '(empty)'}")
        print(f"  ✅ Beam(3): {' '.join(beam) if beam else '(empty)'}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # 6. Checkpoint round-trip
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 6/6 — Checkpoint Save & Load")
    print("=" * 60)
    try:
        os.makedirs("saved_models", exist_ok=True)
        path = "saved_models/_verify_ckpt.pth"
        ckpt = {
            "epoch": 1,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "embed_size": 256, "hidden_size": 512,
            "num_layers": 1, "vocab_size": len(vocab),
        }
        torch.save(ckpt, path)
        loaded = torch.load(path, map_location=device, weights_only=False)
        assert loaded["epoch"] == 1
        assert loaded["vocab_size"] == len(vocab)
        os.remove(path)
        print(f"  ✅ Checkpoint save/load round-trip OK")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    icon = "✅" if passed == total else "⚠️"
    print(f"{icon}  VERIFICATION: {passed}/{total} checks passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    args = p.parse_args()
    success = run_verification(args.data_dir)
    sys.exit(0 if success else 1)
