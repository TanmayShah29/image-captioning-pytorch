"""
Training Script for Image Captioning
=====================================
End-to-end training on Flickr8k with:
  • Mandatory dataset preparation (auto-detect, normalize, validate)
  • YAML config + CLI overrides
  • Train / val split with BLEU evaluation
  • LR scheduling  (ReduceLROnPlateau)
  • Early stopping
  • Mixed-precision (AMP)
  • Full checkpoint save / resume
"""

import argparse
import math
import os
import sys
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from evaluate import evaluate_bleu, print_bleu


# ======================================================================
# Config helpers
# ======================================================================

DEFAULT_CONFIG = {
    "data_dir": "data",
    "save_dir": "saved_models",
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "embed_size": 256,
    "hidden_size": 512,
    "num_layers": 1,
    "dropout": 0.5,
    "freq_threshold": 5,
    "max_length": 50,
    "beam_size": 3,
    "early_stop_patience": 5,
    "scheduler_patience": 2,
    "use_amp": True,
    "num_workers": 2,
    "seed": 42,
}


def load_config(config_path=None):
    cfg = dict(DEFAULT_CONFIG)
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update(file_cfg)
        print(f"Config loaded from {config_path}")
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Train Image Captioning Model")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    for k, v in DEFAULT_CONFIG.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=str, default=None)
        else:
            p.add_argument(f"--{k}", type=type(v), default=None)
    return p.parse_args()


def apply_cli_overrides(cfg, args):
    for k in DEFAULT_CONFIG:
        v = getattr(args, k, None)
        if v is not None:
            if isinstance(DEFAULT_CONFIG[k], bool):
                cfg[k] = str(v).lower() in ("true", "1", "yes")
            else:
                cfg[k] = v
    return cfg


# ======================================================================
# Main training
# ======================================================================

def train_model(cfg, resume_path=None):

    # ------------------------------------------------------------------
    # 0. MANDATORY — Prepare & validate dataset  (THE GATE)
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  STEP 0 — DATASET PREPARATION  (mandatory)")
    print("=" * 64)

    try:
        ds_stats = prepare_dataset(data_root=cfg["data_dir"])
    except DatasetError as e:
        print(str(e))
        print("\n⛔ Training CANNOT start until the dataset is valid.")
        print("   Fix the issue above and re-run.\n")
        sys.exit(1)

    img_dir = ds_stats["images_dir"]
    captions_file = ds_stats["captions_file"]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["use_amp"] and device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ------------------------------------------------------------------
    # 1. Load canonical captions & build vocabulary
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  STEP 1 — Vocabulary")
    print("=" * 64)

    caption_map = load_caption_map(captions_file)
    all_captions = [c for caps in caption_map.values() for c in caps]
    vocab = Vocabulary(freq_threshold=cfg["freq_threshold"])
    vocab.build_vocabulary(all_captions)

    vocab_path = os.path.join(cfg["save_dir"], "vocabulary.pkl")
    vocab.save_vocabulary(vocab_path)

    # ------------------------------------------------------------------
    # 2. Split & DataLoaders
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  STEP 2 — Splits & DataLoaders")
    print("=" * 64)

    train_map, val_map, _test_map = split_dataset(caption_map, seed=cfg["seed"])

    train_ds = FlickrDataset(img_dir, train_map, vocab,
                             transform=get_transform(train=True),
                             max_length=cfg["max_length"])
    val_ds = FlickrDataset(img_dir, val_map, vocab,
                           transform=get_transform(train=False),
                           max_length=cfg["max_length"])

    if len(train_ds) == 0:
        print("❌ Training dataset is empty. Cannot proceed.")
        sys.exit(1)
    if len(val_ds) == 0:
        print("❌ Validation dataset is empty. Cannot proceed.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"],
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"],
                            pin_memory=True)

    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # 3. Models
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  STEP 3 — Models")
    print("=" * 64)

    encoder = Encoder(embed_size=cfg["embed_size"]).to(device)
    encoder.eval()

    decoder = Decoder(
        embed_size=cfg["embed_size"],
        hidden_size=cfg["hidden_size"],
        vocab_size=len(vocab),
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    # ------------------------------------------------------------------
    # 4. Optimizer / Scheduler / Scaler
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg["scheduler_patience"],
        factor=0.5)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # Resume
    if resume_path and os.path.isfile(resume_path):
        print(f"\n  Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  STEP 4 — Training")
    print("=" * 64)

    train_losses, val_losses = [], []

    for epoch in range(start_epoch, cfg["num_epochs"]):
        decoder.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")
        for batch_idx, (images, captions) in enumerate(pbar):
            images, captions = images.to(device), captions.to(device)

            with torch.no_grad():
                features = encoder(images)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = decoder(features, captions)
                targets = captions[:, 1:]
                loss = criterion(outputs.reshape(-1, outputs.size(2)),
                                 targets.reshape(-1))

            if math.isnan(loss.item()):
                print(f"\n❌ NaN loss at epoch {epoch+1}, batch {batch_idx}. Aborting.")
                return

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # ---- Validation ----
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in val_loader:
                images, captions = images.to(device), captions.to(device)
                features = encoder(images)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = decoder(features, captions)
                    targets = captions[:, 1:]
                    loss = criterion(outputs.reshape(-1, outputs.size(2)),
                                     targets.reshape(-1))
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1}/{cfg['num_epochs']}]  "
              f"Train: {avg_train:.4f}  Val: {avg_val:.4f}  LR: {lr_now:.6f}")

        # BLEU every 3 epochs + last
        if (epoch + 1) % 3 == 0 or (epoch + 1) == cfg["num_epochs"]:
            try:
                bleu = evaluate_bleu(encoder, decoder, val_ds, vocab, device, 200)
                print_bleu(bleu, prefix=f"Epoch {epoch+1}")
            except Exception as e:
                print(f"  ⚠ BLEU eval skipped: {e}")

        scheduler.step(avg_val)

        # ---- Checkpoint ----
        ckpt = {
            "epoch": epoch + 1,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train, "val_loss": avg_val,
            "best_val_loss": best_val_loss,
            "vocab_size": len(vocab),
            "embed_size": cfg["embed_size"],
            "hidden_size": cfg["hidden_size"],
            "num_layers": cfg["num_layers"],
            "config": cfg,
        }
        torch.save(ckpt, os.path.join(cfg["save_dir"], "checkpoint_latest.pth"))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, os.path.join(cfg["save_dir"], "best_model.pth"))
            print(f"  ★ Best model saved (val_loss={best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{cfg['early_stop_patience']})")

        if patience_counter >= cfg["early_stop_patience"]:
            print(f"\n⏹ Early stopping after {epoch+1} epochs.")
            break

    # ------------------------------------------------------------------
    # 6. Final evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  STEP 5 — Final Evaluation")
    print("=" * 64)

    best_path = os.path.join(cfg["save_dir"], "best_model.pth")
    if os.path.isfile(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        decoder.load_state_dict(best_ckpt["decoder_state_dict"])
        print(f"Loaded best model from epoch {best_ckpt['epoch']}")

    try:
        bleu = evaluate_bleu(encoder, decoder, val_ds, vocab, device, 500)
        print_bleu(bleu, prefix="Final Val")
    except Exception as e:
        print(f"  ⚠ Final BLEU failed: {e}")

    # Loss plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(train_losses)+1), train_losses, "o-", label="Train")
        ax.plot(range(1, len(val_losses)+1), val_losses, "s-", label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss"); ax.legend(); ax.grid(True)
        plot_path = os.path.join(cfg["save_dir"], "training_loss.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Loss plot → {plot_path}")
    except Exception:
        pass

    print("\n" + "=" * 64)
    print("  ✅ Training Complete!")
    print("=" * 64)
    print(f"  Best model  : {best_path}")
    print(f"  Vocabulary  : {vocab_path}")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    print("=" * 64)
    print("  IMAGE CAPTIONING — TRAINING")
    print("=" * 64)

    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    print("\nConfiguration:")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")

    train_model(cfg, resume_path=args.resume)
