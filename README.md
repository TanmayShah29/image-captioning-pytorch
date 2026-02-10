# Image Captioning with PyTorch

End-to-end image captioning on **Flickr8k** using a CNN–LSTM architecture (ResNet50 encoder + LSTM decoder).

## ✨ Key Features

- **Bulletproof dataset handling** — any Flickr8k format auto-detected and normalized
- **Beam search inference** for higher-quality captions
- **BLEU-1→4 evaluation** every 3 epochs
- **Early stopping**, LR scheduling, mixed-precision training
- **Full checkpoint save/resume** — safe for Colab disconnects
- **YAML config** with CLI overrides

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the Flickr8k dataset

Download from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) or any other source and place files under `data/`:

```
data/
├── Flickr8k_Dataset/   ← or Images/, or any folder with .jpg files
├── Flickr8k.token.txt  ← or captions.csv, or any caption format
```

> **Any layout works.** The pipeline auto-detects and normalizes everything.

### 3. Verify dataset (optional but recommended)

```bash
python test_dataset_loading.py
```

### 4. Train

```bash
python train.py
```

### 5. Generate captions

```bash
python inference.py --image path/to/image.jpg --beam_size 3
```

---

## 🔒 Dataset Auto-Normalization

**Training will refuse to start if the dataset is broken.**

When you run `train.py` (or `verify.py` or `test_dataset_loading.py`), the preparation pipeline automatically:

1. **Finds images** in any subfolder under `data/` (Kaggle layout, manual layout, nested folders — all supported)
2. **Finds captions** in any format:
   - `image.jpg#0<TAB>caption` (Flickr8k token format)
   - `image,caption` (CSV with or without header)
   - `image<TAB>caption` (plain TSV)
3. **Copies images** into `data/images/` (flat, one directory)
4. **Writes clean** `data/captions.txt` (tab-separated, no headers, no `#0` suffixes)
5. **Cross-validates** that every caption points to an existing image
6. **Crashes with a clear error** if anything is wrong — telling you WHAT failed, WHY, and HOW to fix it

After preparation, all code reads **only** the canonical format:

```
data/
├── images/           ← all .jpg files here
└── captions.txt      ← image_name<TAB>caption
```

> [!NOTE]
> Once prepared, re-running skips the normalization step (fast path).

---

## 📁 Project Structure

```
├── config.yaml              ← hyperparameters (YAML)
├── train.py                 ← training with val loop, early stopping, AMP
├── inference.py             ← caption generation (greedy + beam search)
├── evaluate.py              ← BLEU-1→4 scoring
├── verify.py                ← 6-check end-to-end verification
├── test_dataset_loading.py  ← dataset dry-run test
├── models/
│   ├── encoder.py           ← ResNet50 feature extractor (frozen)
│   └── decoder.py           ← LSTM + beam search
├── utils/
│   ├── prepare_dataset.py   ← dataset auto-detection & normalization
│   ├── dataset.py           ← PyTorch Dataset (canonical format only)
│   └── vocabulary.py        ← word ↔ index mapping
└── requirements.txt
```

---

## ⚙️ Configuration

All hyperparameters live in `config.yaml` and can be overridden via CLI:

```bash
# Override any parameter
python train.py --num_epochs 20 --learning_rate 0.0005 --batch_size 64

# Resume interrupted training
python train.py --resume saved_models/checkpoint_latest.pth
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 10 | Training epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 0.001 | Initial learning rate |
| `embed_size` | 256 | Embedding dimension |
| `hidden_size` | 512 | LSTM hidden dimension |
| `freq_threshold` | 5 | Min word frequency for vocabulary |
| `early_stop_patience` | 5 | Epochs without improvement before stopping |
| `use_amp` | true | Mixed-precision training |

---

## 📊 Evaluation

BLEU scores are computed automatically during training. For standalone evaluation:

```python
from evaluate import evaluate_bleu
bleu = evaluate_bleu(encoder, decoder, val_dataset, vocab, device, max_samples=500)
```

---

## 🧪 Verification

```bash
# Full pipeline check (6 automated tests)
python verify.py

# Dataset-only dry run (5 samples)
python test_dataset_loading.py
```

---

## License

MIT
