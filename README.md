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

## 🚀 Quick Start (Local)

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

> **Note:** Once prepared, re-running skips the normalization step (fast path).

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

# Google Colab Setup Guide

Complete setup instructions for running this image captioning project in Google Colab.

## ⚡ Quick Start (All-in-One Setup)

### Prerequisites
1. **Enable GPU**: Runtime → Change runtime type → T4 GPU → Save
2. **Get Kaggle API credentials**: 
   - Go to [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token
   - This downloads `kaggle.json`

---

## 🚀 Setup Cells (Run in Order)

### Cell 1: Mount Drive & Check GPU
```python
# ==========================================
# CELL 1: MOUNT DRIVE & CHECK GPU
# ==========================================

from google.colab import drive
drive.mount('/content/drive')

import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: GPU not enabled! Go to Runtime → Change runtime type → GPU")
```

**Expected output:**
```
GPU available: True
GPU: Tesla T4
```

---

### Cell 2: Upload Kaggle Credentials
```python
# ==========================================
# CELL 2: UPLOAD KAGGLE.JSON
# ==========================================

from google.colab import files

print("Upload your kaggle.json file:")
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Test Kaggle API
!kaggle datasets list -s flickr | head -5

print("✓ Kaggle configured!")
```

**Action required:** Click "Choose Files" and select your `kaggle.json`

---

### Cell 3: Clone Repository
```python
# ==========================================
# CELL 3: CLONE REPOSITORY
# ==========================================

%cd /content
!git clone https://github.com/TanmayShah29/image-captioning-pytorch.git
%cd image-captioning-pytorch

!ls -la
```

---

### Cell 4: Install Dependencies
```python
# ==========================================
# CELL 4: INSTALL DEPENDENCIES
# ==========================================

!pip install -q -r requirements.txt

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

print("✓ Dependencies installed")
```

---

### Cell 5: Download Flickr8k Dataset
```python
# ==========================================
# CELL 5: DOWNLOAD FLICKR8K DATASET
# ==========================================

!mkdir -p data

# Download from Kaggle (~1GB, takes 2-3 minutes)
!kaggle datasets download -d adityajn105/flickr8k

# Check download
!ls -lh flickr8k.zip

# Extract
!unzip -q flickr8k.zip -d data/

# Verify extraction
!ls -la data/
!find data/ -type f | head -20
```

**Expected:** Should show `captions.txt` and `Images/` directory

---

### Cell 6: Fix Caption Format
```python
# ==========================================
# CELL 6: FIX CAPTION FORMAT
# ==========================================

import pandas as pd

# Read CSV format
df = pd.read_csv('data/captions.txt')
print(f"Loaded {len(df)} captions")

# Convert to tab-separated format (required by the code)
with open('data/captions_fixed.txt', 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        f.write(f"{row['image']}\t{row['caption']}\n")

# Backup original and replace
!mv data/captions.txt data/captions_original.txt  
!mv data/captions_fixed.txt data/captions.txt

print("✓ Captions converted to tab-separated format")
!head -5 data/captions.txt
```

**Expected output:** Tab-separated format without CSV header

> **Note:** The auto-preparation pipeline (`prepare_dataset.py`) can handle CSV format automatically, but this manual step ensures the cleanest setup.

---

### Cell 7: Setup Checkpoints to Google Drive
```python
# ==========================================
# CELL 7: SETUP CHECKPOINT DIRECTORY
# ==========================================

!mkdir -p /content/drive/MyDrive/image_captioning_checkpoints

# Update config to save checkpoints to Drive (persists after disconnect)
!sed -i 's|saved_models|/content/drive/MyDrive/image_captioning_checkpoints|g' config.yaml

# Verify
!grep save_dir config.yaml

print("✓ Checkpoints will save to Google Drive")
```

**Why?** Colab sessions disconnect after 12 hours. Saving to Drive prevents data loss.

---

### Cell 8: Test Dataset Loading
```python
# ==========================================
# CELL 8: TEST DATASET LOADING
# ==========================================

!python test_dataset_loading.py
```

**Expected:** Should show ✅ for all checks and confirm 8091 images, 40455 captions

---

### Cell 9: Start Training 🚀
```python
# ==========================================
# CELL 9: START TRAINING
# ==========================================

!python train.py --num_epochs 5 --batch_size 16
```

**Training time:**
- Per epoch: ~5-10 minutes (on T4 GPU)
- Total (5 epochs): ~25-50 minutes

**What to expect:**
```
Device: cuda  |  AMP: True
Epoch 1/5: 100%|██████████| 1770/1770 [06:23<00:00]
Train Loss: 3.1234  |  Val Loss: 2.8765
```

---

### Cell 10: Monitor GPU (Optional)
```python
# ==========================================
# CELL 10: MONITOR GPU USAGE
# ==========================================

!nvidia-smi
```

Run this in a separate cell while training to monitor GPU memory and utilization.

---

## 🔄 Quick Restore (For Future Sessions)

After your first successful setup, save the dataset to Drive to avoid re-downloading:

```python
# ONE-TIME: Save dataset to Drive after first setup
!cp -r /content/image-captioning-pytorch/data /content/drive/MyDrive/flickr8k_backup
```

**For future sessions, use this fast restore:**

```python
# ==========================================
# QUICK RESTORE (replaces Cells 3-6)
# ==========================================

from google.colab import drive
drive.mount('/content/drive')

%cd /content
!git clone https://github.com/TanmayShah29/image-captioning-pytorch.git
%cd image-captioning-pytorch

!pip install -q -r requirements.txt

# Restore dataset from Drive (much faster than re-downloading)
!cp -r /content/drive/MyDrive/flickr8k_backup/* data/

# Point checkpoints to Drive
!sed -i 's|saved_models|/content/drive/MyDrive/image_captioning_checkpoints|g' config.yaml

# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

print("✓ Quick restore complete! Ready to train.")
```

---

## 🛠️ Troubleshooting

### Issue 1: "CUDA out of memory"
**Solution:** Reduce batch size
```python
!python train.py --num_epochs 5 --batch_size 8  # or even 4
```

### Issue 2: "Device: cpu | AMP: False"
**Solution:** GPU not enabled
1. Runtime → Change runtime type
2. Hardware accelerator → T4 GPU
3. Save → Runtime will restart

### Issue 3: Session disconnected during training
**Solution:** Resume from checkpoint
```python
!python train.py --resume /content/drive/MyDrive/image_captioning_checkpoints/checkpoint_latest.pth
```

### Issue 4: "No such file or directory: data/captions.txt"
**Solution:** Re-run Cell 5 (dataset download) and Cell 6 (format conversion)

### Issue 5: Kaggle API authentication error
**Solution:** Re-upload kaggle.json
- Make sure you downloaded it from kaggle.com → Settings → API
- Re-run Cell 2

---

## 📊 After Training

### Generate Captions
```python
!python inference.py --image data/images/1000268201_693b08cb0e.jpg --beam_size 3
```

### Evaluate Model (BLEU Scores)
```python
!python evaluate.py
```

### Download Trained Model
```python
from google.colab import files

# Download best model
files.download('/content/drive/MyDrive/image_captioning_checkpoints/best_model.pth')

# Download vocabulary
files.download('/content/drive/MyDrive/image_captioning_checkpoints/vocabulary.pkl')
```

---

## 💡 Tips

1. **Enable GPU before starting** - Training on CPU takes 10-20x longer
2. **Save dataset to Drive** - Saves 5-10 minutes on future runs
3. **Monitor GPU usage** - Run `!nvidia-smi` to check memory
4. **Use beam search** - Beam size 3-5 produces better captions than greedy
5. **Reduce batch size if OOM** - Start with 16, reduce to 8 or 4 if needed
6. **Checkpoints auto-save** - Training can be resumed after disconnect

---

## 📈 Expected Results

After 5 epochs:
- **Training Loss**: ~2.0-2.5
- **Validation Loss**: ~2.3-2.8
- **BLEU-1**: ~0.55-0.65
- **BLEU-4**: ~0.15-0.25

For better results, train for 10-15 epochs.

---

## 🔗 Resources

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [Original Repository](https://github.com/TanmayShah29/image-captioning-pytorch)
- [Google Colab](https://colab.research.google.com/)
- [Kaggle API Setup](https://github.com/Kaggle/kaggle-api)

---

## License

MIT
