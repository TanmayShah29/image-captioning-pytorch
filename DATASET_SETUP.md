# Dataset Setup Instructions

## 📥 Downloading Flickr8k Dataset

### Option 1: Kaggle (Recommended)

1. **Visit Kaggle**:
   - Go to: https://www.kaggle.com/datasets/adityajn105/flickr8k
   - Sign in or create a free account

2. **Download Files**:
   - Click "Download" button
   - You'll get: `archive.zip` (~1.2 GB)

3. **Extract Files**:
   ```bash
   # Extract the downloaded zip
   unzip archive.zip
   
   # You should see:
   # - Flicker8k_Dataset/ (folder with images)
   # - Flickr8k.token.txt (captions file)
   # - Other text files
   ```

4. **Organize in Project**:
   ```bash
   # Create data directories
   mkdir -p data/Flickr8k_Dataset
   mkdir -p data/Flickr8k_text
   
   # Move images
   mv Flicker8k_Dataset/* data/Flickr8k_Dataset/
   
   # Move caption files
   mv Flickr8k.token.txt data/Flickr8k_text/
   mv Flickr_8k.trainImages.txt data/Flickr8k_text/
   mv Flickr_8k.testImages.txt data/Flickr8k_text/
   mv Flickr_8k.devImages.txt data/Flickr8k_text/
   ```

### Option 2: Direct Download

If Kaggle is not accessible:

1. **Alternative sources**:
   - University of Illinois: http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html
   - GitHub mirrors (search for "Flickr8k dataset")

2. **Same extraction process** as Option 1

---

## 📂 Expected Directory Structure

After setup, your project should look like this:

```
image_captioning/
│
├── data/
│   ├── Flickr8k_Dataset/
│   │   ├── 1000268201_693b08cb0e.jpg
│   │   ├── 1001773457_577c3a7d70.jpg
│   │   ├── ... (8,000 images total)
│   │
│   └── Flickr8k_text/
│       ├── Flickr8k.token.txt          # All captions
│       ├── Flickr_8k.trainImages.txt   # Training split
│       ├── Flickr_8k.testImages.txt    # Test split
│       └── Flickr_8k.devImages.txt     # Validation split
│
├── models/
│   ├── __init__.py
│   ├── encoder.py
│   └── decoder.py
│
├── utils/
│   ├── __init__.py
│   ├── vocabulary.py
│   └── dataset.py
│
├── train.py
├── inference.py
├── requirements.txt
├── README.md
├── VIVA_GUIDE.md
└── DATASET_SETUP.md (this file)
```

---

## ✅ Verification

### Check Dataset

```bash
# Count images
ls data/Flickr8k_Dataset/*.jpg | wc -l
# Should output: 8091 (or similar, around 8000)

# Check captions file
head -n 5 data/Flickr8k_text/Flickr8k.token.txt
# Should show: image_name#0<tab>caption
```

### Test Loading

```python
# Quick test
import os

# Check images
img_dir = "data/Flickr8k_Dataset"
images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
print(f"Found {len(images)} images")

# Check captions
captions_file = "data/Flickr8k_text/Flickr8k.token.txt"
with open(captions_file, 'r') as f:
    lines = f.readlines()
print(f"Found {len(lines)} captions")
```

Expected output:
```
Found 8091 images
Found 40455 captions
```

---

## 🔧 Troubleshooting

### Issue: "Dataset not found"

**Solution**:
```bash
# Verify paths
ls data/Flickr8k_Dataset/
ls data/Flickr8k_text/

# If empty, re-extract files
```

### Issue: "Permission denied"

**Solution**:
```bash
# Fix permissions
chmod -R 755 data/
```

### Issue: "Wrong directory structure"

**Solution**:
```bash
# Remove and recreate
rm -rf data/
mkdir -p data/Flickr8k_Dataset
mkdir -p data/Flickr8k_text

# Re-extract files
```

---

## 📊 Dataset Statistics

- **Total Images**: 8,091
- **Total Captions**: 40,455 (5 per image)
- **Training Images**: ~6,000
- **Validation Images**: ~1,000
- **Test Images**: ~1,000
- **Vocabulary Size**: ~8,000 unique words
- **Average Caption Length**: 10-15 words
- **Dataset Size**: ~1.2 GB

---

## 🎓 Dataset Information

### Caption Format

```
# File: Flickr8k.token.txt
# Format: image_name#caption_number<tab>caption_text

1000268201_693b08cb0e.jpg#0    A child in a pink dress is climbing up a set of stairs
1000268201_693b08cb0e.jpg#1    A girl going into a wooden building
1000268201_693b08cb0e.jpg#2    A little girl climbing into a wooden playhouse
1000268201_693b08cb0e.jpg#3    A little girl climbing the stairs to her playhouse
1000268201_693b08cb0e.jpg#4    A little girl in a pink dress going into a wooden cabin
```

### Split Files

- **Flickr_8k.trainImages.txt**: List of training image filenames
- **Flickr_8k.testImages.txt**: List of test image filenames
- **Flickr_8k.devImages.txt**: List of validation image filenames

---

## 🚀 Next Steps

After dataset setup:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start training**:
   ```bash
   python train.py
   ```

3. **Test inference**:
   ```bash
   python inference.py --image data/Flickr8k_Dataset/1000268201_693b08cb0e.jpg
   ```

---

## 📚 References

- **Dataset Paper**: "Framing Image Description as a Ranking Task" (Hodosh et al., 2013)
- **Kaggle Link**: https://www.kaggle.com/datasets/adityajn105/flickr8k
- **Original Source**: http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/

---

**Dataset setup complete! Ready to train your model! 🎓**
