# Image Captioning with PyTorch

A beginner-friendly implementation of image captioning using CNN+LSTM architecture with transfer learning on the Flickr8k dataset.

## 📋 Project Overview

### What is Image Captioning?

Image captioning is the task of automatically generating textual descriptions for images. It combines:
- **Computer Vision**: Understanding image content
- **Natural Language Processing**: Generating descriptive text

### Real-World Applications

- 🦾 **Assistive Technology**: Helping visually impaired users understand images
- 📱 **Social Media**: Automatic alt-text generation for accessibility
- 🔍 **Search Engines**: Content indexing and image search
- 🏥 **Medical Imaging**: Automated report generation

### Architecture: CNN + LSTM

This project uses a two-part architecture:

1. **CNN Encoder (ResNet50)**
   - Pre-trained on ImageNet (1.2M images)
   - Extracts visual features from images
   - **Frozen weights** (transfer learning)

2. **LSTM Decoder**
   - Generates captions word-by-word
   - **Trained from scratch** on Flickr8k
   - Uses teacher forcing during training

> **Important for Viva**: You trained the LSTM decoder from scratch. The CNN uses transfer learning (pre-trained weights). This is a standard and academically valid approach in deep learning.

---

## 🗂️ Project Structure

```
image_captioning/
│
├── data/
│   ├── Flickr8k_Dataset/          # Images (8,000 images)
│   └── Flickr8k_text/              # Captions text files
│
├── models/
│   ├── encoder.py                  # CNN feature extractor (ResNet50)
│   └── decoder.py                  # LSTM caption generator
│
├── utils/
│   ├── vocabulary.py               # Vocabulary builder
│   └── dataset.py                  # Dataset loader
│
├── train.py                        # Training script
├── inference.py                    # Caption generation script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📦 Dataset: Flickr8k

### Why Flickr8k?

- ✅ **Beginner-friendly**: Only 8,000 images (manageable size)
- ✅ **CPU/Colab compatible**: Can train without expensive GPU
- ✅ **Quality captions**: 5 captions per image
- ✅ **Academic standard**: Widely used in research

### Download Dataset

1. **Download from Kaggle**:
   ```bash
   # Visit: https://www.kaggle.com/datasets/adityajn105/flickr8k
   # Download both:
   # - Flickr8k_Dataset.zip (images)
   # - Flickr8k_text.zip (captions)
   ```

2. **Extract to project**:
   ```bash
   # Extract images to:
   data/Flickr8k_Dataset/
   
   # Extract captions to:
   data/Flickr8k_text/
   ```

3. **Verify structure**:
   ```
   data/
   ├── Flickr8k_Dataset/
   │   ├── 1000268201_693b08cb0e.jpg
   │   ├── 1001773457_577c3a7d70.jpg
   │   └── ...
   └── Flickr8k_text/
       ├── Flickr8k.token.txt
       ├── Flickr_8k.trainImages.txt
       └── Flickr_8k.testImages.txt
   ```

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
# Clone or download this project
cd image_captioning

# Install required packages
pip install -r requirements.txt
```

### Dependencies

- `torch`: PyTorch deep learning framework
- `torchvision`: Pre-trained models and image transformations
- `Pillow`: Image processing
- `numpy`: Numerical operations
- `matplotlib`: Visualization
- `tqdm`: Progress bars

---

## 🎓 Training

### Quick Start

```bash
python train.py
```

### Training Configuration

Edit `train.py` to customize:

```python
config = {
    "data_dir": "data",
    "num_epochs": 10,          # Increase for better results (20-30)
    "batch_size": 32,          # Reduce if out of memory (16 or 8)
    "learning_rate": 0.001,
    "embed_size": 256,
    "hidden_size": 512,
    "num_layers": 1,
    "save_dir": "saved_models"
}
```

### Training Process

The training script will:

1. ✅ Load Flickr8k dataset
2. ✅ Build vocabulary from captions
3. ✅ Initialize encoder (frozen ResNet50) and decoder (trainable LSTM)
4. ✅ Train decoder using teacher forcing
5. ✅ Save model checkpoints after each epoch
6. ✅ Generate training loss plot

### Expected Training Time

- **CPU**: ~30 minutes per epoch
- **GPU**: ~5 minutes per epoch
- **Google Colab (free GPU)**: ~5 minutes per epoch

### Output Files

After training, you'll find:

```
saved_models/
├── final_model.pth           # Trained model
├── vocabulary.pkl            # Vocabulary object
├── training_loss.png         # Loss plot
└── checkpoint_epoch_*.pth    # Checkpoints
```

---

## 🔮 Inference (Testing)

### Generate Caption for Single Image

```bash
python inference.py --image path/to/your/image.jpg
```

### Save Result

```bash
python inference.py --image path/to/image.jpg --save result.png
```

### Use Specific Checkpoint

```bash
python inference.py --image test.jpg --model saved_models/checkpoint_epoch_5.pth
```

### Example

```bash
python inference.py --image data/Flickr8k_Dataset/1000268201_693b08cb0e.jpg
```

Output:
```
Using device: cpu

Loading model and vocabulary...
Model loaded from epoch 10
Vocabulary size: 2538

Generating caption for: data/Flickr8k_Dataset/1000268201_693b08cb0e.jpg

Generated Caption: a child in a pink dress is climbing up a set of stairs
```

---

## 🎯 Key Concepts Explained

### 1. Transfer Learning

**What**: Using a pre-trained model (ResNet50) trained on ImageNet

**Why**: 
- Saves training time
- Leverages learned visual features
- Works well with limited data

**In this project**:
- CNN encoder is pre-trained and frozen
- Only LSTM decoder is trained

### 2. Teacher Forcing

**What**: During training, feed the actual previous word (not predicted)

**Why**: Helps model learn faster and more stably

**Example**:
```
Actual caption: "a dog running"
Step 1: Input <start> → Predict "a"
Step 2: Input "a" (actual) → Predict "dog"
Step 3: Input "dog" (actual) → Predict "running"
```

### 3. Greedy Decoding

**What**: At each step, pick word with highest probability

**Why**: Simple and fast for inference

**Example**:
```
Step 1: Input <start> → Pick "a" (highest prob)
Step 2: Input "a" → Pick "dog" (highest prob)
Step 3: Input "dog" → Pick "running" (highest prob)
Step 4: Input "running" → Pick <end> (stop)
```

### 4. Special Tokens

- `<start>`: Beginning of caption
- `<end>`: End of caption
- `<pad>`: Padding for shorter captions
- `<unk>`: Unknown words (not in vocabulary)

### 5. Vocabulary Building

**Process**:
1. Tokenize all captions (split into words)
2. Count word frequencies
3. Keep words appearing ≥ 5 times
4. Map words to numerical indices

**Why**: Neural networks work with numbers, not text

---

## 📊 Model Architecture

### Encoder (CNN)

```
Input Image (224×224×3)
    ↓
ResNet50 (pre-trained, frozen)
    ↓
Remove final classification layer
    ↓
Linear projection
    ↓
Feature Vector (256 dimensions)
```

### Decoder (LSTM)

```
Image Features (256)
    ↓
Word Embeddings (vocab_size → 256)
    ↓
LSTM (256 → 512)
    ↓
Fully Connected (512 → vocab_size)
    ↓
Softmax → Predicted Word
```

### Parameters

- **Encoder**: ~23M parameters (frozen)
- **Decoder**: ~5M parameters (trainable)
- **Total trainable**: ~5M parameters

---

## 🎤 Viva Preparation

See [VIVA_GUIDE.md](VIVA_GUIDE.md) for:
- Common viva questions with answers
- One-minute project explanation
- Block diagram description
- Technical concepts explained simply

---

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size in `train.py`
```python
"batch_size": 16  # or 8
```

### Issue: Dataset not found

**Solution**: Verify dataset structure
```bash
ls data/Flickr8k_Dataset/
ls data/Flickr8k_text/
```

### Issue: Slow training

**Solution**: Use Google Colab for free GPU
1. Upload project to Google Drive
2. Open Colab notebook
3. Mount Drive and run training

### Issue: Poor caption quality

**Solution**: Train for more epochs
```python
"num_epochs": 20  # or 30
```

---

## 📚 Learning Resources

### Understanding the Code

Each file has extensive comments explaining:
- What each function does
- Why it's needed
- Input/output shapes
- Key concepts

### Recommended Reading

1. **CNN**: [CS231n Convolutional Neural Networks](http://cs231n.github.io/)
2. **LSTM**: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
3. **Transfer Learning**: [CS231n Transfer Learning](http://cs231n.github.io/transfer-learning/)

---

## 🎓 Academic Honesty

### For Viva Defense

✅ **Correct statements**:
- "I used transfer learning for the CNN encoder"
- "I trained the LSTM decoder from scratch"
- "This is a standard approach in deep learning"
- "The encoder extracts features, the decoder generates captions"

❌ **Avoid saying**:
- "I trained the entire model from scratch"
- "I created ResNet50"

### What You Actually Trained

- ✅ LSTM decoder (~5M parameters)
- ✅ Word embeddings
- ✅ Linear projection layers
- ❌ CNN encoder (pre-trained, frozen)

---

## 📝 License

This project is for educational purposes. Feel free to use for college projects and learning.

---

## 🙏 Acknowledgments

- **Dataset**: Flickr8k by Hodosh et al.
- **Framework**: PyTorch
- **Pre-trained Model**: ResNet50 from torchvision

---

## 📧 Support

For questions or issues:
1. Check [VIVA_GUIDE.md](VIVA_GUIDE.md)
2. Review code comments
3. Consult PyTorch documentation

---

**Good luck with your project and viva! 🎓**
