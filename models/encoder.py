"""
CNN Encoder for Image Captioning

This module extracts visual features from images using a pre-trained ResNet50.

Key Concepts:
- Transfer Learning: Using a model trained on ImageNet (1.2M images)
- Feature Extraction: Converting images to fixed-size feature vectors
- Frozen Weights: We don't train the CNN, only use it for features

Architecture:
    Input Image (224×224×3)
        ↓
    ResNet50 (pre-trained on ImageNet)
        ↓
    Remove final classification layer
        ↓
    Feature Vector (2048 dimensions)

Why ResNet50?
- Pre-trained on ImageNet dataset
- Excellent at extracting visual features
- Saves training time and computational resources
- Proven performance in computer vision tasks
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Import for PyTorch 1.13+ compatibility
try:
    from torchvision.models import ResNet50_Weights
    WEIGHTS_AVAILABLE = True
except ImportError:
    WEIGHTS_AVAILABLE = False


class Encoder(nn.Module):
    """
    CNN-based image encoder using pre-trained ResNet50.
    
    What is nn.Module?
    - Base class for all neural network modules in PyTorch
    - Must implement __init__ and forward methods
    
    Attributes:
        resnet (ResNet): Pre-trained ResNet50 model
        embed_size (int): Size of output feature vector
    """
    
    def __init__(self, embed_size=256):
        """
        Initialize encoder with pre-trained ResNet50.
        
        Args:
            embed_size (int): Dimension of output features
                             Default: 256 (can be adjusted)
        
        Process:
            1. Load pre-trained ResNet50
            2. Remove final classification layer
            3. Freeze all parameters (no training)
            4. Add custom linear layer for embedding
        
        Why embed_size?
        - ResNet50 outputs 2048 features
        - We reduce to embed_size (e.g., 256) for efficiency
        - This matches the LSTM input size
        """
        super(Encoder, self).__init__()
        
        # Load pre-trained ResNet50
        # Use weights parameter for PyTorch 1.13+, fallback to pretrained for older versions
        if WEIGHTS_AVAILABLE:
            # PyTorch 1.13+ (recommended)
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            # Older PyTorch versions (backward compatibility)
            resnet = models.resnet50(pretrained=True)
        
        # Remove the final classification layer (fc layer)
        # ResNet50 architecture:
        #   - conv layers (feature extraction) ✓ Keep this
        #   - avgpool (global average pooling) ✓ Keep this
        #   - fc (classification layer) ✗ Remove this
        # We only need feature extraction, not classification
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze all ResNet parameters
        # Why freeze? We use pre-trained features, don't need to retrain
        # This is transfer learning: reuse learned features
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Linear layer to project ResNet features to embed_size
        # Input: 2048 (ResNet50 output)
        # Output: embed_size (e.g., 256)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        
        # Batch normalization for stable training
        # Normalizes features to have mean=0, std=1
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        print(f"Encoder initialized with embed_size={embed_size}")
        print(f"ResNet50 parameters frozen (transfer learning)")
    
    def forward(self, images):
        """
        Forward pass: Extract features from images.
        
        Args:
            images (Tensor): Batch of images
                           Shape: (batch_size, 3, 224, 224)
                           - batch_size: number of images
                           - 3: RGB channels
                           - 224×224: image dimensions
        
        Returns:
            features (Tensor): Image features
                             Shape: (batch_size, embed_size)
        
        Process:
            1. Pass images through ResNet50
            2. Flatten output
            3. Project to embed_size
            4. Apply batch normalization
        
        Example:
            Input: (32, 3, 224, 224)  # 32 images
            After ResNet: (32, 2048, 1, 1)
            After flatten: (32, 2048)
            After linear: (32, 256)
            Output: (32, 256)
        """
        # Set to evaluation mode (important for frozen model)
        # This disables dropout and uses running stats for batch norm
        with torch.no_grad():
            # Extract features using ResNet50
            # Output shape: (batch_size, 2048, 1, 1)
            features = self.resnet(images)
        
        # Reshape from (batch_size, 2048, 1, 1) to (batch_size, 2048)
        # This removes the spatial dimensions (1×1)
        features = features.reshape(features.size(0), -1)
        
        # Project to embed_size
        # (batch_size, 2048) -> (batch_size, embed_size)
        features = self.linear(features)
        
        # Apply batch normalization
        # Helps with training stability
        features = self.bn(features)
        
        return features


# Example usage (for testing)
if __name__ == "__main__":
    # Create encoder
    encoder = Encoder(embed_size=256)
    
    # Create dummy batch of images
    # Shape: (batch_size=4, channels=3, height=224, width=224)
    dummy_images = torch.randn(4, 3, 224, 224)
    
    # Extract features
    features = encoder(dummy_images)
    
    print(f"\nInput shape: {dummy_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: (4, 256)")
    
    # Verify frozen parameters
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in encoder.parameters())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print("\n✓ Encoder test passed!")
