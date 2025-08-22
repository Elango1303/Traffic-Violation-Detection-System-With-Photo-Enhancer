# 🚗📸 Traffic Violation Detection System with Photo Enhancer

**An advanced super-resolution system using Enhanced ESRGAN for improving low-quality traffic violation images**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

This project implements a state-of-the-art **Enhanced Super-Resolution Generative Adversarial Network (ESRGAN)** specifically designed to improve the quality of low-resolution traffic violation images. By enhancing image clarity and detail, this system enables more accurate traffic violation detection and analysis.

### ✨ Key Features

- **🔥 Enhanced ESRGAN Architecture**: Advanced generator with self-attention and spectral normalization
- **📈 4x Super-Resolution**: Upscale images from low-resolution to high-quality outputs
- **🎯 Traffic-Focused**: Optimized for traffic violation detection scenarios
- **⚡ Real-time Processing**: Fast inference for practical deployment
- **📊 Comprehensive Metrics**: PSNR, SSIM, LPIPS evaluation framework
- **🔧 Production Ready**: Complete training and testing pipeline

## 🏗️ Architecture

### Enhanced Generator Features:
- **Residual-in-Residual Dense Blocks (RRDB)**
- **Self-Attention Mechanisms**
- **Multi-scale Feature Extraction**
- **Advanced Upsampling Layers**

### Enhanced Discriminator Features:
- **Spectral Normalization**
- **Progressive Feature Learning**
- **Relativistic Loss Functions**

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA 11.0+ (optional, for GPU acceleration)
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Elango1303/Traffic-Violation-Detection-System-With-Photo-Enhancer.git
cd Traffic-Violation-Detection-System-Photo-Enhancer/photo_enhancement
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Quick system test:**
```bash
python quick_test.py
```

### 🎮 Usage Examples

#### 1. **Component Testing**
```bash
# Test all components
python quick_test.py

# Check project status
python project_status.py
```

#### 2. **Training the Model**
```bash
# Basic training
python enhanced_esrgan.py --dataset_name your_dataset --n_epochs 100

# Advanced training with monitoring
python enhanced_esrgan.py --dataset_name your_dataset \
                         --batch_size 4 \
                         --use_wandb \
                         --mixed_precision \
                         --gradient_clipping 0.5
```

#### 3. **Image Enhancement (Inference)**
```bash
# Enhance a directory of images
python enhanced_test.py --model_path saved_models/best_generator.pth \
                       --lr_dir test_images/ \
                       --output_dir enhanced_output/

# Single image enhancement with plots
python enhanced_test.py --model_path saved_models/best_generator.pth \
                       --single_image test.jpg \
                       --create_plots
```

## 📁 Project Structure

```
Traffic-Violation-Detection-System-Photo-Enhancer/
│
├── photo_enhancement/
│   ├── config_and_utils.py          # Configuration and utilities
│   ├── enhanced_models.py           # Enhanced ESRGAN architectures
│   ├── enhanced_datasets.py         # Data loading and preprocessing
│   ├── enhanced_esrgan.py          # Main training script
│   ├── enhanced_test.py            # Testing and evaluation
│   ├── quick_test.py               # Quick system validation
│   ├── project_status.py           # Project status checker
│   ├── requirements.txt            # Python dependencies
│   │
│   ├── configs/                    # Configuration files
│   ├── data/                      # Training datasets
│   ├── saved_models/              # Trained model weights
│   ├── results/                   # Output results
│   ├── images/                    # Training samples
│   └── logs/                      # Training logs
│
└── README.md                       # This file
```

## 🎯 Model Performance

### Current Results
- **Processing Speed**: ~0.18 FPS on CPU
- **Model Size**: ~39.5 MB (Generator)
- **Upscaling Factor**: 4x
- **Supported Formats**: JPG, PNG, JPEG

### Evaluation Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **MSE/MAE**: Mean Squared/Absolute Error

## 📊 Dataset Support

### Supported Datasets
- **CelebA**: Celebrity faces (202,599 images) ✅
- **Custom Traffic Data**: Add your own traffic violation images
- **Sample Dataset**: 5 test images included

### Dataset Format
```
data/
├── your_dataset/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
```

## 🔧 Configuration

### Training Configuration
```python
# Key parameters in enhanced_esrgan.py
--batch_size 4              # Batch size for training
--n_epochs 200              # Number of training epochs
--lr 0.0001                 # Learning rate
--residual_blocks 23        # Number of RRDB blocks
--hr_height 256             # High-resolution image height
--hr_width 256              # High-resolution image width
--mixed_precision           # Use mixed precision training
--use_wandb                 # Enable Weights & Biases logging
```

## 🚀 Advanced Features

### 1. **Mixed Precision Training**
Accelerate training with automatic mixed precision:
```bash
python enhanced_esrgan.py --mixed_precision --gradient_clipping 1.0
```

### 2. **Experiment Tracking**
Monitor training with Weights & Biases:
```bash
python enhanced_esrgan.py --use_wandb
```

### 3. **Model Evaluation**
Comprehensive testing framework:
```bash
python enhanced_test.py --create_plots --results_dir evaluation/
```

## 📈 Results Gallery

### Sample Enhancement Results
*Coming soon - Enhanced traffic violation images*

## 🛠️ Development

### Adding Custom Datasets
1. Place images in `data/your_dataset_name/`
2. Update training script:
```bash
python enhanced_esrgan.py --dataset_name your_dataset_name
```

### Model Customization
Modify `enhanced_models.py` to adjust:
- Network architecture
- Attention mechanisms
- Loss functions

## 🚦 Traffic Violation Use Cases

### Primary Applications
- **License Plate Enhancement**: Improve readability of blurry license plates
- **Vehicle Detail Enhancement**: Clarify vehicle make, model, color
- **Scene Reconstruction**: Enhance overall image quality for evidence
- **Real-time Processing**: On-the-fly enhancement in traffic monitoring systems

### Integration Examples
```python
from enhanced_models import EnhancedGeneratorRRDB
import torch

# Load trained model
model = EnhancedGeneratorRRDB(channels=3, filters=64, num_res_blocks=6)
model.load_state_dict(torch.load('saved_models/best_generator.pth'))

# Enhance traffic violation image
enhanced_image = model(low_res_image)
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the deep learning framework
- **ESRGAN Authors** for the original architecture
- **CelebA Dataset** for training data
- **Community Contributors** for improvements and feedback

## 📞 Contact & Support

- **Author**: Elango
- **Repository**: [Traffic-Violation-Detection-System-With-Photo-Enhancer](https://github.com/Elango1303/Traffic-Violation-Detection-System-With-Photo-Enhancer)
- **Issues**: [GitHub Issues](https://github.com/Elango1303/Traffic-Violation-Detection-System-With-Photo-Enhancer/issues)

---

### 🎉 **Ready to enhance your traffic violation detection system? Get started now!**

```bash
git clone https://github.com/Elango1303/Traffic-Violation-Detection-System-With-Photo-Enhancer.git
cd Traffic-Violation-Detection-System-Photo-Enhancer/photo_enhancement
pip install -r requirements.txt
python quick_test.py
```

**Made with ❤️ for better traffic safety**
