#!/usr/bin/env python3
"""
Quick test script for Enhanced ESRGAN components
This tests individual components without full training
"""

import torch
import torch.nn as nn
from enhanced_models import EnhancedGeneratorRRDB, EnhancedDiscriminator, FeatureExtractor
from enhanced_datasets import create_training_dataset
from torch.utils.data import DataLoader
import numpy as np

def test_models():
    """Test model architectures"""
    print("🧪 Testing Model Architectures...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test Generator
    print("\n1️⃣ Testing Enhanced Generator...")
    generator = EnhancedGeneratorRRDB(
        channels=3, 
        filters=64, 
        num_res_blocks=4,  # Smaller for testing
        num_upsample=2,
        use_attention=True
    ).to(device)
    
    # Test input
    test_lr = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        test_sr = generator(test_lr)
    
    print(f"✅ Generator: {test_lr.shape} -> {test_sr.shape}")
    print(f"   Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Test Discriminator
    print("\n2️⃣ Testing Enhanced Discriminator...")
    discriminator = EnhancedDiscriminator(
        input_shape=(3, 128, 128),
        use_spectral_norm=False,  # Disable for testing
        use_self_attention=False
    ).to(device)
    
    test_hr = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        disc_output = discriminator(test_hr)
    
    print(f"✅ Discriminator: {test_hr.shape} -> {disc_output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Test Feature Extractor
    print("\n3️⃣ Testing Feature Extractor...")
    feature_extractor = FeatureExtractor().to(device)
    
    with torch.no_grad():
        features = feature_extractor(test_hr)
    
    print(f"✅ Feature Extractor: {test_hr.shape} -> {features.shape}")
    
    return generator, discriminator, feature_extractor

def test_dataset():
    """Test dataset loading"""
    print("\n🗂️  Testing Dataset...")
    
    try:
        dataset = create_training_dataset(
            root='data/sample_dataset',
            hr_shape=(128, 128),
            scale_factor=4,
            augment=False  # Disable augmentation for testing
        )
        
        print(f"✅ Dataset loaded with {len(dataset)} images")
        
        # Test sample loading
        sample = dataset[0]
        print(f"✅ Sample shapes - LR: {sample['lr'].shape}, HR: {sample['hr'].shape}")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        print(f"✅ Batch shapes - LR: {batch['lr'].shape}, HR: {batch['hr'].shape}")
        
        return dataset, dataloader
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return None, None

def test_forward_pass():
    """Test complete forward pass"""
    print("\n⚡ Testing Forward Pass...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple models
    generator = EnhancedGeneratorRRDB(channels=3, filters=32, num_res_blocks=2, num_upsample=2)
    discriminator = EnhancedDiscriminator(input_shape=(3, 128, 128), use_spectral_norm=False, use_self_attention=False)
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Test input
    lr_input = torch.randn(2, 3, 32, 32).to(device)
    hr_target = torch.randn(2, 3, 128, 128).to(device)
    
    # Forward pass
    with torch.no_grad():
        # Generator
        sr_output = generator(lr_input)
        print(f"✅ Generator forward: {lr_input.shape} -> {sr_output.shape}")
        
        # Discriminator
        real_pred = discriminator(hr_target)
        fake_pred = discriminator(sr_output)
        print(f"✅ Discriminator forward: {hr_target.shape} -> {real_pred.shape}")
        print(f"✅ Discriminator forward: {sr_output.shape} -> {fake_pred.shape}")
    
    # Test loss calculation
    criterion = nn.BCEWithLogitsLoss()
    valid = torch.ones_like(real_pred)
    fake = torch.zeros_like(fake_pred)
    
    d_loss_real = criterion(real_pred, valid)
    d_loss_fake = criterion(fake_pred, fake)
    d_loss = (d_loss_real + d_loss_fake) / 2
    
    print(f"✅ Loss calculation successful: D_loss = {d_loss.item():.4f}")

def test_inference():
    """Test inference on actual images"""
    print("\n🖼️  Testing Inference...")
    
    try:
        from PIL import Image
        import torchvision.transforms as transforms
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create simple generator
        generator = EnhancedGeneratorRRDB(
            channels=3, 
            filters=32, 
            num_res_blocks=2, 
            num_upsample=2
        ).to(device)
        generator.eval()
        
        # Load a sample image
        img_path = 'data/sample_dataset/sample_000.jpg'
        img = Image.open(img_path).convert('RGB')
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        lr_tensor = transform(img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            sr_tensor = generator(lr_tensor)
        
        print(f"✅ Inference successful: {lr_tensor.shape} -> {sr_tensor.shape}")
        
        # Denormalize and save result
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        sr_denorm = sr_tensor * std + mean
        sr_denorm = torch.clamp(sr_denorm, 0, 1)
        
        # Convert to PIL and save
        sr_pil = transforms.ToPILImage()(sr_denorm.squeeze(0).cpu())
        sr_pil.save('results/test_output.png')
        print("✅ Test output saved to results/test_output.png")
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Enhanced ESRGAN Component Testing")
    print("=" * 50)
    
    # Test 1: Model architectures
    models = test_models()
    
    # Test 2: Dataset loading
    dataset_result = test_dataset()
    
    # Test 3: Forward pass
    test_forward_pass()
    
    # Test 4: Inference
    test_inference()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("=" * 50)
    
    print("\n💡 Next Steps:")
    print("1. Add your training images to data/your_dataset_name/")
    print("2. Run full training: python enhanced_esrgan.py --dataset_name your_dataset_name")
    print("3. For testing trained models: python enhanced_test.py --model_path saved_models/best_generator.pth --lr_dir test_images/")

if __name__ == "__main__":
    main()
