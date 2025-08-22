#!/usr/bin/env python3
"""
Enhanced ESRGAN Project Status and Summary
Shows the current state of the project and available operations
"""

import os
import torch
import glob
from datetime import datetime

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking Dependencies...")
    dependencies = {
        'torch': torch.__version__,
        'torchvision': None,
        'PIL': None,
        'cv2': None,
        'numpy': None,
        'yaml': None,
        'tqdm': None,
        'matplotlib': None
    }
    
    try:
        import torchvision
        dependencies['torchvision'] = torchvision.__version__
    except ImportError:
        pass
    
    try:
        import PIL
        dependencies['PIL'] = PIL.__version__
    except ImportError:
        pass
        
    try:
        import cv2
        dependencies['cv2'] = cv2.__version__
    except ImportError:
        pass
        
    try:
        import numpy as np
        dependencies['numpy'] = np.__version__
    except ImportError:
        pass
        
    try:
        import yaml
        dependencies['yaml'] = "Available"
    except ImportError:
        pass
        
    try:
        import tqdm
        dependencies['tqdm'] = tqdm.__version__
    except ImportError:
        pass
        
    try:
        import matplotlib
        dependencies['matplotlib'] = matplotlib.__version__
    except ImportError:
        pass
    
    for dep, version in dependencies.items():
        status = f"✅ {version}" if version else "❌ Not found"
        print(f"  {dep}: {status}")
    
    return all(v is not None for v in dependencies.values())

def check_project_structure():
    """Check project directory structure"""
    print("\n📁 Project Structure:")
    
    required_dirs = [
        'data', 'images', 'saved_models', 'checkpoints', 
        'results', 'logs', 'configs'
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            count = len(os.listdir(dir_name))
            print(f"  ✅ {dir_name}/ ({count} items)")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
    
    # Check important files
    important_files = [
        'config_and_utils.py',
        'enhanced_models.py',
        'enhanced_datasets.py',
        'enhanced_esrgan.py',
        'enhanced_test.py',
        'quick_test.py',
        'configs/default_config.yaml'
    ]
    
    print("\n📄 Important Files:")
    for file_name in important_files:
        if os.path.exists(file_name):
            size = os.path.getsize(file_name)
            print(f"  ✅ {file_name} ({size} bytes)")
        else:
            print(f"  ❌ {file_name} (missing)")

def check_datasets():
    """Check available datasets"""
    print("\n🗂️ Available Datasets:")
    
    data_dir = 'data'
    if os.path.exists(data_dir):
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        if subdirs:
            for subdir in subdirs:
                img_count = len(glob.glob(os.path.join(data_dir, subdir, '*.*')))
                print(f"  📁 {subdir}: {img_count} files")
        else:
            print("  ⚠️ No datasets found in data/ directory")
    else:
        print("  ❌ data/ directory not found")

def check_models():
    """Check saved models"""
    print("\n🤖 Saved Models:")
    
    models_dir = 'saved_models'
    if os.path.exists(models_dir):
        model_files = glob.glob(os.path.join(models_dir, '*.pth'))
        
        if model_files:
            for model_file in model_files:
                size = os.path.getsize(model_file) / (1024 * 1024)  # MB
                mtime = datetime.fromtimestamp(os.path.getmtime(model_file))
                print(f"  🔮 {os.path.basename(model_file)}: {size:.1f} MB (modified: {mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print("  ⚠️ No trained models found")
    else:
        print("  ❌ saved_models/ directory not found")

def check_results():
    """Check results and outputs"""
    print("\n📊 Results & Outputs:")
    
    # Check images
    images_dir = 'images'
    if os.path.exists(images_dir):
        for subdir in ['training', 'validation']:
            full_path = os.path.join(images_dir, subdir)
            if os.path.exists(full_path):
                count = len(glob.glob(os.path.join(full_path, '*.*')))
                print(f"  🖼️ {subdir} samples: {count} files")
    
    # Check results
    results_dir = 'results'
    if os.path.exists(results_dir):
        count = len(glob.glob(os.path.join(results_dir, '*.*')))
        print(f"  📈 results/: {count} files")

def show_hardware_info():
    """Show hardware information"""
    print("\n💻 Hardware Information:")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"  🔥 CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"  📱 GPU Devices: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("  ⚠️ Using CPU only - training will be much slower")

def show_usage_examples():
    """Show usage examples"""
    print("\n💡 Usage Examples:")
    print("=" * 50)
    
    print("\n1️⃣ Setup Project (already done):")
    print("   python config_and_utils.py")
    
    print("\n2️⃣ Test Components:")
    print("   python quick_test.py")
    
    print("\n3️⃣ Train Model:")
    print("   python enhanced_esrgan.py --dataset_name sample_dataset --n_epochs 100")
    print("   python enhanced_esrgan.py --dataset_name your_dataset --batch_size 4 --use_wandb")
    
    print("\n4️⃣ Test Trained Model:")
    print("   python enhanced_test.py --model_path saved_models/best_generator.pth --lr_dir test_images/")
    print("   python enhanced_test.py --model_path saved_models/best_generator.pth --single_image test.jpg --create_plots")
    
    print("\n5️⃣ Advanced Training:")
    print("   python enhanced_esrgan.py --dataset_name your_data --mixed_precision --gradient_clipping 0.5 --residual_blocks 23")

def show_project_info():
    """Show general project information"""
    print("🚀 Enhanced ESRGAN Project Status")
    print("=" * 50)
    print("📌 Project: Traffic Violation Detection System - Photo Enhancer")
    print("🏗️ Architecture: Enhanced Super-Resolution GAN (ESRGAN)")
    print("🎯 Purpose: Enhance low-quality images for better traffic violation detection")
    print(f"📅 Status Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main status check function"""
    show_project_info()
    
    # Run all checks
    deps_ok = check_dependencies()
    check_project_structure()
    check_datasets()
    check_models()
    check_results()
    show_hardware_info()
    show_usage_examples()
    
    # Final recommendations
    print("\n🎯 Recommendations:")
    print("=" * 50)
    
    if not deps_ok:
        print("❗ Install missing dependencies: pip install -r requirements.txt")
    
    # Check if we have any datasets
    has_data = os.path.exists('data') and any(os.path.isdir(os.path.join('data', d)) for d in os.listdir('data'))
    if not has_data:
        print("📁 Add your training images to data/your_dataset_name/ folder")
    
    # Check if we have trained models
    has_models = os.path.exists('saved_models') and glob.glob('saved_models/*.pth')
    if not has_models:
        print("🏃 Start training: python enhanced_esrgan.py --dataset_name your_dataset_name")
    else:
        print("✅ You have trained models - you can run inference tests!")
    
    if not torch.cuda.is_available():
        print("⚡ Consider using GPU for faster training - install CUDA-enabled PyTorch")
    
    print("\n🎉 Project is ready for use!")
    print("📚 Run any of the usage examples above to get started")

if __name__ == "__main__":
    main()
