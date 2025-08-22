# config.py - Configuration management
import yaml
import os
from dataclasses import dataclass, asdict
from typing import List, Optional
import glob
import shutil
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from functools import partial

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class TrainingConfig:
    # Model parameters
    channels: int = 3
    filters: int = 64
    num_res_blocks: int = 23
    num_upsample: int = 2
    use_attention: bool = True
    use_spectral_norm: bool = False
    
    # Training parameters
    n_epochs: int = 200
    batch_size: int = 4
    lr: float = 1e-4
    b1: float = 0.9
    b2: float = 0.999
    decay_epoch: int = 100
    
    # Loss weights
    lambda_adv: float = 5e-3
    lambda_pixel: float = 1e-2
    lambda_perceptual: float = 1.0
    lambda_tv: float = 1e-6
    
    # Dataset parameters
    hr_height: int = 256
    hr_width: int = 256
    dataset_name: str = "img_align_celeba"
    augment: bool = True
    degradation_mode: str = "bicubic"
    scale_factors: List[int] = None
    
    # Training features
    warmup_batches: int = 500
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    ema_decay: float = 0.999
    
    # Logging and checkpoints
    sample_interval: int = 100
    checkpoint_interval: int = 500
    use_wandb: bool = False
    
    # Hardware
    n_cpu: int = 8
    device: str = "cuda"
    
    def __post_init__(self):
        if self.scale_factors is None:
            self.scale_factors = [4]
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        # Resolve to photo_enhancement directory if relative
        resolved_path = yaml_path if os.path.isabs(yaml_path) else os.path.join(BASE_DIR, yaml_path)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        
        with open(resolved_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

# utils.py - Utility functions
def setup_directories(base_path: str = None):
    """Setup required directories for training"""
    if base_path is None:
        base_path = BASE_DIR
    dirs = [
        "data",
        "images/training", 
        "images/validation",
        "saved_models",
        "checkpoints",
        "results",
        "logs",
        "configs"  # Added configs directory
    ]
    
    for dir_name in dirs:
        full_path = os.path.join(base_path, dir_name)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def print_model_info(model, model_name="Model"):
    """Print comprehensive model information"""
    total_params = count_parameters(model)
    model_size = calculate_model_size(model)
    
    print(f"\n{model_name} Information:")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    print(f"{'='*50}\n")

def save_checkpoint(epoch, model_g, model_d, optimizer_g, optimizer_d, 
                   scheduler_g, scheduler_d, loss_g, loss_d, metrics, 
                   checkpoint_path, is_best=False):
    """Save training checkpoint with all necessary information"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': model_g.state_dict(),
        'discriminator_state_dict': model_d.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'scheduler_g_state_dict': scheduler_g.state_dict(),
        'scheduler_d_state_dict': scheduler_d.state_dict(),
        'loss_g': loss_g,
        'loss_d': loss_d,
        'metrics': metrics,
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.replace('.pth', '_best.pth')
        shutil.copyfile(checkpoint_path, best_path)

def load_checkpoint(checkpoint_path, model_g, model_d, optimizer_g, 
                   optimizer_d, scheduler_g, scheduler_d, device):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_g.load_state_dict(checkpoint['generator_state_dict'])
    model_d.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
    scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
    
    epoch = checkpoint['epoch']
    loss_g = checkpoint['loss_g']
    loss_d = checkpoint['loss_d']
    metrics = checkpoint.get('metrics', {})
    
    return epoch, loss_g, loss_d, metrics

def visualize_training_progress(log_file, output_path="training_progress.png"):
    """Visualize training progress from log file"""
    # This would parse your log file and create visualizations
    # Implementation depends on your logging format
    pass

def create_learning_rate_plot(lr_schedule, epochs, output_path="lr_schedule.png"):
    """Create learning rate schedule visualization"""
    plt.figure(figsize=(10, 6))
    epochs_list = list(range(epochs))
    lrs = [lr_schedule(epoch) for epoch in epochs_list]
    
    plt.plot(epochs_list, lrs, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def benchmark_model(model, input_shape=(1, 3, 64, 64), device='cuda', num_runs=100):
    """Benchmark model inference speed"""
    model.eval()
    model = model.to(device)
    
    # Warmup
    dummy_input = torch.randn(*input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    
    if device == 'cuda':
        start_time.record()
    else:
        import time
        start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
    else:
        elapsed_time = time.time() - start_time
    
    avg_time = elapsed_time / num_runs
    fps = 1.0 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    return avg_time, fps

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def validate_model(model, dataloader, criterion, device, max_batches=None):
    """Validate model on validation dataset"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
                
            lr_imgs = batch['lr'].to(device)
            hr_imgs = batch['hr'].to(device)
            
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            
            total_loss += loss.item()
            
            # Calculate PSNR and SSIM here
            # Implementation depends on your metric calculation functions
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0
    avg_ssim = total_ssim / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_psnr, avg_ssim

# data_preparation.py - Data preparation utilities
def resize_images_parallel(input_dir, output_dir, target_size=(256, 256), 
                          quality=95, num_workers=4):
    """Resize images in parallel for faster preprocessing"""
    
    def resize_single_image(img_path, input_dir, output_dir, target_size, quality):
        try:
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load and resize image
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize(target_size, Image.LANCZOS)
            
            # Save with high quality
            img_resized.save(output_path, quality=quality, optimize=True)
            return True
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(input_dir, '**', f'*{ext}')
        image_files.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(input_dir, '**', f'*{ext.upper()}')
        image_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(image_files)} images to resize")
    
    # Process in parallel
    resize_func = partial(resize_single_image, 
                         input_dir=input_dir, 
                         output_dir=output_dir,
                         target_size=target_size, 
                         quality=quality)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(resize_func, image_files),
            total=len(image_files),
            desc="Resizing images"
        ))
    
    successful = sum(results)
    print(f"Successfully resized {successful}/{len(image_files)} images")

def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                 output_dir="split_data", seed=42):
    """Split dataset into train/val/test sets"""
    import random
    
    random.seed(seed)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(data_dir, f"*{ext.upper()}")))
    
    random.shuffle(image_files)
    
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Create directories and copy files
    for split_name, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for file_path in tqdm(file_list, desc=f"Copying {split_name} files"):
            filename = os.path.basename(file_path)
            dest_path = os.path.join(split_dir, filename)
            shutil.copy2(file_path, dest_path)
    
    print(f"Dataset split completed:")
    print(f"Train: {len(train_files)} images")
    print(f"Val: {len(val_files)} images") 
    print(f"Test: {len(test_files)} images")

if __name__ == "__main__":
    # Example usage
    print("Setting up Enhanced ESRGAN project...")
    setup_directories()
    
    # Create default config
    config = TrainingConfig()
    config.to_yaml("configs/default_config.yaml")
    print("Default configuration saved to configs/default_config.yaml")
    
    print("\n" + "="*60)
    print("✅ PROJECT SETUP COMPLETE!")
    print("="*60)
    print("📁 Directories created successfully")
    print("⚙️  Configuration file created")
    print("\n🚀 NEXT STEPS:")
    print("1. Add your images to data/your_dataset_name/ folder")
    print("2. Run training: python enhanced_esrgan.py --dataset_name your_dataset_name")
    print("="*60)