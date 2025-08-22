import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 1)

class RandomDegradation:
    """Apply random degradation to simulate real-world low-quality images"""
    def __init__(self, scale_factor=4):
        self.scale_factor = scale_factor
    
    def __call__(self, image):
        # Random blur
        if random.random() < 0.5:
            blur_radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Random noise
        if random.random() < 0.3:
            image_np = np.array(image).astype(np.float32)
            noise_strength = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_strength * 255, image_np.shape)
            image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_np)
        
        # Random JPEG compression
        if random.random() < 0.4:
            import io
            quality = random.randint(30, 85)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            image = Image.open(buffer)
        
        # Random downsampling artifacts
        if random.random() < 0.2:
            # Simulate aliasing effects
            temp_size = (image.size[0] // 2, image.size[1] // 2)
            image = image.resize(temp_size, Image.NEAREST)
            image = image.resize((image.size[0] * 2, image.size[1] * 2), Image.NEAREST)
        
        return image

class AdvancedAugmentation:
    """Advanced augmentation for training robustness"""
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            # Color jitter
            if random.random() < 0.8:
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            # Brightness adjustment
            if random.random() < 0.8:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            # Contrast adjustment
            if random.random() < 0.8:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            # Sharpness adjustment
            if random.random() < 0.5:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(random.uniform(0.5, 1.5))
            
            # Random hue shift
            if random.random() < 0.3:
                image_np = np.array(image)
                hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] += random.uniform(-10, 10)  # Hue shift
                hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
                image_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                image = Image.fromarray(image_np)
        
        return image

class GeometricAugmentation:
    """Geometric augmentations"""
    def __init__(self, probability=0.3):
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            # Random rotation
            if random.random() < 0.5:
                angle = random.uniform(-5, 5)
                image = TF.rotate(image, angle, fill=0)
            
            # Random horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
            
            # Random vertical flip (less common)
            if random.random() < 0.1:
                image = TF.vflip(image)
            
            # Random perspective transform
            if random.random() < 0.2:
                width, height = image.size
                # Define random perspective transformation
                startpoints = [[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]
                endpoints = []
                for point in startpoints:
                    x, y = point
                    x += random.uniform(-width*0.05, width*0.05)
                    y += random.uniform(-height*0.05, height*0.05)
                    endpoints.append([x, y])
                
                try:
                    image = TF.perspective(image, startpoints, endpoints, fill=0)
                except:
                    pass  # Skip if transformation fails
        
        return image

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, scale_factor=4, augment=True, degradation_mode='bicubic'):
        self.hr_height, self.hr_width = hr_shape
        self.scale_factor = scale_factor
        self.lr_height = self.hr_height // scale_factor
        self.lr_width = self.hr_width // scale_factor
        self.augment = augment
        self.degradation_mode = degradation_mode
        
        # Data augmentation transforms
        self.geometric_aug = GeometricAugmentation() if augment else None
        self.color_aug = AdvancedAugmentation() if augment else None
        self.degradation_aug = RandomDegradation(scale_factor) if augment else None
        
        # Enhanced LR generation methods
        self.lr_methods = {
            'bicubic': self._bicubic_downsample,
            'bilinear': self._bilinear_downsample,
            'nearest': self._nearest_downsample,
            'area': self._area_downsample,
            'lanczos': self._lanczos_downsample,
            'matlab': self._matlab_imresize,
        }
        
        self.files = sorted(glob.glob(root + "/*.*"))
        if not self.files:
            raise ValueError(f"No images found in {root}")
        
        print(f"Found {len(self.files)} images in dataset")
    
    def _bicubic_downsample(self, image):
        return image.resize((self.lr_width, self.lr_height), Image.BICUBIC)
    
    def _bilinear_downsample(self, image):
        return image.resize((self.lr_width, self.lr_height), Image.BILINEAR)
    
    def _nearest_downsample(self, image):
        return image.resize((self.lr_width, self.lr_height), Image.NEAREST)
    
    def _area_downsample(self, image):
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Resize using area interpolation
        resized = cv2.resize(img_cv, (self.lr_width, self.lr_height), interpolation=cv2.INTER_AREA)
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    
    def _lanczos_downsample(self, image):
        return image.resize((self.lr_width, self.lr_height), Image.LANCZOS)
    
    def _matlab_imresize(self, image):
        """Simulate MATLAB's imresize function"""
        img_np = np.array(image).astype(np.float64)
        
        # Convert to double precision and normalize to [0, 1]
        img_np = img_np / 255.0
        
        # Use OpenCV's resize with INTER_CUBIC (similar to MATLAB's bicubic)
        resized = cv2.resize(img_np, (self.lr_width, self.lr_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert back to [0, 255] and uint8
        resized = np.clip(resized * 255.0, 0, 255).astype(np.uint8)
        
        return Image.fromarray(resized)
    
    def _crop_image(self, image, target_size):
        """Random crop or center crop to target size"""
        w, h = image.size
        target_w, target_h = target_size
        
        if w < target_w or h < target_h:
            # Resize if image is smaller than target
            image = TF.resize(image, (max(target_h, h), max(target_w, w)))
            w, h = image.size
        
        if self.augment and random.random() < 0.8:
            # Random crop
            top = random.randint(0, h - target_h)
            left = random.randint(0, w - target_w)
        else:
            # Center crop
            top = (h - target_h) // 2
            left = (w - target_w) // 2
        
        return TF.crop(image, top, left, target_h, target_w)
    
    def __getitem__(self, index):
        try:
            # Load image
            img_path = self.files[index % len(self.files)]
            img = Image.open(img_path).convert('RGB')
            
            # Ensure minimum size
            min_size = max(self.hr_height, self.hr_width)
            if min(img.size) < min_size:
                scale = min_size / min(img.size)
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                img = img.resize(new_size, Image.BICUBIC)
            
            # Crop to HR size
            img_hr = self._crop_image(img, (self.hr_width, self.hr_height))
            
            # Apply geometric augmentations
            if self.geometric_aug:
                img_hr = self.geometric_aug(img_hr)
            
            # Create LR version
            img_lr_pil = self.lr_methods[self.degradation_mode](img_hr)
            
            # Apply degradation augmentations to LR
            if self.degradation_aug:
                img_lr_pil = self.degradation_aug(img_lr_pil)
            
            # Apply color augmentations to both HR and LR
            if self.color_aug:
                if random.random() < 0.5:  # Apply to both with same random seed
                    random_state = random.getstate()
                    img_hr = self.color_aug(img_hr)
                    random.setstate(random_state)
                    img_lr_pil = self.color_aug(img_lr_pil)
            
            # Convert to tensors and normalize
            img_hr_tensor = TF.to_tensor(img_hr)
            img_hr_tensor = TF.normalize(img_hr_tensor, mean, std)
            
            img_lr_tensor = TF.to_tensor(img_lr_pil)
            img_lr_tensor = TF.normalize(img_lr_tensor, mean, std)
            
            return {"lr": img_lr_tensor, "hr": img_hr_tensor, "lr_path": img_path}
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random other image
            return self.__getitem__(random.randint(0, len(self.files) - 1))
    
    def __len__(self):
        return len(self.files)

class ValidationDataset(Dataset):
    """Validation dataset without augmentation"""
    def __init__(self, root, hr_shape, scale_factor=4):
        self.hr_height, self.hr_width = hr_shape
        self.scale_factor = scale_factor
        self.lr_height = self.hr_height // scale_factor
        self.lr_width = self.hr_width // scale_factor
        
        # Transforms for validation (no augmentation)
        self.lr_transform = transforms.Compose([
            transforms.Resize((self.lr_height, self.lr_width), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        self.hr_transform = transforms.Compose([
            transforms.Resize((self.hr_height, self.hr_width), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        self.files = sorted(glob.glob(root + "/*.*"))
        if not self.files:
            raise ValueError(f"No images found in {root}")
        
        print(f"Found {len(self.files)} validation images")
    
    def __getitem__(self, index):
        try:
            img_path = self.files[index % len(self.files)]
            img = Image.open(img_path).convert('RGB')
            
            # Center crop to maintain aspect ratio
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
            
            img_lr = self.lr_transform(img)
            img_hr = self.hr_transform(img)
            
            return {"lr": img_lr, "hr": img_hr, "lr_path": img_path}
            
        except Exception as e:
            print(f"Error loading validation image {img_path}: {e}")
            # Return a random other image
            return self.__getitem__(random.randint(0, len(self.files) - 1))
    
    def __len__(self):
        return len(self.files)

class MultiScaleDataset(Dataset):
    """Dataset that provides multiple scale factors for progressive training"""
    def __init__(self, root, hr_shape, scale_factors=[2, 3, 4], augment=True, degradation_mode='bicubic'):
        self.hr_height, self.hr_width = hr_shape
        self.scale_factors = scale_factors
        self.augment = augment
        self.degradation_mode = degradation_mode
        
        self.datasets = {}
        for scale in scale_factors:
            self.datasets[scale] = ImageDataset(
                root, hr_shape, 
                scale_factor=scale, 
                augment=augment, 
                degradation_mode=degradation_mode
            )
        
        self.files = self.datasets[scale_factors[0]].files
    
    def __getitem__(self, index):
        scale = random.choice(self.scale_factors)
        item = self.datasets[scale][index]
        item['scale_factor'] = scale
        return item
    
    def __len__(self):
        return len(self.files)

class PairedDataset(Dataset):
    """Dataset for paired HR-LR images (when you have pre-existing LR images)"""
    def __init__(self, hr_root, lr_root, hr_shape, augment=True):
        self.hr_height, self.hr_width = hr_shape
        self.augment = augment
        
        # Data augmentation transforms
        self.geometric_aug = GeometricAugmentation() if augment else None
        self.color_aug = AdvancedAugmentation() if augment else None
        
        self.hr_files = sorted(glob.glob(hr_root + "/*.*"))
        self.lr_files = sorted(glob.glob(lr_root + "/*.*"))
        
        if len(self.hr_files) != len(self.lr_files):
            print(f"Warning: HR images ({len(self.hr_files)}) != LR images ({len(self.lr_files)})")
            min_len = min(len(self.hr_files), len(self.lr_files))
            self.hr_files = self.hr_files[:min_len]
            self.lr_files = self.lr_files[:min_len]
        
        if not self.hr_files:
            raise ValueError(f"No paired images found")
        
        print(f"Found {len(self.hr_files)} paired images")
    
    def __getitem__(self, index):
        try:
            hr_path = self.hr_files[index % len(self.hr_files)]
            lr_path = self.lr_files[index % len(self.lr_files)]
            
            hr_img = Image.open(hr_path).convert('RGB')
            lr_img = Image.open(lr_path).convert('RGB')
            
            # Resize to target dimensions
            hr_img = hr_img.resize((self.hr_width, self.hr_height), Image.BICUBIC)
            lr_img = lr_img.resize((self.hr_width // 4, self.hr_height // 4), Image.BICUBIC)
            
            # Apply same augmentations to both images
            if self.geometric_aug and random.random() < 0.5:
                # Apply same geometric transform to both
                if random.random() < 0.5:
                    hr_img = TF.hflip(hr_img)
                    lr_img = TF.hflip(lr_img)
                
                if random.random() < 0.1:
                    hr_img = TF.vflip(hr_img)
                    lr_img = TF.vflip(lr_img)
            
            # Apply same color augmentations
            if self.color_aug and random.random() < 0.5:
                random_state = random.getstate()
                hr_img = self.color_aug(hr_img)
                random.setstate(random_state)
                lr_img = self.color_aug(lr_img)
            
            # Convert to tensors
            hr_tensor = TF.to_tensor(hr_img)
            hr_tensor = TF.normalize(hr_tensor, mean, std)
            
            lr_tensor = TF.to_tensor(lr_img)
            lr_tensor = TF.normalize(lr_tensor, mean, std)
            
            return {"lr": lr_tensor, "hr": hr_tensor, "lr_path": lr_path, "hr_path": hr_path}
            
        except Exception as e:
            print(f"Error loading paired images {hr_path}, {lr_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.files) - 1))
    
    def __len__(self):
        return len(self.hr_files)

class TestDataset(Dataset):
    """Test dataset for inference - only loads LR images"""
    def __init__(self, root, target_size=None):
        self.target_size = target_size
        self.files = sorted(glob.glob(root + "/*.*"))
        
        if not self.files:
            raise ValueError(f"No test images found in {root}")
        
        print(f"Found {len(self.files)} test images")
    
    def __getitem__(self, index):
        try:
            img_path = self.files[index]
            img = Image.open(img_path).convert('RGB')
            
            # Resize if target size is specified
            if self.target_size:
                img = img.resize(self.target_size, Image.BICUBIC)
            
            # Convert to tensor and normalize
            img_tensor = TF.to_tensor(img)
            img_tensor = TF.normalize(img_tensor, mean, std)
            
            return {"lr": img_tensor, "lr_path": img_path}
            
        except Exception as e:
            print(f"Error loading test image {img_path}: {e}")
            # Return a black image as fallback
            if self.target_size:
                img = Image.new('RGB', self.target_size, (0, 0, 0))
            else:
                img = Image.new('RGB', (256, 256), (0, 0, 0))
            
            img_tensor = TF.to_tensor(img)
            img_tensor = TF.normalize(img_tensor, mean, std)
            
            return {"lr": img_tensor, "lr_path": img_path}
    
    def __len__(self):
        return len(self.files)

# Utility functions for dataset creation
def create_train_val_split(root_dir, train_ratio=0.8, seed=42):
    """Split dataset into training and validation sets"""
    random.seed(seed)
    all_files = sorted(glob.glob(root_dir + "/*.*"))
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    return train_files, val_files

def get_dataset_stats(dataset):
    """Get basic statistics about the dataset"""
    print(f"Dataset size: {len(dataset)}")
    
    # Sample a few items to get tensor shapes
    sample = dataset[0]
    lr_shape = sample['lr'].shape
    hr_shape = sample['hr'].shape
    
    print(f"LR tensor shape: {lr_shape}")
    print(f"HR tensor shape: {hr_shape}")
    print(f"Scale factor: {hr_shape[-1] // lr_shape[-1]}")

# Example usage and factory functions
def create_training_dataset(root, hr_shape=(256, 256), scale_factor=4, 
                          augment=True, degradation_mode='bicubic'):
    """Factory function to create training dataset"""
    return ImageDataset(
        root=root,
        hr_shape=hr_shape,
        scale_factor=scale_factor,
        augment=augment,
        degradation_mode=degradation_mode
    )

def create_validation_dataset(root, hr_shape=(256, 256), scale_factor=4):
    """Factory function to create validation dataset"""
    return ValidationDataset(
        root=root,
        hr_shape=hr_shape,
        scale_factor=scale_factor
    )

def create_multiscale_dataset(root, hr_shape=(256, 256), 
                            scale_factors=[2, 3, 4], augment=True):
    """Factory function to create multi-scale dataset"""
    return MultiScaleDataset(
        root=root,
        hr_shape=hr_shape,
        scale_factors=scale_factors,
        augment=augment
    )

if __name__ == "__main__":
    # Example usage
    print("Testing Enhanced Dataset Classes...")
    
    # Create sample dataset (replace with actual path)
    try:
        dataset = create_training_dataset(
            root="./train_images",
            hr_shape=(128, 128),
            scale_factor=4,
            augment=True
        )
        
        print(f"Created dataset with {len(dataset)} images")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"Sample LR shape: {sample['lr'].shape}")
        print(f"Sample HR shape: {sample['hr'].shape}")
        
    except ValueError as e:
        print(f"Dataset creation failed: {e}")
        print("Make sure to provide a valid image directory path")