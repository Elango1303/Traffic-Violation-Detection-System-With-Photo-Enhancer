import os
import glob
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
import time
from PIL import Image
import json
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.color import rgb2gray
import importlib
import importlib.util
lpips = None
try:
    if importlib.util.find_spec('lpips') is not None:
        lpips = importlib.import_module('lpips')
except Exception:
    lpips = None
import torch.nn.functional as F

# Import your model architectures
arch = None
try:
    if importlib.util.find_spec('RRDBNet_arch') is not None:
        arch = importlib.import_module('RRDBNet_arch')
except Exception:
    arch = None
from enhanced_models import EnhancedGeneratorRRDB
if arch is None:
    print("Warning: RRDBNet_arch not found. Falling back to EnhancedGeneratorRRDB only.")
from config_and_utils import BASE_DIR

class ImageQualityMetrics:
    """Comprehensive image quality assessment metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize LPIPS for perceptual similarity
        if lpips is not None:
            try:
                self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            except Exception:
                print("Warning: LPIPS not available. Perceptual metrics will be skipped.")
                self.lpips_fn = None
        else:
            print("Warning: LPIPS package not installed. Perceptual metrics will be skipped.")
            self.lpips_fn = None
    
    def calculate_psnr(self, img1, img2, data_range=255):
        """Calculate PSNR between two images"""
        return psnr(img1, img2, data_range=data_range)
    
    def calculate_ssim(self, img1, img2, data_range=255):
        """Calculate SSIM between two images"""
        if len(img1.shape) == 3:
            return ssim(img1, img2, multichannel=True, channel_axis=2, data_range=data_range)
        else:
            return ssim(img1, img2, data_range=data_range)
    
    def calculate_lpips(self, img1, img2):
        """Calculate LPIPS perceptual distance"""
        if self.lpips_fn is None:
            return 0.0
        
        # Convert to tensors and normalize to [-1, 1]
        img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).float() / 127.5 - 1.0
        img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).float() / 127.5 - 1.0
        
        img1_tensor = img1_tensor.unsqueeze(0).to(self.device)
        img2_tensor = img2_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            distance = self.lpips_fn(img1_tensor, img2_tensor)
        
        return distance.item()
    
    def calculate_mse(self, img1, img2):
        """Calculate Mean Squared Error"""
        return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    
    def calculate_mae(self, img1, img2):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))
    
    def calculate_all_metrics(self, sr_img, hr_img):
        """Calculate all available metrics"""
        metrics = {}
        
        # Ensure images are in the same format
        if sr_img.shape != hr_img.shape:
            min_h, min_w = min(sr_img.shape[0], hr_img.shape[0]), min(sr_img.shape[1], hr_img.shape[1])
            sr_img = sr_img[:min_h, :min_w]
            hr_img = hr_img[:min_h, :min_w]
        
        # Basic metrics
        metrics['psnr'] = self.calculate_psnr(sr_img, hr_img)
        metrics['ssim'] = self.calculate_ssim(sr_img, hr_img)
        metrics['mse'] = self.calculate_mse(sr_img, hr_img)
        metrics['mae'] = self.calculate_mae(sr_img, hr_img)
        
        # Perceptual metrics
        if len(sr_img.shape) == 3:  # Color image
            metrics['lpips'] = self.calculate_lpips(sr_img, hr_img)
        else:
            metrics['lpips'] = 0.0
        
        return metrics

class EnhancedImageTester:
    """Enhanced image testing framework for super-resolution models"""
    
    def __init__(self, device='cuda', upscale=4):
        self.device = device
        self.upscale = upscale
        self.model = None
        self.metrics_calculator = ImageQualityMetrics(device)
        
    def load_model(self, model_path, model_type='RRDB'):
        """Load the super-resolution model"""
        print(f"Loading model from: {model_path}")
        
        if model_type == 'RRDB':
            try:
                # Try loading enhanced RRDB model with correct parameters
                self.model = EnhancedGeneratorRRDB(
                    channels=3, 
                    filters=64, 
                    num_res_blocks=6,  # Match the training configuration
                    num_upsample=2,    # For 4x upscaling
                    use_attention=True
                )
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print("Enhanced RRDB model loaded successfully!")
            except Exception as e:
                print(f"Error loading enhanced model: {e}")
                if arch is not None:
                    try:
                        # Fallback to standard RRDB if available
                        self.model = arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
                        state_dict = torch.load(model_path, map_location=self.device)
                        self.model.load_state_dict(state_dict, strict=False)
                        print("Standard RRDB model loaded successfully!")
                    except Exception as e2:
                        print(f"Error loading standard model: {e2}")
                        return False
                else:
                    return False
        
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        return True
    
    def preprocess_image(self, img_path):
        """Preprocess input image for the model"""
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, img
    
    def postprocess_output(self, output_tensor):
        """Convert model output back to image format"""
        output = output_tensor.squeeze(0).cpu().detach().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        return output
    
    def enhance_image(self, img_path):
        """Enhance a single image using the loaded model"""
        if self.model is None:
            print("Error: No model loaded!")
            return None, None
        
        # Preprocess
        img_tensor, original_img = self.preprocess_image(img_path)
        if img_tensor is None:
            return None, None
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            output = self.model(img_tensor)
        inference_time = time.time() - start_time
        
        # Postprocess
        enhanced_img = self.postprocess_output(output)
        
        return enhanced_img, inference_time
    
    def test_single_image(self, lr_path, hr_path=None, save_path=None):
        """Test enhancement on a single image"""
        print(f"Processing: {lr_path}")
        
        # Enhance image
        enhanced_img, inference_time = self.enhance_image(lr_path)
        if enhanced_img is None:
            return None
        
        results = {
            'image_path': lr_path,
            'inference_time': inference_time,
            'output_shape': enhanced_img.shape
        }
        
        # Calculate metrics if ground truth is available
        if hr_path and os.path.exists(hr_path):
            hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            
            metrics = self.metrics_calculator.calculate_all_metrics(enhanced_img, hr_img)
            results.update(metrics)
            print(f"PSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.4f}")
        
        # Save enhanced image
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, enhanced_img_bgr)
            print(f"Saved: {save_path}")
        
        return results
    
    def test_dataset(self, lr_dir, hr_dir=None, output_dir=None, image_extensions=['*.jpg', '*.png', '*.jpeg']):
        """Test enhancement on a dataset"""
        print(f"Testing dataset in: {lr_dir}")
        
        # Get all image files
        lr_images = []
        for ext in image_extensions:
            lr_images.extend(glob.glob(os.path.join(lr_dir, ext)))
            lr_images.extend(glob.glob(os.path.join(lr_dir, ext.upper())))
        
        if not lr_images:
            print("No images found in the specified directory!")
            return None
        
        print(f"Found {len(lr_images)} images to process")
        
        all_results = []
        total_time = 0
        
        for lr_path in tqdm(lr_images, desc="Processing images"):
            # Determine corresponding HR path
            hr_path = None
            if hr_dir:
                basename = os.path.basename(lr_path)
                hr_path = os.path.join(hr_dir, basename)
            
            # Determine output path
            save_path = None
            if output_dir:
                basename = os.path.basename(lr_path)
                name, ext = os.path.splitext(basename)
                save_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
            
            # Process image
            result = self.test_single_image(lr_path, hr_path, save_path)
            if result:
                all_results.append(result)
                total_time += result['inference_time']
        
        # Calculate summary statistics
        if all_results:
            summary = self.calculate_summary_stats(all_results, total_time)
            return all_results, summary
        
        return None, None
    
    def calculate_summary_stats(self, results, total_time):
        """Calculate summary statistics from results"""
        summary = {
            'total_images': len(results),
            'total_time': total_time,
            'average_time': total_time / len(results),
            'fps': len(results) / total_time
        }
        
        # Calculate metric averages if available
        metric_keys = ['psnr', 'ssim', 'mse', 'mae', 'lpips']
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                summary[f'avg_{key}'] = np.mean(values)
                summary[f'std_{key}'] = np.std(values)
        
        return summary
    
    def create_comparison_plot(self, lr_path, hr_path, enhanced_img, metrics=None):
        """Create a comparison plot of LR, HR, and enhanced images"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Load and display LR image
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(lr_img)
        axes[0].set_title('Low Resolution')
        axes[0].axis('off')
        
        # Display enhanced image
        axes[1].imshow(enhanced_img)
        axes[1].set_title('Enhanced (Ours)')
        axes[1].axis('off')
        
        # Load and display HR image
        if hr_path and os.path.exists(hr_path):
            hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            axes[2].imshow(hr_img)
            axes[2].set_title('Ground Truth')
        else:
            axes[2].imshow(enhanced_img)
            axes[2].set_title('Enhanced (No GT)')
        axes[2].axis('off')
        
        # Add metrics text if available
        if metrics:
            metric_text = f"PSNR: {metrics.get('psnr', 0):.2f}dB\n"
            metric_text += f"SSIM: {metrics.get('ssim', 0):.4f}\n"
            metric_text += f"LPIPS: {metrics.get('lpips', 0):.4f}"
            plt.figtext(0.02, 0.02, metric_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        return fig
    
    def save_results_report(self, results, summary, output_path):
        """Save detailed results report"""
        report = {
            'summary': summary,
            'individual_results': results,
            'model_info': {
                'device': self.device,
                'upscale_factor': self.upscale
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Results report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Image Super-Resolution Testing Framework')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--lr_dir', type=str, required=True, help='Directory containing low-resolution images')
    parser.add_argument('--hr_dir', type=str, default=None, help='Directory containing high-resolution ground truth images')
    parser.add_argument('--output_dir', type=str, default=os.path.join(BASE_DIR, 'enhanced_outputs'), help='Directory to save enhanced images')
    parser.add_argument('--results_dir', type=str, default=os.path.join(BASE_DIR, 'results'), help='Directory to save results and reports')
    parser.add_argument('--model_type', type=str, default='RRDB', choices=['RRDB'], help='Type of model architecture')
    parser.add_argument('--upscale', type=int, default=4, help='Super-resolution upscaling factor')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--create_plots', action='store_true', help='Create comparison plots')
    parser.add_argument('--single_image', type=str, default=None, help='Test on a single image instead of directory')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize tester
    tester = EnhancedImageTester(device=args.device, upscale=args.upscale)
    
    # Load model
    if not tester.load_model(args.model_path, args.model_type):
        print("Failed to load model. Exiting...")
        return
    
    if args.single_image:
        # Test single image
        hr_path = None
        if args.hr_dir:
            basename = os.path.basename(args.single_image)
            hr_path = os.path.join(args.hr_dir, basename)
        
        output_path = os.path.join(args.output_dir, f"enhanced_{os.path.basename(args.single_image)}")
        result = tester.test_single_image(args.single_image, hr_path, output_path)
        
        if result and args.create_plots:
            enhanced_img, _ = tester.enhance_image(args.single_image)
            if enhanced_img is not None:
                fig = tester.create_comparison_plot(args.single_image, hr_path, enhanced_img, result)
                plot_path = os.path.join(args.results_dir, "comparison_plot.png")
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Comparison plot saved to: {plot_path}")
    
    else:
        # Test dataset
        results, summary = tester.test_dataset(args.lr_dir, args.hr_dir, args.output_dir)
        
        if results and summary:
            # Print summary
            print("\n" + "="*50)
            print("TESTING SUMMARY")
            print("="*50)
            print(f"Total images processed: {summary['total_images']}")
            print(f"Total processing time: {summary['total_time']:.2f}s")
            print(f"Average time per image: {summary['average_time']:.4f}s")
            print(f"Processing speed: {summary['fps']:.2f} FPS")
            
            if 'avg_psnr' in summary:
                print(f"Average PSNR: {summary['avg_psnr']:.2f}dB (±{summary['std_psnr']:.2f})")
                print(f"Average SSIM: {summary['avg_ssim']:.4f} (±{summary['std_ssim']:.4f})")
                print(f"Average LPIPS: {summary['avg_lpips']:.4f} (±{summary['std_lpips']:.4f})")
            
            # Save detailed report
            report_path = os.path.join(args.results_dir, "test_results.json")
            tester.save_results_report(results, summary, report_path)
            
            # Create summary plot if metrics are available
            if 'avg_psnr' in summary and args.create_plots:
                plt.figure(figsize=(12, 8))
                
                # PSNR distribution
                psnr_values = [r['psnr'] for r in results if 'psnr' in r]
                if psnr_values:
                    plt.subplot(2, 2, 1)
                    plt.hist(psnr_values, bins=20, alpha=0.7, edgecolor='black')
                    plt.xlabel('PSNR (dB)')
                    plt.ylabel('Frequency')
                    plt.title('PSNR Distribution')
                    plt.axvline(summary['avg_psnr'], color='red', linestyle='--', label=f'Mean: {summary["avg_psnr"]:.2f}')
                    plt.legend()
                
                # SSIM distribution
                ssim_values = [r['ssim'] for r in results if 'ssim' in r]
                if ssim_values:
                    plt.subplot(2, 2, 2)
                    plt.hist(ssim_values, bins=20, alpha=0.7, edgecolor='black')
                    plt.xlabel('SSIM')
                    plt.ylabel('Frequency')
                    plt.title('SSIM Distribution')
                    plt.axvline(summary['avg_ssim'], color='red', linestyle='--', label=f'Mean: {summary["avg_ssim"]:.4f}')
                    plt.legend()
                
                # Processing time distribution
                time_values = [r['inference_time'] for r in results]
                plt.subplot(2, 2, 3)
                plt.hist(time_values, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Processing Time (s)')
                plt.ylabel('Frequency')
                plt.title('Processing Time Distribution')
                plt.axvline(summary['average_time'], color='red', linestyle='--', label=f'Mean: {summary["average_time"]:.4f}s')
                plt.legend()
                
                # PSNR vs SSIM scatter
                if psnr_values and ssim_values:
                    plt.subplot(2, 2, 4)
                    plt.scatter(psnr_values, ssim_values, alpha=0.6)
                    plt.xlabel('PSNR (dB)')
                    plt.ylabel('SSIM')
                    plt.title('PSNR vs SSIM')
                    
                    # Add correlation coefficient
                    correlation = np.corrcoef(psnr_values, ssim_values)[0, 1]
                    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                           transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                
                plt.tight_layout()
                plot_path = os.path.join(args.results_dir, "summary_plots.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Summary plots saved to: {plot_path}")
        
        else:
            print("No results to display.")

if __name__ == "__main__":
    main()