import argparse
import os
import numpy as np
import math
import itertools
import sys
import time
from PIL import Image
import wandb
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import cv2
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    from enhanced_datasets import ImageDataset, denormalize
    from enhanced_models import GeneratorRRDB, Discriminator, FeatureExtractor
    from config_and_utils import BASE_DIR

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=500, help="batch interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    parser.add_argument("--lambda_perceptual", type=float, default=1.0, help="perceptual loss weight")
    parser.add_argument("--pretrained_weights", type=str, default="", help="path to pretrained weights")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision training")
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="gradient clipping value")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")

    opt = parser.parse_args()
    print(opt)

    # Initialize wandb if enabled
    if opt.use_wandb:
        wandb.init(project="esrgan-enhanced", config=opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize models
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    # EMA for generator
    class EMA:
        def __init__(self, model, decay=0.999):
            self.model = model
            self.decay = decay
            self.shadow = {}
            self.backup = {}
        def register(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()
        def update(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()
        def apply_shadow(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    self.backup[name] = param.data
                    param.data = self.shadow[name]
        def restore(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.backup
                    param.data = self.backup[name]
            self.backup = {}

    ema = EMA(generator, decay=opt.ema_decay)
    ema.register()

    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    if opt.pretrained_weights:
        weights_path = opt.pretrained_weights if os.path.isabs(opt.pretrained_weights) else os.path.join(BASE_DIR, opt.pretrained_weights)
        if os.path.exists(weights_path):
            generator.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded pretrained weights from {weights_path}")
        else:
            print(f"Warning: Pretrained weights file not found at {weights_path}")
            generator.apply(weights_init_normal)
    else:
        generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=opt.n_epochs, eta_min=1e-7)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=opt.n_epochs, eta_min=1e-7)

    dataset_path = os.path.join(BASE_DIR, "data", opt.dataset_name)
    print(f"Looking for dataset at: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        print("Please make sure your dataset is placed in the correct directory.")
        sys.exit(1)

    try:
        dataset = ImageDataset(dataset_path, hr_shape=hr_shape)
        print(f"Successfully loaded dataset with {len(dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        persistent_workers=True if opt.n_cpu > 0 else False,
    )

    os.makedirs(os.path.join(BASE_DIR, "images", "training"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "saved_models"), exist_ok=True)
    print("Created output directories")

    scaler = torch.cuda.amp.GradScaler() if opt.mixed_precision else None

    def calculate_metrics(hr_imgs, sr_imgs):
        psnr_vals = []
        ssim_vals = []
        hr_imgs = hr_imgs.cpu().numpy()
        sr_imgs = sr_imgs.cpu().numpy()
        for i in range(hr_imgs.shape[0]):
            hr_img = np.transpose(hr_imgs[i], (1, 2, 0))
            sr_img = np.transpose(sr_imgs[i], (1, 2, 0))
            hr_img = np.clip(hr_img * 255, 0, 255).astype(np.uint8)
            sr_img = np.clip(sr_img * 255, 0, 255).astype(np.uint8)
            psnr_val = psnr(hr_img, sr_img)
            psnr_vals.append(psnr_val)
            ssim_val = ssim(hr_img, sr_img, channel_axis=2, data_range=255)
            ssim_vals.append(ssim_val)
        return np.mean(psnr_vals), np.mean(ssim_vals)

    print("Starting training...")
    prev_time = time.time()
    best_psnr = 0
    patience = 0
    max_patience = 20

    for epoch in range(opt.epoch, opt.n_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        num_batches = 0

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")

        for i, imgs in progress_bar:
            batches_done = epoch * len(dataloader) + i
            imgs_lr = Variable(imgs["lr"].type(torch.FloatTensor)).to(device, non_blocking=True)
            imgs_hr = Variable(imgs["hr"].type(torch.FloatTensor)).to(device, non_blocking=True)
            valid = Variable(torch.ones((imgs_lr.size(0), *discriminator.output_shape)),
                            requires_grad=False).to(device, non_blocking=True)
            fake = Variable(torch.zeros((imgs_lr.size(0), *discriminator.output_shape)),
                           requires_grad=False).to(device, non_blocking=True)
            optimizer_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=opt.mixed_precision):
                gen_hr = generator(imgs_lr)
                loss_pixel = criterion_pixel(gen_hr, imgs_hr)
                if batches_done < opt.warmup_batches:
                    loss_G = loss_pixel
                else:
                    pred_real = discriminator(imgs_hr).detach()
                    pred_fake = discriminator(gen_hr)
                    loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
                    gen_features = feature_extractor(gen_hr)
                    real_features = feature_extractor(imgs_hr).detach()
                    loss_content = criterion_content(gen_features, real_features)
                    loss_G = (loss_content * opt.lambda_perceptual +
                             loss_GAN * opt.lambda_adv +
                             loss_pixel * opt.lambda_pixel)
            if opt.mixed_precision:
                scaler.scale(loss_G).backward()
                if opt.gradient_clipping > 0:
                    scaler.unscale_(optimizer_G)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), opt.gradient_clipping)
                scaler.step(optimizer_G)
                scaler.update()
            else:
                loss_G.backward()
                if opt.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), opt.gradient_clipping)
                optimizer_G.step()
            ema.update()
            if batches_done >= opt.warmup_batches:
                optimizer_D.zero_grad()
                with torch.cuda.amp.autocast(enabled=opt.mixed_precision):
                    pred_real = discriminator(imgs_hr)
                    pred_fake = discriminator(gen_hr.detach())
                    loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
                    loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
                    loss_D = (loss_real + loss_fake) / 2
                if opt.mixed_precision:
                    scaler.scale(loss_D).backward()
                    if opt.gradient_clipping > 0:
                        scaler.unscale_(optimizer_D)
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), opt.gradient_clipping)
                    scaler.step(optimizer_D)
                    scaler.update()
                else:
                    loss_D.backward()
                    if opt.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), opt.gradient_clipping)
                    optimizer_D.step()
            else:
                loss_D = torch.tensor(0.0)
            with torch.no_grad():
                gen_hr_denorm = denormalize(gen_hr.clone())
                imgs_hr_denorm = denormalize(imgs_hr.clone())
                batch_psnr, batch_ssim = calculate_metrics(imgs_hr_denorm, gen_hr_denorm)
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            num_batches += 1
            progress_bar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}',
                'PSNR': f'{batch_psnr:.2f}',
                'SSIM': f'{batch_ssim:.3f}'
            })
            if opt.use_wandb:
                wandb.log({
                    'G_loss': loss_G.item(),
                    'D_loss': loss_D.item(),
                    'PSNR': batch_psnr,
                    'SSIM': batch_ssim,
                    'lr_G': scheduler_G.get_last_lr()[0],
                    'lr_D': scheduler_D.get_last_lr()[0]
                })
            if batches_done % opt.sample_interval == 0:
                ema.apply_shadow()
                with torch.no_grad():
                    gen_hr_ema = generator(imgs_lr)
                ema.restore()
                imgs_lr_denorm = denormalize(imgs_lr.clone())
                # Upsample LR images to match HR dimensions for visualization
                imgs_lr_upsampled = F.interpolate(imgs_lr_denorm, size=(opt.hr_height, opt.hr_width), mode='bicubic', align_corners=False)
                gen_hr_ema_denorm = denormalize(gen_hr_ema.clone())
                imgs_hr_denorm = denormalize(imgs_hr.clone())
                img_sample = torch.cat((imgs_lr_upsampled, gen_hr_ema_denorm, imgs_hr_denorm), -2)
                save_image(img_sample, os.path.join(BASE_DIR, "images", "training", f"{batches_done}.png"), nrow=1, normalize=True)
            if opt.checkpoint_interval != -1 and batches_done % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), os.path.join(BASE_DIR, "saved_models", f"generator_{batches_done}.pth"))
                torch.save(discriminator.state_dict(), os.path.join(BASE_DIR, "saved_models", f"discriminator_{batches_done}.pth"))
        scheduler_G.step()
        scheduler_D.step()
        avg_g_loss = epoch_g_loss / num_batches if num_batches > 0 else 0
        avg_d_loss = epoch_d_loss / num_batches if num_batches > 0 else 0
        avg_psnr = epoch_psnr / num_batches if num_batches > 0 else 0
        avg_ssim = epoch_ssim / num_batches if num_batches > 0 else 0
        print(f"\nEpoch {epoch} - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}, "
              f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.3f}")
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            patience = 0
            ema.apply_shadow()
            torch.save(generator.state_dict(), os.path.join(BASE_DIR, "saved_models", "best_generator.pth"))
            ema.restore()
            print(f"New best PSNR: {best_psnr:.2f}dB - Model saved")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict(),
            'ema_shadow': ema.shadow,
            'best_psnr': best_psnr,
        }, os.path.join(BASE_DIR, "saved_models", "checkpoint_latest.pth"))
    print("Training completed!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # For Windows compatibility
    main()