import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import time
import csv
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
from model.wdnet import WDNet
import torchvision.models as models
from dataLoader import DerainDataset

# DeRaindrop
Mean = [0.4970002770423889, 0.5053070783615112, 0.4676517844200134]
Std = [0.24092243611812592, 0.23609396815299988, 0.25256040692329407]

# 保持原有的工具函数不变
def denormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def get_input_shape(data_loader):
    data_iter = iter(data_loader)
    rain_images, _ = next(data_iter)
    return rain_images.shape

def calculate_metrics_rgb(output, target):
    output = output.cpu().numpy().transpose(1, 2, 0)
    target = target.cpu().numpy().transpose(1, 2, 0)
    
    output = (output * 255.0).clip(0, 255).astype(np.uint8)
    target = (target * 255.0).clip(0, 255).astype(np.uint8)
    
    psnr_value = psnr(target, output, data_range=255)
    ssim_value = ssim(target, output, data_range=255, channel_axis=-1)
    
    return psnr_value, ssim_value

def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    y  =  65.738 * r /256 + 129.057 * g / 256 + 25.064 * b / 256 + 16
    cb = -37.945 * r / 256 - 74.494 * g / 256 + 112.439 * b / 256 + 128
    cr =  112.439 * r / 256 - 94.154 * g / 256 - 18.285 * b / 256 + 128

    ycbcr_img = np.stack([y, cb, cr], axis=-1)
    ycbcr_img = np.clip(ycbcr_img, 0, 255).astype(np.uint8)
    return ycbcr_img

def calculate_metrics(output, target):
    output = output.cpu().numpy().transpose(1, 2, 0)
    target = target.cpu().numpy().transpose(1, 2, 0)
    
    output = (output * 255.0).clip(0, 255).astype(np.uint8)
    target = (target * 255.0).clip(0, 255).astype(np.uint8)
    
    output_ycbcr = rgb_to_ycbcr(output)
    target_ycbcr = rgb_to_ycbcr(target)
    
    output_y = output_ycbcr[:, :, 0]
    target_y = target_ycbcr[:, :, 0]
    
    psnr_value = psnr(target_y, output_y, data_range=255)
    ssim_value = ssim(target_y, output_y, data_range=255)
    
    return psnr_value, ssim_value

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model='vgg16', resize=True, device='cuda'):
        super(PerceptualLoss, self).__init__()
        if vgg_model == 'vgg16':
            self.vgg = models.vgg16(pretrained=True).features[:16].eval()
        elif vgg_model == 'vgg19':
            self.vgg = models.vgg19(pretrained=True).features[:16].eval()
        else:
            raise ValueError("Unsupported VGG model. Choose 'vgg16' or 'vgg19'.")
        self.vgg.to(device)
        self.resize = resize
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        if self.resize:
            input = nn.functional.interpolate(input, size=(256, 256), mode='bilinear', align_corners=False)
            target = nn.functional.interpolate(target, size=(256, 256), mode='bilinear', align_corners=False)
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return self.criterion(input_features, target_features)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, vgg_model='vgg16', device='cuda'):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(vgg_model=vgg_model, device=device)

    def forward(self, input, target):
        l1_loss_value = self.l1_loss(input, target)
        perceptual_loss_value = self.perceptual_loss(input, target)
        return self.alpha * l1_loss_value + (1 - self.alpha) * perceptual_loss_value

def parse_args():
    parser = argparse.ArgumentParser(description='WDNet Training for Rain Drop Removal')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='datas/archive',
                        help='Path to the dataset directory')
    parser.add_argument('--image_size', type=str, default='480,720',
                        help='Image size for training (height,width)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='wdnet.pth',
                        help='Path to save the best model')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for L1 loss in combined loss')
    parser.add_argument('--vgg_model', type=str, default='vgg16', choices=['vgg16', 'vgg19'],
                        help='VGG model for perceptual loss')
    
    # Loss explosion detection parameters
    parser.add_argument('--max_loss', type=float, default=10.0,
                        help='Maximum loss threshold for explosion detection')
    parser.add_argument('--loss_patience', type=int, default=50,
                        help='Number of steps to monitor for loss explosion')
    
    # Logging parameters
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to save training log (default: auto-generated)')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')
    
    return parser.parse_args()

def train_epoch(model, train_loader, optimizer, criterion, recent_losses, max_loss_threshold):
    model.train()
    running_loss = 0.0
    
    for i, (rain_images, clean_images) in enumerate(tqdm(train_loader)):
        rain_images, clean_images = rain_images.to(device), clean_images.to(device)
        optimizer.zero_grad()
        predicted_clean_image = model(rain_images)
        loss = criterion(predicted_clean_image, clean_images)
        
        if loss.item() > max_loss_threshold:
            print(f"\nLoss explosion detected: {loss.item():.4f}")
            return None, True
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        recent_losses.append(loss.item())
        if len(recent_losses) > args.loss_patience:
            recent_losses.pop(0)
            
        if len(recent_losses) == args.loss_patience:
            if np.mean(recent_losses[-10:]) > np.mean(recent_losses[:10]) * 2:
                print("\nContinuous loss increase detected")
                return None, True
                
    return running_loss / len(train_loader), False

def test_model(model, test_loader):
    model.eval()
    total_psnr, total_ssim = 0.0, 0.0
    num_images = 0
    
    with torch.no_grad():
        for rain_images, clean_images in tqdm(test_loader, desc='Testing'):
            rain_images, clean_images = rain_images.to(device), clean_images.to(device)
            output = model(rain_images)
            denorm_output = denormalize(output[0], mean=Mean, std=Std)
            denorm_clean = denormalize(clean_images[0], mean=Mean, std=Std)
            psnr_value, ssim_value = calculate_metrics(denorm_output, denorm_clean)
            total_psnr += psnr_value
            total_ssim += ssim_value
            num_images += 1
            
    return total_psnr / num_images, total_ssim / num_images

def initialize_training(model, best_model_path, learning_rate, weight_decay):
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded model from {best_model_path}")
    else:
        print("No existing model found, training from scratch")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

if __name__ == "__main__":
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Parse image size
    height, width = map(int, args.image_size.split(','))
    
    # Create log file name if not provided
    if args.log_file is None:
        args.log_file = f'training_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Mean, std=Std),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Mean, std=Std),
    ])
    
    # Initialize model, loss function
    model = WDNet().to(device)
    criterion = CombinedLoss(alpha=args.alpha, vgg_model=args.vgg_model, device=device)
    
    # Load datasets
    train_dataset = DerainDataset(data_dir=args.data_dir, split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = DerainDataset(data_dir=args.data_dir, split='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Create CSV file for logging
    with open(args.log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Test PSNR", "Test SSIM", "Learning Rate"])
    
    # Main training loop
    best_psnr = -1
    recent_losses = []
    optimizer = initialize_training(model, args.model_path, args.lr, args.weight_decay)
    
    # Print training configuration
    print("Training Configuration:")
    print(f"Data Directory: {args.data_dir}")
    print(f"Image Size: {height}x{width}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Alpha (L1 weight): {args.alpha}")
    print(f"VGG Model: {args.vgg_model}")
    print(f"Log File: {args.log_file}")
    print(f"Model Path: {args.model_path}")
    print(f"Max Loss Threshold: {args.max_loss}")
    print(f"Loss Patience: {args.loss_patience}")
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train one epoch
        avg_train_loss, loss_exploded = train_epoch(model, train_loader, optimizer, criterion, recent_losses, args.max_loss)
        
        # If loss exploded, restart training from last best model
        if loss_exploded:
            print("Restarting training from last best model...")
            if os.path.exists(args.model_path):
                model.load_state_dict(torch.load(args.model_path))
                optimizer = initialize_training(model, args.model_path, args.lr, args.weight_decay)
                recent_losses = []
                continue
            else:
                print("No best model found to reload. Stopping training.")
                break
        
        # Test and evaluate
        avg_psnr, avg_ssim = test_model(model, test_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save training log
        with open(args.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train_loss, avg_psnr, avg_ssim, current_lr])
        
        print(f"Loss: {avg_train_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), args.model_path)
            print(f"New best PSNR: {best_psnr:.4f}, model saved to {args.model_path}")
        
        # Check if learning rate has reached minimum threshold
        if current_lr < 1e-6:
            print("Learning rate has reached minimum threshold. Stopping training.")
            break