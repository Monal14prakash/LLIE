"""
Training script optimized for Google Colab
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from pathlib import Path
import argparse

from generator import UNetGenerator
from discriminator import PatchGANDiscriminator
from data_loader import get_data_loaders

class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Create models
        self.generator = UNetGenerator().to(self.device)
        self.discriminator = PatchGANDiscriminator().to(self.device)
        
        # Loss functions
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config['lr'],
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr'],
            betas=(0.5, 0.999)
        )
        
        # Data loaders
        self.train_loader, self.val_loader = get_data_loaders(
            low_dir=config['low_dir'],
            high_dir=config['high_dir'],
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            num_workers=config['num_workers']
        )
        
        # TensorBoard
        self.writer = SummaryWriter(config['log_dir'])
        
        # Create checkpoint directory
        Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (low_imgs, high_imgs) in enumerate(pbar):
            low_imgs = low_imgs.to(self.device)
            high_imgs = high_imgs.to(self.device)
            
            # Get discriminator output shape dynamically
            with torch.no_grad():
                fake_imgs_temp = self.generator(low_imgs)
                pred_shape = self.discriminator(fake_imgs_temp).shape
            
            real_label = torch.ones(pred_shape).to(self.device)
            fake_label = torch.zeros(pred_shape).to(self.device)
            
            # Train Discriminator
            self.optimizer_D.zero_grad()
            fake_imgs = self.generator(low_imgs)
            pred_real = self.discriminator(high_imgs)
            loss_real = self.criterion_GAN(pred_real, real_label)
            pred_fake = self.discriminator(fake_imgs.detach())
            loss_fake = self.criterion_GAN(pred_fake, fake_label)
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            self.optimizer_D.step()
            
            # Train Generator
            self.optimizer_G.zero_grad()
            fake_imgs = self.generator(low_imgs)
            pred_fake = self.discriminator(fake_imgs)
            loss_GAN = self.criterion_GAN(pred_fake, real_label)
            loss_L1 = self.criterion_L1(fake_imgs, high_imgs)
            loss_G = loss_GAN + (self.config['lambda_L1'] * loss_L1)
            loss_G.backward()
            self.optimizer_G.step()
            
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            
            pbar.set_postfix({
                'G_loss': f"{loss_G.item():.4f}",
                'D_loss': f"{loss_D.item():.4f}",
                'L1': f"{loss_L1.item():.4f}"
            })
            
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/G_loss', loss_G.item(), step)
            self.writer.add_scalar('Train/D_loss', loss_D.item(), step)
            self.writer.add_scalar('Train/L1_loss', loss_L1.item(), step)
        
        return epoch_g_loss / len(self.train_loader), epoch_d_loss / len(self.train_loader)
    
    def validate(self, epoch):
        """Validate the model"""
        self.generator.eval()
        val_loss = 0
        
        with torch.no_grad():
            for low_imgs, high_imgs in self.val_loader:
                low_imgs = low_imgs.to(self.device)
                high_imgs = high_imgs.to(self.device)
                
                fake_imgs = self.generator(low_imgs)
                loss = self.criterion_L1(fake_imgs, high_imgs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        self.writer.add_scalar('Validation/L1_loss', avg_val_loss, epoch)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model at epoch {epoch+1}")
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*50}")
        print(f"Starting Training")
        print(f"{'='*50}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"{'='*50}\n")
        
        for epoch in range(self.config['epochs']):
            train_g_loss, train_d_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"Train - G Loss: {train_g_loss:.4f}, D Loss: {train_d_loss:.4f}")
            print(f"Val - L1 Loss: {val_loss:.4f}")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.config['save_freq'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*50}\n")
        
        self.writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_dir', type=str, required=True)
    parser.add_argument('--high_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs/gan_training')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lambda_L1', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    
    config = vars(args)
    
    trainer = GANTrainer(config)
    trainer.train()