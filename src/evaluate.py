import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from generator import UNetGenerator
from data_loader import get_data_loaders

class ModelEvaluator:
    def __init__(self, checkpoint_path, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.generator = UNetGenerator().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
        print(f"✅ Loaded model from {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch'] + 1}")
        print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
        
        # Get data loader
        _, self.val_loader = get_data_loaders(
            low_dir=config['low_dir'],
            high_dir=config['high_dir'],
            batch_size=1,  # Process one at a time for metrics
            image_size=config['image_size']
        )
        
        # Create results directory
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def denormalize(self, tensor):
        """Convert from [-1, 1] to [0, 1]"""
        return (tensor + 1) / 2
    
    def tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array for metric calculation"""
        img = self.denormalize(tensor)
        img = img.squeeze().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        return np.clip(img, 0, 1)
    
    def calculate_metrics(self, generated, target):
        """Calculate PSNR and SSIM"""
        gen_np = self.tensor_to_numpy(generated)
        target_np = self.tensor_to_numpy(target)
        
        # PSNR
        psnr_value = psnr(target_np, gen_np, data_range=1.0)
        
        # SSIM
        ssim_value = ssim(target_np, gen_np, channel_axis=2, data_range=1.0)
        
        return psnr_value, ssim_value
    
    def evaluate(self):
        """Evaluate model on validation set"""
        print("\n" + "="*50)
        print("Starting Evaluation")
        print("="*50 + "\n")
        
        psnr_scores = []
        ssim_scores = []
        
        with torch.no_grad():
            for idx, (low_img, high_img) in enumerate(tqdm(self.val_loader, desc="Evaluating")):
                low_img = low_img.to(self.device)
                high_img = high_img.to(self.device)
                
                # Generate enhanced image
                enhanced_img = self.generator(low_img)
                
                # Calculate metrics
                psnr_val, ssim_val = self.calculate_metrics(enhanced_img, high_img)
                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)
                
                # Save comparison image
                if idx < 10:  # Save first 10 comparisons
                    comparison = torch.cat([
                        self.denormalize(low_img),
                        self.denormalize(enhanced_img),
                        self.denormalize(high_img)
                    ], dim=3)  # Concatenate horizontally
                    
                    save_path = self.results_dir / f"comparison_{idx:03d}.png"
                    save_image(comparison, save_path)
        
        # Calculate statistics
        avg_psnr = np.mean(psnr_scores)
        std_psnr = np.std(psnr_scores)
        avg_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)
        
        # Print results
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        print(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        print(f"\nComparison images saved to: {self.results_dir}")
        print("="*50 + "\n")
        
        # Save metrics to file
        with open(self.results_dir / "metrics.txt", "w") as f:
            f.write(f"Evaluation Metrics\n")
            f.write(f"==================\n\n")
            f.write(f"Number of images: {len(psnr_scores)}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n\n")
            f.write(f"Individual Scores:\n")
            f.write(f"-----------------\n")
            for i, (p, s) in enumerate(zip(psnr_scores, ssim_scores)):
                f.write(f"Image {i+1}: PSNR={p:.2f} dB, SSIM={s:.4f}\n")
        
        # Create visualization
        self.plot_metrics(psnr_scores, ssim_scores)
        
        return avg_psnr, avg_ssim
    
    def plot_metrics(self, psnr_scores, ssim_scores):
        """Create metric visualization plots"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # PSNR histogram
        axes[0].hist(psnr_scores, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('PSNR (dB)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('PSNR Distribution')
        axes[0].axvline(np.mean(psnr_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(psnr_scores):.2f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # SSIM histogram
        axes[1].hist(ssim_scores, bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('SSIM')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('SSIM Distribution')
        axes[1].axvline(np.mean(ssim_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(ssim_scores):.4f}')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'metrics_distribution.png', dpi=150)
        print(f"✅ Metrics plot saved to: {self.results_dir / 'metrics_distribution.png'}")
        plt.close()

if __name__ == "__main__":
    config = {
        'low_dir': 'data/processed/low',
        'high_dir': 'data/processed/high',
        'image_size': 256,
        'results_dir': 'results'
    }
    
    checkpoint_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first!")
    else:
        evaluator = ModelEvaluator(checkpoint_path, config)
        evaluator.evaluate()