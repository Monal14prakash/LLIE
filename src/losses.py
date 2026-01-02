"""
Custom loss functions for image enhancement [added after colab model turned out poor]
"""
import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss
    Compares high-level features instead of raw pixels
    Results in better visual quality and detail preservation
    """
    def __init__(self, device='cpu'):
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True)
        
        # Use features up to conv3_3 (layer 16)
        # This captures mid-level features (edges, textures)
        self.feature_extractor = vgg.features[:16].to(device).eval()
        
        # Freeze VGG parameters (don't train it)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()
        
        # Normalization values for VGG (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_vgg(self, x):
        """
        Normalize input for VGG
        Input range: [-1, 1] (from our generator)
        VGG expects: ImageNet normalized
        """
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        
        # Normalize with ImageNet stats
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, output, target):
        """
        Compare VGG features of output vs target
        """
        # Normalize for VGG
        output_norm = self.normalize_vgg(output)
        target_norm = self.normalize_vgg(target)
        
        # Extract features
        output_features = self.feature_extractor(output_norm)
        target_features = self.feature_extractor(target_norm)
        
        # Compare features (L1 distance)
        loss = self.criterion(output_features, target_features)
        return loss


class ColorLoss(nn.Module):
    """
    Color consistency loss
    Prevents color shifts and over-saturation
    """
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.criterion = nn.L1Loss()
    
    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space"""
        # This is a simplified version
        # For production, use proper color space conversion
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        
        max_rgb, _ = torch.max(rgb, dim=1, keepdim=True)
        min_rgb, _ = torch.min(rgb, dim=1, keepdim=True)
        delta = max_rgb - min_rgb
        
        # Value
        v = max_rgb
        
        # Saturation
        s = delta / (max_rgb + 1e-8)
        
        return s, v
    
    def forward(self, output, target):
        """
        Compare color saturation and value
        """
        # Get saturation and value
        output_s, output_v = self.rgb_to_hsv(output)
        target_s, target_v = self.rgb_to_hsv(target)
        
        # Loss on saturation (prevent over-saturation)
        loss_s = self.criterion(output_s, target_s)
        
        # Loss on value/brightness (prevent over-brightness)
        loss_v = self.criterion(output_v, target_v)
        
        return loss_s + loss_v
#trying to push this 