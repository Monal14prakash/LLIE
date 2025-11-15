import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator
    Outputs a feature map where each element represents the realness of a patch
    Receptive field: 70x70 patches
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(PatchGANDiscriminator, self).__init__()
        
        layers = []
        
        # First layer (no normalization)
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        
        # Hidden layers
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                self._block(in_channels, feature, stride=2)
            )
            in_channels = feature
        
        # Final layer - outputs probability map
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.model = nn.Sequential(*layers)
    
    def _block(self, in_channels, out_channels, stride):
        """
        Discriminator block: Conv -> BatchNorm -> LeakyReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass
        Returns: NxN feature map (not a single value!)
        Each pixel in output corresponds to a patch in input
        """
        return self.model(x)

# Test the discriminator
if __name__ == "__main__":
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PatchGANDiscriminator(in_channels=3).to(device)
    
    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output is a {output.shape[2]}x{output.shape[3]} feature map")
    print(f"Each value represents realness of a patch")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("✅ Discriminator test successful!")