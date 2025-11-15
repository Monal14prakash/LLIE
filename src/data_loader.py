import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
from pathlib import Path

class ImageEnhancementDataset(Dataset):
    """
    Dataset for paired low-light and high-quality images
    """
    def __init__(self, low_dir, high_dir, transform=None):
        """
        Args:
            low_dir: Directory with low-light images
            high_dir: Directory with high-quality images
            transform: Optional transforms to apply
        """
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(low_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.image_files)} image pairs")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image filename
        img_name = self.image_files[idx]
        
        # Load low and high quality images
        low_path = os.path.join(self.low_dir, img_name)
        high_path = os.path.join(self.high_dir, img_name)
        
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            # Use same random seed for both images to ensure same augmentation
            seed = torch.random.seed()
            
            torch.manual_seed(seed)
            low_img = self.transform(low_img)
            
            torch.manual_seed(seed)
            high_img = self.transform(high_img)
        
        return low_img, high_img

def get_transforms(image_size=256, is_train=True):
    """
    Get image transforms for training or testing
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def get_data_loaders(low_dir, high_dir, batch_size=16, image_size=256, num_workers=4):
    """
    Create train and validation data loaders
    """
    # Create dataset
    dataset = ImageEnhancementDataset(
        low_dir=low_dir,
        high_dir=high_dir,
        transform=get_transforms(image_size, is_train=True)
    )
    
    # Split into train and validation (80-20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Test the data loader
if __name__ == "__main__":
    LOW_DIR = "data/processed/low"
    HIGH_DIR = "data/processed/high"
    
    print("Testing data loader...")
    train_loader, val_loader = get_data_loaders(LOW_DIR, HIGH_DIR, batch_size=4)
    
    # Test loading one batch
    for low_imgs, high_imgs in train_loader:
        print(f"Low images shape: {low_imgs.shape}")
        print(f"High images shape: {high_imgs.shape}")
        print(f"Low images range: [{low_imgs.min():.2f}, {low_imgs.max():.2f}]")
        print(f"High images range: [{high_imgs.min():.2f}, {high_imgs.max():.2f}]")
        break
    
    print("✅ Data loader test successful!")