import cv2
import numpy as np
import os
from pathlib import Path

def create_low_light_image(image, gamma=2.5, noise_level=10):
    """
    Create a synthetic low-light version of an image
    
    Args:
        image: Input image (numpy array)
        gamma: Gamma correction value (higher = darker)
        noise_level: Amount of noise to add
    """
    # Apply gamma correction to darken
    normalized = image / 255.0
    darkened = np.power(normalized, gamma)
    darkened = (darkened * 255).astype(np.uint8)
    
    # Add noise
    noise = np.random.normal(0, noise_level, image.shape)
    noisy = np.clip(darkened + noise, 0, 255).astype(np.uint8)
    
    return noisy

def process_dataset(input_dir, output_dir_low, output_dir_high):
    """
    Process all images in input directory and create paired dataset
    """
    # Create output directories
    Path(output_dir_low).mkdir(parents=True, exist_ok=True)
    Path(output_dir_high).mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for fmt in formats:
        image_files.extend(Path(input_dir).glob(fmt))
    
    print(f"Found {len(image_files)} images to process")
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        print("Please add some images to data/raw/ directory")
        return
    
    for idx, img_path in enumerate(image_files):
        # Read image
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"Failed to read: {img_path}")
            continue
        
        # Resize to reasonable size (optional, for faster training)
        height, width = image.shape[:2]
        if height > 512 or width > 512:
            scale = 512 / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Create low-light version
        low_light = create_low_light_image(image)
        
        # Save both versions
        filename = f"image_{idx:04d}.png"
        cv2.imwrite(os.path.join(output_dir_low, filename), low_light)
        cv2.imwrite(os.path.join(output_dir_high, filename), image)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images")
    
    print(f"\n✅ Dataset creation complete!")
    print(f"Low-light images: {output_dir_low}")
    print(f"High-quality images: {output_dir_high}")
    print(f"Total pairs created: {len(image_files)}")

if __name__ == "__main__":
    # Configure paths
    INPUT_DIR = "data/raw"
    OUTPUT_LOW = "data/processed/low"
    OUTPUT_HIGH = "data/processed/high"
    
    print("Starting dataset generation...")
    print(f"Looking for images in: {INPUT_DIR}\n")
    
    process_dataset(INPUT_DIR, OUTPUT_LOW, OUTPUT_HIGH)