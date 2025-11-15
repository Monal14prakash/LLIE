import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from generator import UNetGenerator

# Page configuration
st.set_page_config(
    page_title="Image Enhancement GAN",
    page_icon="🌟",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(checkpoint_path):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator().to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['generator_state_dict'])
        model.eval()
        return model, device, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device, False

def transform_image(image):
    """Transform PIL image to tensor"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def denormalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    tensor = denormalize(tensor)
    tensor = tensor.squeeze().cpu().clamp(0, 1)
    np_img = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray((np_img * 255).astype(np.uint8))

def calculate_simple_metrics(original, enhanced):
    """Calculate simple brightness improvement metric"""
    orig_brightness = np.array(original.convert('L')).mean()
    enh_brightness = np.array(enhanced.convert('L')).mean()
    improvement = ((enh_brightness - orig_brightness) / orig_brightness) * 100
    return orig_brightness, enh_brightness, improvement

# Header
st.markdown('<p class="main-header">🌟 Image Enhancement GAN</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Brighten your low-light images with AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This application uses a **Generative Adversarial Network (GAN)** 
    to enhance low-light images.
    
    **Architecture:**
    - Generator: U-Net (31M parameters)
    - Discriminator: PatchGAN (2.7M parameters)
    
    **Training:**
    - Dataset: 64 synthetic image pairs
    - Epochs: 50
    - Validation Loss: 0.0862
    
    **Performance:**
    - PSNR: 25.79 ± 2.45 dB
    - SSIM: 0.8346 ± 0.0899
    """)
    
    st.header("Instructions")
    st.write("""
    1. Upload a low-light image
    2. Wait for processing
    3. Compare results
    4. Download enhanced image
    5. (Optional) Provide feedback
    """)

# Load model
checkpoint_path = os.path.join('checkpoints', 'best_model.pth')
model, device, model_loaded = load_model(checkpoint_path)

if not model_loaded:
    st.error("❌ Model not found! Please train the model first.")
    st.stop()

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a low-light image", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image that you want to enhance"
    )

if uploaded_file is not None:
    # Load and display original image
    original_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)
    
    # Process image
    with st.spinner('🔄 Enhancing image...'):
        # Transform and enhance
        input_tensor = transform_image(original_image).to(device)
        
        with torch.no_grad():
            enhanced_tensor = model(input_tensor)
        
        enhanced_image = tensor_to_pil(enhanced_tensor)
        
        # Calculate metrics
        orig_bright, enh_bright, improvement = calculate_simple_metrics(
            original_image, enhanced_image
        )
    
    # Display enhanced image
    with col2:
        st.subheader("✨ Enhanced Image")
        st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)
    
    # Metrics
    st.subheader("📊 Enhancement Metrics")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric(
            label="Original Brightness",
            value=f"{orig_bright:.1f}",
            help="Average brightness of original image (0-255)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric(
            label="Enhanced Brightness",
            value=f"{enh_bright:.1f}",
            delta=f"{improvement:+.1f}%",
            help="Average brightness of enhanced image"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric(
            label="Model Confidence",
            value="High" if improvement > 20 else "Medium",
            help="Based on brightness improvement"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download button
    st.subheader("💾 Download Enhanced Image")
    
    # Convert to bytes for download
    from io import BytesIO
    buf = BytesIO()
    enhanced_image.save(buf, format='PNG')
    byte_im = buf.getvalue()
    
    st.download_button(
        label="⬇️ Download Enhanced Image",
        data=byte_im,
        file_name="enhanced_image.png",
        mime="image/png"
    )
    
    # Feedback section
    st.subheader("📝 Feedback (Optional)")
    with st.form("feedback_form"):
        rating = st.slider("How satisfied are you with the enhancement?", 1, 5, 3)
        comments = st.text_area("Additional comments:")
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            # Save feedback to file
            feedback_file = "results/feedback.txt"
            os.makedirs("results", exist_ok=True)
            
            with open(feedback_file, "a") as f:
                f.write(f"\n--- New Feedback ---\n")
                f.write(f"Rating: {rating}/5\n")
                f.write(f"Comments: {comments}\n")
                f.write(f"Brightness Improvement: {improvement:.1f}%\n")
            
            st.success("✅ Thank you for your feedback!")

else:
    # Show example
    st.info("👆 Upload an image to get started!")
    
    # Show sample if exists
    sample_path = "data/processed/low/image_0000.png"
    if os.path.exists(sample_path):
        st.subheader("📸 Sample Images")
        st.write("Here's an example from our training data:")
        
        sample_low = Image.open(sample_path)
        sample_high_path = sample_path.replace('/low/', '/high/')
        
        if os.path.exists(sample_high_path):
            sample_high = Image.open(sample_high_path)
            
            demo_col1, demo_col2 = st.columns(2)
            with demo_col1:
                st.image(sample_low, caption="Low-light Sample", use_container_width=True)
            with demo_col2:
                st.image(sample_high, caption="Target (Ground Truth)", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using PyTorch & Streamlit | 
        GAN Architecture: U-Net Generator + PatchGAN Discriminator
    </div>
""", unsafe_allow_html=True)