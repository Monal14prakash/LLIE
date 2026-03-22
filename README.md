# 🌟 Image Enhancement GAN

Brighten low-light images using Generative Adversarial Networks (U-Net Generator + PatchGAN Discriminator).

## 🎯 Features

- **Deep Learning**: GAN-based image enhancement
- **Architecture**: U-Net Generator (31M params) + PatchGAN Discriminator (2.7M params)
- **Performance**: PSNR 25.79 dB, SSIM 0.8346
- **Interactive Demo**: Streamlit web interface

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Monal14prakash/LLIE.git 
cd image-enhancement-gan
```

### 2. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### 3. Download Pre-trained Model
[Download best_model.pth](https://drive.google.com/file/d/1Um04-W_75WF-aWcWwoOzF0tWx5MBeVmN/view?usp=drive_link) and place in `checkpoints/`

### 4. Run Demo
```bash
streamlit run app/streamlit_app.py
```

## 📊 Project Structure
```
image-enhancement-gan/
├── data/
│   ├── raw/              # Original images
│   ├── processed/        # Training pairs
│   └── enhanced/         # Output images
├── src/
│   ├── generator.py      # U-Net Generator
│   ├── discriminator.py  # PatchGAN Discriminator
│   ├── data_loader.py    # Dataset & DataLoader
│   ├── train.py          # Training loop
│   └── evaluate.py       # Evaluation metrics
├── app/
│   └── streamlit_app.py  # Web demo interface
├── checkpoints/          # Model checkpoints
├── results/              # Evaluation results
└── requirements.txt
```

## 🎓 Training

### Prepare Dataset
```bash
# Add images to data/raw/
python src/create_synthetic_data.py
```

### Train Model
```bash
python src/train.py --epochs 50 --batch_size 8
```

### Evaluate
```bash
python src/evaluate.py
```

## 📈 Results

| Metric | Score |
|--------|-------|
| **PSNR** | 25.79 ± 2.45 dB |
| **SSIM** | 0.8346 ± 0.0899 |
| **Training** | 50 epochs |
| **Dataset** | 64 synthetic pairs |

## 🏗️ Architecture

### Generator (U-Net)
- Encoder-Decoder with skip connections
- 4 downsampling + 4 upsampling blocks
- Preserves fine details

### Discriminator (PatchGAN)
- 70×70 receptive field
- Outputs patch-wise realness scores
- Ensures local realism

## 🛠️ Technologies

- **Framework**: PyTorch
- **UI**: Streamlit
- **Metrics**: PSNR, SSIM
- **Visualization**: TensorBoard, Matplotlib

## 📝 TODO

- [ ] Train on real LOL dataset
- [ ] Increase to 200 epochs
- [ ] Add more augmentation
- [ ] Deploy to Streamlit Cloud

## 👨‍💻 Authors

- Monal prakash - [GitHub] https://github.com/Monal14prakash
- Ishita Goyal - [GitHub] https://github.com/aiguanai 

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- LOL Dataset
- PyTorch team
- Streamlit community
