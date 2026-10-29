# 🎨 NeuroCanvas
### Arbitrary Neural Style Transfer Using Adaptive Instance Normalization (AdaIN)

**Transform ordinary photographs into artistic artwork using Deep Learning.**

🌐 **Live Demo:** https://adain-neural-style-transfer-7dxl.onrender.com/

💻 **GitHub Repository:** https://github.com/ritikasainii1234-cloud/AdaIN-Neural-Style-Transfer

---

## 📌 Project Overview

NeuroCanvas is an AI-powered image stylization application developed using **Adaptive Instance Normalization (AdaIN)**.

The system combines the content of one image with the artistic style of another to generate a stylized image.

It uses a pretrained VGG encoder, AdaIN feature transformation, and a trained decoder to perform arbitrary neural style transfer.

The project includes an interactive Flask web application deployed on Render.

## ✨ Key Features

- 🎨 Arbitrary neural style transfer
- 🧠 VGG-based feature extraction
- ⚡ Adaptive Instance Normalization
- 🖼️ Custom content and style image uploads
- 🔄 Image generation using a trained decoder
- 💻 Interactive Flask web application
- ☁️ Cloud deployment using Render
- 🚀 GPU-accelerated training on Kaggle

## 🧠 Model Architecture

```text
Content Image ──► VGG Encoder ──► Content Features ──┐
                                                      │
                                                      ▼
                                                    AdaIN
                                                      │
                                                      ▼
Style Image ───► VGG Encoder ──► Style Features ──────┘
                                                      │
                                                      ▼
                                               Trained Decoder
                                                      │
                                                      ▼
                                                Stylized Image
```

### Architecture Components

**1. VGG Encoder:** Extracts deep features from content and style images.

**2. AdaIN:** Aligns the mean and standard deviation of content features with style features.

**3. Decoder:** Reconstructs transformed features into a stylized output image.

**4. Flask Application:** Provides a web interface for uploading images and viewing results.

## 🔬 Adaptive Instance Normalization

AdaIN transfers style by adjusting the statistical properties of content features.

**Formula:**

AdaIN(c, s) = σ(s) × [(c − μ(c)) / σ(c)] + μ(s)

Where:

- **c:** Content features
- **s:** Style features
- **μ:** Channel-wise mean
- **σ:** Channel-wise standard deviation

### Why AdaIN?

- Supports arbitrary artistic styles.
- Does not require a separate model for each style.
- Performs efficient feature-level transformations.
- Avoids iterative optimization during inference.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision, Pillow |
| Neural Network | VGG Encoder |
| Algorithm | Adaptive Instance Normalization |
| Backend | Flask |
| Frontend | HTML, CSS |
| GPU Training | Kaggle Tesla T4 |
| Development | VS Code |
| Version Control | Git, GitHub |
| Deployment | Render |

## 📁 Project Structure

```text
AdaIN-Neural-Style-Transfer/
│
├── Demo_IO_Images/
├── NST_Code/
│   ├── app.py
│   ├── train.py
│   ├── utils/
│   │   ├── models.py
│   │   └── utils.py
│   └── static/
├── requirements.txt
├── Procfile.txt
└── README.md
```

*Additional templates, datasets, and model checkpoints are stored in their respective project directories.*

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/ritikasainii1234-cloud/AdaIN-Neural-Style-Transfer.git
```

### Step 2: Navigate to the Project

```bash
cd AdaIN-Neural-Style-Transfer
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment

**Windows:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
cd NST_Code
python app.py
```

Ensure the required VGG encoder and trained decoder checkpoints are available at the paths expected by the application.

## 🏋️ Model Training

The decoder was trained using PyTorch on an NVIDIA Tesla T4 GPU through Kaggle.

### Training Process

1. Load content and style image datasets.
2. Preprocess the input images.
3. Extract features using a pretrained VGG encoder.
4. Apply Adaptive Instance Normalization.
5. Reconstruct images using the decoder.
6. Calculate content and style losses.
7. Optimize decoder parameters through backpropagation.
8. Save the trained model checkpoint.

### Training Objective

**Total Loss = Content Loss + λ × Style Loss**

- **Content Loss:** Encourages preservation of structural content.
- **Style Loss:** Encourages matching of artistic feature statistics.
- **λ:** Controls the contribution of style loss.

### Model Checkpoints

- `vgg_normalised.pth` — Pretrained VGG encoder weights
- `decoder_final.pth` — Trained decoder weights

## 🖼️ Results

The application accepts two inputs:

**Content Image:** The original image whose structure should be preserved.

**Style Image:** The reference image containing the desired artistic characteristics.

**Generated Output:** An image combining the original content with the reference artistic style.

### Processing Pipeline

```text
Content Image + Style Image
            |
            ▼
      VGG Encoder
            |
            ▼
           AdaIN
            |
            ▼
      Trained Decoder
            |
            ▼
      Stylized Output
```

The application allows users to experiment with different image combinations and generate artistic results.

## ☁️ Deployment

The project is deployed using Render.

### 🌐 Live Application

**[🎨 Launch NeuroCanvas](https://adain-neural-style-transfer-7dxl.onrender.com/)**

**Deployment Details:**

- Platform: Render
- Backend: Flask
- Deep Learning: PyTorch
- Model: VGG + AdaIN + Decoder

**Note:** The application uses Render's free tier, so the first request after inactivity may take additional time.

## 🎯 Applications

- Digital art generation
- Creative image editing
- Artistic photo transformation
- Photography enhancement
- Computer vision research
- AI-assisted content creation

## 🚀 Future Enhancements

- Improve stylization quality with larger datasets.
- Optimize inference speed.
- Support higher-resolution images.
- Introduce adjustable style-strength controls.
- Add more artistic examples.
- Improve mobile responsiveness.

## 📚 Research Reference

Huang, X., & Belongie, S. (2017).

**Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.**

IEEE International Conference on Computer Vision (ICCV).

## 👩‍💻 Project Information

**Project Name:** NeuroCanvas

**Project Title:** Arbitrary Neural Style Transfer Using Adaptive Instance Normalization (AdaIN)

**Domain:** Artificial Intelligence, Deep Learning & Computer Vision

**Project Type:** B.Tech Major Project

**Technologies:** Python, PyTorch, VGG, AdaIN, Flask, Render

**Live Demo:** https://adain-neural-style-transfer-7dxl.onrender.com/

---

### 🎨 NeuroCanvas — Where Deep Learning Meets Creativity
