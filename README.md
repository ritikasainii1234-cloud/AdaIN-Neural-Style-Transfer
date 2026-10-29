# 🎨 STYLEFORGE AI
### Arbitrary Neural Style Transfer Using Adaptive Instance Normalization (AdaIN)

**Transform ordinary photographs into artistic creations using Deep Learning.**

🚀 **[Live Demo — Try StyleForge AI](https://adain-neural-style-transfer-7dxl.onrender.com/)**

💻 **[GitHub Repository — Source Code](https://github.com/ritikasainii1234-cloud/AdaIN-Neural-Style-Transfer)**

---

## 📌 Project Overview

**StyleForge AI** is a deep learning-powered web application that performs **Arbitrary Neural Style Transfer** using **Adaptive Instance Normalization (AdaIN)**.

The application combines the structural content of one image with the artistic characteristics of another image to generate a stylized output.

Unlike traditional neural style transfer methods that require iterative optimization for each image, AdaIN transforms deep feature representations using an encoder-decoder architecture.

The project integrates **PyTorch for deep learning**, **Flask for the web application**, and **Render for cloud deployment**.

---

## ✨ Key Features

- 🎨 **Arbitrary Style Transfer:** Apply artistic styles from reference images.
- 🧠 **VGG Feature Extraction:** Extract deep representations of content and style images.
- ⚡ **Adaptive Instance Normalization:** Transfer style using feature statistics.
- 🖼️ **Custom Image Uploads:** Upload content and style images directly.
- 🔄 **Trained Decoder:** Reconstruct stylized outputs.
- 💻 **Interactive Web Interface:** Flask-based application.
- 🚀 **GPU Training:** Model trained using Kaggle's NVIDIA Tesla T4 GPU.
- ☁️ **Cloud Deployment:** Publicly accessible through Render.

---

## 🧠 System Architecture

StyleForge AI uses an **Encoder → AdaIN → Decoder** architecture.

```text
       CONTENT IMAGE                  STYLE IMAGE
             |                             |
             v                             v
       +-------------+               +-------------+
       | VGG Encoder |               | VGG Encoder |
       +-------------+               +-------------+
             |                             |
             v                             v
      Content Features               Style Features
             |                             |
             +-------------+---------------+
                           |
                           v
                    +-------------+
                    |    AdaIN    |
                    +-------------+
                           |
                           v
                  Transformed Features
                           |
                           v
                    +-------------+
                    |   Decoder   |
                    +-------------+
                           |
                           v
                    STYLIZED IMAGE
```

### Architecture Components

**1. VGG Encoder**

A pretrained VGG network extracts deep feature representations from the content and style images.

**2. Adaptive Instance Normalization (AdaIN)**

AdaIN aligns the channel-wise mean and standard deviation of content features with those of the style features.

**3. Decoder**

A trained convolutional decoder reconstructs the transformed feature representations into stylized images.

**4. Flask Web Application**

Provides an interactive interface for uploading images, performing inference, and displaying generated outputs.

---

## 🔬 How AdaIN Works

Adaptive Instance Normalization transfers artistic style by aligning the statistical properties of content features with those of style features.

### Mathematical Formula

```text
AdaIN(c, s) = σ(s) × [(c − μ(c)) / σ(c)] + μ(s)
```

Where:

- **c:** Content feature representation
- **s:** Style feature representation
- **μ:** Channel-wise mean
- **σ:** Channel-wise standard deviation

### Working Process

1. Upload a content image and a style image.
2. Extract features using the pretrained VGG encoder.
3. Calculate content and style feature statistics.
4. Apply AdaIN to align content features with style statistics.
5. Pass the transformed features through the trained decoder.
6. Generate the final stylized image.

### Why AdaIN?

- Supports arbitrary artistic styles.
- Does not require a separate model for every style.
- Avoids iterative image optimization during inference.
- Enables efficient feature-level transformations.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision, Pillow |
| Feature Extractor | Pretrained VGG |
| Style Transfer Algorithm | AdaIN |
| Backend | Flask |
| Frontend | HTML, CSS |
| Training Environment | Kaggle GPU (Tesla T4) |
| Development | VS Code |
| Version Control | Git, GitHub |
| Deployment | Render |

---

## 📁 Project Structure

```text
AdaIN-Neural-Style-Transfer/
│
├── Demo_IO_Images/
│
├── NST_Code/
│   ├── app.py
│   ├── train.py
│   ├── utils/
│   │   ├── models.py
│   │   └── utils.py
│   └── static/
│
├── .gitattributes
├── .gitignore
├── Procfile.txt
├── code.ipynb
├── requirements.txt
└── README.md
```

**Model Checkpoints:**

- `vgg_normalised.pth` — Pretrained VGG encoder weights
- `decoder_final.pth` — Trained decoder weights

The model files must be placed at the paths expected by the application.

---

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

Navigate to the directory containing `app.py`:

```bash
cd NST_Code
python app.py
```

Make sure the required model checkpoints are accessible before running the application.

Open the local address displayed in the terminal.

---

## 🏋️ Model Training

The decoder was trained using **PyTorch** with GPU acceleration on **Kaggle**.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Platform | Kaggle |
| GPU | NVIDIA Tesla T4 |
| Framework | PyTorch |
| Encoder | Pretrained VGG |
| Style Transfer | Adaptive Instance Normalization |
| Decoder | Trainable Convolutional Decoder |
| Optimization | Backpropagation |
| Output | Trained Decoder Checkpoint |

### Training Pipeline

1. Load content and style image datasets.
2. Apply image preprocessing.
3. Extract features using the pretrained VGG encoder.
4. Apply Adaptive Instance Normalization.
5. Reconstruct images using the decoder.
6. Calculate content and style losses.
7. Optimize decoder parameters through backpropagation.
8. Save the trained decoder weights.

### Training Objective

```text
Total Loss = Content Loss + λ × Style Loss
```

- **Content Loss:** Encourages preservation of content information.
- **Style Loss:** Encourages matching of artistic feature statistics.
- **λ:** Controls the contribution of style loss.

---

## 🖼️ Results and Output

StyleForge AI accepts two images and generates one stylized output.

| Input/Output | Description |
|---|---|
| Content Image | Original image whose structure is preserved |
| Style Image | Reference image providing artistic characteristics |
| Generated Image | Final stylized output |

### Image Generation Workflow

```text
Content Image + Style Image
            |
            v
     VGG Feature Extraction
            |
            v
            AdaIN
            |
            v
      Trained Decoder
            |
            v
      Stylized Output
```

### Output Examples

The application supports artistic transformations using different combinations of content and style images.

Actual output screenshots can be added to this section to demonstrate the results.

---

## ☁️ Deployment

StyleForge AI is deployed using **Render**.

### 🌐 Live Application

### [🎨 Launch StyleForge AI](https://adain-neural-style-transfer-7dxl.onrender.com/)

**Deployment Details:**

- **Platform:** Render
- **Backend:** Flask
- **Deep Learning:** PyTorch
- **Architecture:** VGG + AdaIN + Decoder

> **Note:** The application uses Render's free tier. Initial loading may take additional time when the service restarts after inactivity.

---

## 🎯 Applications

- 🎨 Digital art generation
- 🖼️ Creative image editing
- 📸 Artistic photography
- 💻 AI-assisted content creation
- 🧠 Computer vision research
- 🎭 Visual design and artistic experimentation

---

## 🚀 Future Enhancements

- Improve stylization quality with larger datasets.
- Optimize model inference speed.
- Support higher-resolution images.
- Introduce adjustable style-strength controls.
- Add more artistic style examples.
- Improve mobile responsiveness.
- Explore lightweight model architectures.

---

## 📚 Research Reference

**Huang, X., & Belongie, S. (2017).**

*Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.*

Proceedings of the IEEE International Conference on Computer Vision (ICCV).

---

## 👩‍💻 Project Information

**Project Name:** STYLEFORGE AI

**Project Title:** Arbitrary Neural Style Transfer Using Adaptive Instance Normalization (AdaIN)

**Domain:** Artificial Intelligence | Deep Learning | Computer Vision

**Project Type:** Personal AI/ML Portfolio Project

**Technologies:** Python, PyTorch, VGG, AdaIN, Flask, Render

**Live Demo:** [StyleForge AI Web Application](https://adain-neural-style-transfer-7dxl.onrender.com/)

**Source Code:** [GitHub Repository](https://github.com/ritikasainii1234-cloud/AdaIN-Neural-Style-Transfer)

---

### 🎨 STYLEFORGE AI — Where Deep Learning Meets Creativity
