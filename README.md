# GleasonAI 

> AI-assisted Gleason grading of prostate histopathology images using semantic segmentation.

GleasonAI is a full-stack web application developed as part of my **MSc Computer Science dissertation at the University of Edinburgh**. It allows users to upload a prostate histopathology tissue image and receive an automatically generated semantic segmentation mask together with a predicted Gleason score.

The deployed application uses a **U-Net with an EfficientNet-B4 encoder** trained for multi-class semantic segmentation of prostate tissue.

> **Disclaimer**
>
> This application is a research prototype created for academic purposes only. It is **not** a medical device and must **not** be used for clinical diagnosis, treatment planning, or patient care.

---

## Features

- Upload prostate histopathology images
- Semantic segmentation of tissue into Gleason grades
- Automatic Gleason score prediction
- Interactive visualization of segmentation masks
- Color legend for predicted tissue classes
- Responsive React interface
- FastAPI inference API
- Automatic model download from Hugging Face

---

## Demo

The application follows a simple workflow:

1. Upload a prostate tissue image.
2. The image is passed through the trained segmentation model.
3. A pixel-wise segmentation mask is generated.
4. The two most predominant Gleason grades are identified.
5. The corresponding Gleason score is displayed alongside the segmentation result.

---

## Model

The deployed model is based on the U-Net architecture implemented using **Segmentation Models PyTorch**.

**Architecture**

- U-Net
- EfficientNet-B4 encoder
- ImageNet pretrained encoder
- Weighted Cross-Entropy + Dice loss
- Adam optimizer

During inference the uploaded image is:

- resized to **1024 × 1024**
- divided into overlapping **512 × 512** patches
- segmented patch-by-patch
- reconstructed using Gaussian-weighted stitching
- converted into a final Gleason score by identifying the two predominant Gleason grades while ignoring very small predicted regions. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Predicted Classes

| Color | Tissue |
|--------|--------|
| ⚪ White | Background |
| 🟢 Green | Benign |
| 🔵 Blue | Gleason Grade 3 |
| 🟡 Yellow | Gleason Grade 4 |
| 🔴 Red | Gleason Grade 5 |

---

## Technology Stack

### Frontend

- React
- Vite
- CSS

### Backend

- FastAPI
- PyTorch
- Segmentation Models PyTorch
- NumPy
- Pillow

### Deployment

- Vercel
- Render
- Hugging Face

---

## Installation

### Clone the repository

```bash
git clone https://github.com/FatimahAljishi/GleasonAI.git

cd GleasonAI
```

### Backend

```bash
cd backend

python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the backend

```bash
uvicorn app.main:app --reload
```

---

### Frontend

```bash
cd frontend

npm install

npm run dev
```
---

