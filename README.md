# Deepfake & Certificate Authenticity Detector

A sophisticated AI-powered system for detecting deepfakes in media and verifying the authenticity of certificates.

## 🚀 Features

- **Deepfake Detection**: Analyze images and videos to detect AI-generated faces and synthetic pixel artifacts.
- **Certificate Verification**:
    - **OCR Analysis**: Extracts and validates structural keywords for various document types (Educational, Birth, Identity, Medical).
    - **Visual Forensic Analysis**: Uses Error Level Analysis (ELA) and a fine-tuned Vision Transformer (ViT) model to detect logo splicing and text tampering.
    - **Grayscale Optimization**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to expose hidden artifacts in black & white documents.
- **Unified Dashboard**: Modern React frontend for easy file uploads and detailed analysis reports.

## 🛠️ Technology Stack

- **Backend**: FastAPI, PyTorch, Hugging Face Transformers (ViT), OpenCV, Tesseract OCR.
- **Frontend**: React, TailwindCSS (for styling), Axios, Lucide-React.
- **Models**: Fine-tuned `prithivMLmods/Deep-Fake-Detector-v2-Model` for both faces and document tampering.

## 📋 Installation

### Backend
1. Navigate to the `backend` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Tesseract OCR is installed on your system (default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`).

### Frontend
1. Navigate to the `frontend` directory.
2. Install dependencies:
   ```bash
   npm install
   ```

## 🏃 Running the Project

1. **Start Backend**:
   ```bash
   python backend/main.py
   ```
2. **Start Frontend**:
   ```bash
   npm run dev
   ```

## 🧠 Training

To fine-tune the models on your own dataset:
1. Place images in `dataset/images` or certificates in `dataset/certificates/extract`.
2. Run the training script:
   ```bash
   python backend/train.py
   ```

## 📄 License
This project is for educational and research purposes.
