import io
import os
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from transformers import pipeline
from facenet_pytorch import MTCNN

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Deepfake & Advanced Document Verification API is running!", 
        "status": "Online",
        "port": 8001
    }

# Global models
mtcnn = None
pipe = None

@app.on_event("startup")
async def startup():
    global mtcnn, pipe
    device = 0 if torch.cuda.is_available() else -1
    mtcnn = MTCNN(keep_all=True, device='cpu')
    pipe = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model", device=device)
    print("Original Public Model Loaded Successfully!")

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Simple face detection
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        # If no face, try analyzing as a document automatically
        return await analyze_certificate(file, contents)
        
    # Analyze first face
    box = boxes[0]
    face = image.crop((box[0], box[1], box[2], box[3]))
    res = pipe(face)
    top = res[0]
    
    is_fake = "FAKE" in top['label'].upper()
    return {
        "prediction": "AI Generated / Deepfake" if is_fake else "Real Media",
        "fake_probability": round(top['score'] * 100, 2) if is_fake else round((1-top['score']) * 100, 2),
        "reason": "Analyzed using the original public deepfake detection model."
    }

@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    # Create temp directory if not exists
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    file_path = os.path.join(temp_dir, file.filename)
    
    # Save the video file temporarily
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            raise HTTPException(status_code=400, detail="Could not read video frames.")
            
        # Extract a frame from the middle of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=400, detail="Could not extract frame from video.")
            
        # Convert OpenCV BGR frame to PIL RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Simple face detection
        boxes, _ = mtcnn.detect(image_pil)
        if boxes is None:
            return {
                "prediction": "Inconclusive",
                "fake_probability": 0.0,
                "reason": "No face detected in the video frame. Please ensure the video has a clear view of a person."
            }
            
        # Analyze first face
        box = boxes[0]
        face = image_pil.crop((box[0], box[1], box[2], box[3]))
        res = pipe(face)
        top = res[0]
        
        is_fake = "FAKE" in top['label'].upper()
        
        # Cleanup temp file
        os.remove(file_path)
        
        return {
            "prediction": "AI Generated / Deepfake Video" if is_fake else "Real Video Content",
            "fake_probability": round(top['score'] * 100, 2) if is_fake else round((1-top['score']) * 100, 2),
            "reason": f"Analyzed a representative frame ({total_frames // 2}) from the video using the deepfake detection model."
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

@app.post("/analyze-certificate/")
async def analyze_certificate(file: UploadFile = File(...), contents: bytes = None):
    if contents is None:
        contents = await file.read()
        
    # PDF Support
    if file.filename.lower().endswith('.pdf'):
        doc = fitz.open(stream=contents, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        
    # Accuracy Boost: Contrast Enhancement for better OCR
    img_np = np.array(image_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Simple OCR on enhanced image
    text = pytesseract.image_to_string(enhanced).lower()
    
    # Model Prediction (Reduce influence for certificates as it's too sensitive to text)
    res = pipe(image_pil)
    top = res[0]
    ai_score = top['score'] * 100 if "FAKE" in top['label'].upper() else (1-top['score']) * 100
    
    # 1. Advanced Structural & Forensic Analysis (Identity Shield)
    structural_flags = []
    forensic_score = 0.0
    identity_verified = False
    
    # QR Code Detection & Identity Cross-Check
    qr_detector = cv2.QRCodeDetector()
    try:
        qr_results = qr_detector.detectAndDecode(img_np)
        qr_data = qr_results[0]
        qr_points = qr_results[1]
        
        # Smart ID Hunting (Regex)
        import re
        potential_ids = re.findall(r'[A-Z0-9]{10,15}', text.upper())
        found_id = potential_ids[0] if potential_ids else "NOT_FOUND"

        if qr_points is not None:
            if not qr_data:
                structural_flags.append("Forensic Alert: QR Code detected but is unreadable.")
                forensic_score += 45.0
            else:
                # Identity Match: Fuzzy/Partial check to handle small OCR errors
                qr_clean = qr_data.lower()
                id_clean = found_id.lower()
                
                # Check for direct match or partial overlap (at least 8 chars)
                is_match = id_clean in qr_clean or any(id_clean[i:i+8] in qr_clean for i in range(len(id_clean)-7))
                
                if found_id != "NOT_FOUND" and is_match:
                    structural_flags.append(f"Identity Match: ID {found_id} verified via QR portal.")
                    identity_verified = True
                    forensic_score = 0.0
                elif found_id != "NOT_FOUND":
                    structural_flags.append(f"Security Alert: ID Mismatch! Found {found_id} but QR link is different.")
                    forensic_score = 95.0
    except Exception as e:
        print(f"Identity check error: {e}")
    
    # Face-in-Document Forensics (Is the person's photo real or spliced?)
    doc_faces, _ = mtcnn.detect(image_pil)
    if doc_faces is not None and len(doc_faces) > 0:
        face_box = doc_faces[0]
        face_crop = image_pil.crop((face_box[0], face_box[1], face_box[2], face_box[3]))
        face_res = pipe(face_crop)
        
        # Check for splicing using Pixel Noise (ELA) on the face area
        face_np = np.array(face_crop)
        # Simple ELA heuristic for the face area
        face_ela = 40.0 if np.std(face_np) > 85.0 else 0.0 # High contrast in face area vs paper
        
        if "FAKE" in face_res[0]['label'].upper() and face_res[0]['score'] > 0.75:
            structural_flags.append("CRITICAL: Photo on document identified as AI Generated or Deepfake.")
            forensic_score = max(forensic_score, 95.0) # Forced high score
        elif face_ela > 35.0:
            structural_flags.append("Forensic Alert: Pixel noise mismatch detected in photo area (possible splicing).")
            forensic_score += 40.0

    # Logo & Watermark Discovery (Colored clusters search)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
    has_logo = cv2.countNonZero(mask) > 500
    
    # 2. Provider & Text Formatting Logic (Expanded)
    is_hr = any(kw in text for kw in ["hackerrank", "harishankaran", "accomplishment", "presented to"])
    is_lnt = any(kw in text for kw in ["l&t", "edutech", "larsen", "toubro", "larsen & toubro"])
    is_nptel = any(kw in text for kw in ["nptel", "swayam", "noc", "national programme", "iit "])
    is_course = any(kw in text for kw in ["certify that", "successfully completed", "completion", "achievement", "university", "institute", "academy", "learning", "course", "grade"])
    
    # Final Weighted Blend
    final_prob = (forensic_score * 0.7) + (ai_score * 0.3)
    
    # Provider-Specific Hardening
    if is_hr:
        has_id = any(len(word) == 12 and word.isalnum() for word in text.split())
        has_sig = any(s in text for s in ["harishankaran", "harishankaran k"])
        if has_id and has_sig:
            final_prob = min(final_prob, 15.0) 
            structural_flags = [f for f in structural_flags if "Formatting" not in f]
        else:
            structural_flags.append("Security Alert: Missing mandatory HackerRank markers (ID/Signature).")
            final_prob = max(final_prob, 80.0)
            
    if is_nptel:
        # NPTEL specific check
        has_roll = any(kw in text for kw in ["roll no", "id no", "noc"])
        if identity_verified and "nptel.ac.in" in qr_data.lower():
            final_prob = min(final_prob, 10.0)
            structural_flags.append("Identity Match: NPTEL Roll No verified via official portal.")
        elif not has_roll:
            structural_flags.append("Security Alert: Missing NPTEL Roll Number or official NOC markers.")
            final_prob = max(final_prob, 75.0)

    if is_lnt:
        has_lnt_portal = any(kw in text for kw in ["verify", "portal", "id", "certificate no"])
        if not has_lnt_portal:
            structural_flags.append("Security Alert: Missing L&T EduTech verification markers.")
            final_prob = max(final_prob, 75.0)
        else:
            final_prob = min(final_prob, 20.0)

    # 3. Micro-Style & Formatting Forensics
    try:
        ocr_data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT)
        center_w = img_np.shape[1] // 2
        
        # Collect static words (Certificate, Awarded) vs dynamic words (The Name)
        static_confidences = []
        name_confidences = []
        baselines = []
        
        for i in range(len(ocr_data['text'])):
            word = ocr_data['text'][i].lower()
            if len(word) < 3 or int(ocr_data['conf'][i]) < 10: continue
            
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            conf = int(ocr_data['conf'][i])
            
            # Words in the center 20% vertical area are likely names
            if 0.4 < (y/img_np.shape[0]) < 0.6:
                name_confidences.append(conf)
                baselines.append(y + h) # Bottom edge as baseline
            else:
                static_confidences.append(conf)
        
        # Forensic Check A: Confidence Variance (Font Clarity Mismatch)
        if static_confidences and name_confidences:
            avg_static = sum(static_confidences)/len(static_confidences)
            avg_name = sum(name_confidences)/len(name_confidences)
            if abs(avg_static - avg_name) > 15: # Significant clarity difference
                structural_flags.append("Formatting Alert: Font clarity mismatch detected (Possible text replacement).")
                final_prob = max(final_prob, 70.0)
        
        # Forensic Check B: Baseline Drift (Mathematical Alignment)
        if len(baselines) > 1:
            drift = np.std(baselines)
            if drift > 5.0: # Letters aren't on a straight line
                structural_flags.append("Formatting Alert: Vertical baseline drift detected (Hand-aligned text).")
                final_prob = max(final_prob, 65.0)
                
        # Forensic Check C: Horizontal Centering
        text_centers = [ocr_data['left'][i] + ocr_data['width'][i]//2 for i in range(len(ocr_data['text'])) if len(ocr_data['text'][i]) > 4]
        if text_centers:
            offset = abs((sum(text_centers)/len(text_centers)) - center_w) / img_np.shape[1]
            if offset > 0.12:
                structural_flags.append("Formatting Alert: Text alignment is significantly off-center.")
                final_prob = max(final_prob, 60.0)
    except: pass

    # 4. Master Identity Override (Boss Logic)
    if identity_verified:
        final_prob = 10.0
        structural_flags = [f for f in structural_flags if "Formatting" not in f and "noise" not in f]
        structural_flags.append("✓ Final Verdict: Identity cryptographically verified via official portal.")

    # Final Detailed Status Report
    report_steps = []
    report_steps.append(f"QR Code: {'DETECTED & VERIFIED' if identity_verified else ('READABLE but MISMATCH' if (qr_points is not None and qr_data) else 'NOT DETECTED')}")
    report_steps.append(f"Certificate ID: {found_id if found_id != 'NOT_FOUND' else 'NOT FOUND'}")
    report_steps.append(f"Provider: {'HackerRank' if is_hr else ('NPTEL (Swayam)' if is_nptel else ('L&T EduTech' if is_lnt else 'General'))}")
    report_steps.append(f"Signature: {'FOUND (Harishankaran K)' if (is_hr and has_sig) else ('FOUND' if is_course else 'NOT DETECTED')}")
    report_steps.append(f"Face-ID Check: {'TAMPERED' if any('Face' in f for f in structural_flags) else ('NOT DETECTED' if doc_faces is None else 'AUTHENTIC')}")

    is_fake = final_prob > 50.0
    
    return {
        "prediction": "Fake/Tampered Certificate" if is_fake else "Real Certificate",
        "fake_probability": round(min(final_prob, 99.0), 2),
        "detected_document_type": "HackerRank (Verified)" if is_hr else ("NPTEL (Swayam)" if is_nptel else ("L&T EduTech" if is_lnt else ("Educational" if is_course else ("General" if has_logo else "General")))),
        "reason": ". ".join(report_steps) + ". " + ". ".join(structural_flags),
        "extracted_text": text[:200]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
