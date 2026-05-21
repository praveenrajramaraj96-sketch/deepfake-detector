import cv2
import os
from tqdm import tqdm

# 1. Configuration
REAL_VID_DIR = r"c:\Users\Delli5\OneDrive\deepfake detector\dataset\videos\SDFVD Small-scale Deepfake Forgery Video Dataset\SDFVD\SDFVD\videos_real"
FAKE_VID_DIR = r"c:\Users\Delli5\OneDrive\deepfake detector\dataset\videos\SDFVD Small-scale Deepfake Forgery Video Dataset\SDFVD\SDFVD\videos_fake"

REAL_OUT_DIR = r"c:\Users\Delli5\OneDrive\deepfake detector\dataset\images\Dataset\Train\Real"
FAKE_OUT_DIR = r"c:\Users\Delli5\OneDrive\deepfake detector\dataset\images\Dataset\Train\Fake"

FRAMES_PER_VIDEO = 15  # Extract 15 frames per video to build a strong dataset

def extract_frames(video_dir, output_dir, label_prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    print(f"Found {len(videos)} videos in {video_dir}")
    
    for vid_name in tqdm(videos, desc=f"Processing {label_prefix} videos"):
        vid_path = os.path.join(video_dir, vid_name)
        cap = cv2.VideoCapture(vid_path)
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            continue
            
        # Calculate interval to get FRAMES_PER_VIDEO evenly spaced
        interval = max(1, total_frames // FRAMES_PER_VIDEO)
        
        count = 0
        frame_idx = 0
        while count < FRAMES_PER_VIDEO:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame
            out_name = f"vid_frame_{label_prefix}_{vid_name}_{count}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, frame)
            
            count += 1
            frame_idx += interval
            
        cap.release()

if __name__ == "__main__":
    print("Starting frame extraction...")
    
    # Extract Real frames
    extract_frames(REAL_VID_DIR, REAL_OUT_DIR, "real")
    
    # Extract Fake frames
    extract_frames(FAKE_VID_DIR, FAKE_OUT_DIR, "fake")
    
    print("\nExtraction complete! Your training folders now have new frames from the videos.")
