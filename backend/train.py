import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
import random

# 1. Configuration
DATASET_PATH = r"c:\Users\Delli5\OneDrive\deepfake detector\dataset\certificates\extract"
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"
SUBSET_SIZE = 500 
BATCH_SIZE = 2
EPOCHS = 4        
LEARNING_RATE = 3e-5
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fine_tuned_certificates_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. PIL-based Augmentation Function
def apply_augmentations(image):
    # Random Horizontal Flip
    if random.random() > 0.5:
        image = ImageOps.mirror(image)
    
    # Random Rotation (+/- 10 degrees)
    angle = random.uniform(-10, 10)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # Color Jittering
    if random.random() > 0.5:
        # Brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        
    return image

# 3. Custom Dataset Class
class CertificateDataset(Dataset):
    def __init__(self, root_dir, total_subset_size=1000, processor=None, augment=False):
        self.root_dir = root_dir
        self.processor = processor
        self.augment = augment
        self.image_paths = []
        self.labels = []
        
        all_dirs = os.listdir(root_dir)
        real_dirs = [d for d in all_dirs if not d.endswith("_forg") and os.path.isdir(os.path.join(root_dir, d))]
        forged_dirs = [d for d in all_dirs if d.endswith("_forg") and os.path.isdir(os.path.join(root_dir, d))]
        
        print(f"Found {len(real_dirs)} real subdirectories and {len(forged_dirs)} forged subdirectories.")
        
        num_per_class = total_subset_size // 2
        
        # Load images
        for label, dirs in [(0, real_dirs), (1, forged_dirs)]:
            count = 0
            for d in dirs:
                if count >= num_per_class: break
                dir_path = os.path.join(root_dir, d)
                images = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img in images:
                    if count >= num_per_class: break
                    self.image_paths.append(img)
                    self.labels.append(label)
                    count += 1
            print(f"Loaded {count} {'real' if label==0 else 'forged'} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            
            if self.augment:
                image = apply_augmentations(image)
                
            inputs = self.processor(image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0), torch.tensor(label)
        except Exception:
            return torch.zeros((3, 224, 224)), torch.tensor(self.labels[idx])

# 4. Initialization
print(f"Loading processor and model: {MODEL_NAME}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
model.to(device)

train_dataset = CertificateDataset(DATASET_PATH, total_subset_size=SUBSET_SIZE, processor=processor, augment=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5) 
criterion = nn.CrossEntropyLoss()

# 5. Training Loop
print("Starting Stable Training with PIL Augmentation...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    print(f"Starting Epoch {epoch+1}/{EPOCHS}")
    
    for i, (pixel_values, labels) in enumerate(train_loader):
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if i % 20 == 0:
            print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# 6. Save Model
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Saving robust model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Training complete!")
