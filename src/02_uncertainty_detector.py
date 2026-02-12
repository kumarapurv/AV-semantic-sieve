import os
import torch
import cv2
import numpy as np
from nuimages import NuImages
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# --- CONFIGURATION ---
DATA_ROOT = '/Users/kumarapurv/Projects/AV-semantic-sieve/data/nuimages' 
VERSION = 'v1.0-train'
# Thresholds for "Uncertainty": 
# We want cases where the model isn't sure, but also didn't completely miss it.
CONF_MIN = 0.20 
CONF_MAX = 0.45

def main():
    print("--- Initializing Uncertainty Sieve (Tier 1 Perception) ---")
    
    # 1. Load nuImages for source data
    nuim = NuImages(dataroot=DATA_ROOT, version=VERSION, verbose=False)
    
    # 2. Load YOLOv8 (nano version for speed on M2)
    # This will download the weights automatically on first run
    model = YOLO('yolov8n.pt') 
    
    # Move to MPS if available for M2 Pro acceleration
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    print(f"Using device: {device}")

    uplift_candidates = []

    # 3. Iterate through a small subset of samples to find uncertainty
    print(f"Scanning samples for high-entropy/uncertain detections...")
    
    # Let's check the first 50 samples for demonstration
    for i in range(50):
        sample = nuim.sample[i]
        key_camera_token = sample['key_camera_token']
        sample_data = nuim.get('sample_data', key_camera_token)
        img_path = os.path.join(DATA_ROOT, sample_data['filename'])
        
        # Run Inference
        results = model(img_path, verbose=False)[0]
        
        # Check for detections in our "Uncertainty Zone"
        uncertain_boxes = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if CONF_MIN < conf < CONF_MAX:
                uncertain_boxes.append(box)
        
        if uncertain_boxes:
            uplift_candidates.append({
                'path': img_path,
                'count': len(uncertain_boxes),
                'top_conf': float(uncertain_boxes[0].conf[0]),
                'boxes': results.boxes # keep all for plotting
            })
            if len(uplift_candidates) >= 3: # Just find a few for the demo
                break

    # 4. Visualize a candidate for the Portfolio
    if uplift_candidates:
        candidate = uplift_candidates[0]
        print(f"\n[Uplift Triggered] Found {candidate['count']} uncertain objects in {candidate['path']}")
        
        # Plotting the "Sieve" view
        img = cv2.imread(candidate['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(img)
        
        # Overlay detections
        for box in candidate['boxes']:
            conf = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy()
            
            # Color coding: Green = Confident, Red = Uncertain (Uplifted)
            color = 'red' if CONF_MIN < conf < CONF_MAX else 'lime'
            linewidth = 3 if color == 'red' else 1
            alpha = 1.0 if color == 'red' else 0.3
            
            plt.gca().add_patch(plt.Rectangle(
                (coords[0], coords[1]), coords[2]-coords[0], coords[3]-coords[1],
                fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha
            ))
            plt.text(coords[0], coords[1]-5, f"{conf:.2f}", color=color, fontweight='bold')

        plt.title("Smart Sieve: Tier 1 Uncertainty Detection\n(Red boxes = High Entropy / Uplifted for VLM Reasoning)")
        plt.axis('off')
        plt.show()
    else:
        print("No uncertainty candidates found in this subset. Try increasing the scan range.")

if __name__ == "__main__":
    main()