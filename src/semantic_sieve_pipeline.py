import os
import torch
import cv2
import sys
import matplotlib.pyplot as plt
from PIL import Image
from nuimages import NuImages
from ultralytics import YOLO
from mlx_vlm import load, generate

# --- CONFIGURATION ---
DATA_ROOT = './data/nuimages'
VERSION = 'v1.0-train'
MODEL_PATH = "mlx-community/Phi-3.5-vision-instruct-4bit"
DETECTOR_MODEL = 'yolov8n.pt'

# Thresholds for "Uncertainty Sieve"
CONF_MIN = 0.20
CONF_MAX = 0.45

def initialize_models():
    """Initializes both the Primary Sieve (YOLO) and the Semantic Guardian (VLM)."""
    print("--- Initializing AI Pipeline ---")
    
    try:
        detector = YOLO(DETECTOR_MODEL)
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        detector.to(device)
        print(f"[Detector] YOLOv8 initialized on {device}")
    except Exception as e:
        print(f"[Error] Failed to load YOLO: {e}")
        sys.exit(1)

    print(f"[Guardian] Loading VLM: {MODEL_PATH}...")
    try:
        model, processor = load(MODEL_PATH, trust_remote_code=True)
        print("[Guardian] VLM initialized on Apple Silicon")
        return detector, model, processor
    except Exception as e:
        print(f"[Critical Error] VLM Load Failed: {e}")
        sys.exit(1)

def find_uplift_candidate(nuim, detector, scan_limit=50):
    """Scans nuImages to find a frame that triggers the uncertainty sieve."""
    print(f"--- Scanning {scan_limit} frames for Uncertainty Triggers ---")
    
    for i in range(scan_limit):
        sample = nuim.sample[i]
        key_camera_token = sample['key_camera_token']
        sample_data = nuim.get('sample_data', key_camera_token)
        img_path = os.path.join(DATA_ROOT, sample_data['filename'])
        
        if not os.path.exists(img_path):
            continue

        results = detector(img_path, verbose=False)[0]
        uncertain_boxes = [box for box in results.boxes if CONF_MIN < float(box.conf[0]) < CONF_MAX]
        
        if uncertain_boxes:
            print(f"[Trigger] Found candidate: {img_path} ({len(uncertain_boxes)} uncertain objects)")
            return img_path, results
            
    return None, None

def create_visual_evidence(image_path, results, output_path):
    """Generates an annotated image showing confident vs uncertain detections."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated_img = img.copy()

    for box in results.boxes:
        conf = float(box.conf[0])
        coords = box.xyxy[0].cpu().numpy().astype(int)
        
        # Red = Uncertain (Trigger), Green = Confident (Nominal)
        is_uncertain = CONF_MIN < conf < CONF_MAX
        color = (255, 0, 0) if is_uncertain else (0, 255, 0) # RGB
        thickness = 4 if is_uncertain else 1
        
        cv2.rectangle(annotated_img, (coords[0], coords[1]), (coords[2], coords[3]), color, thickness)
        label = f"{conf:.2f} {'[UPLIFT]' if is_uncertain else ''}"
        cv2.putText(annotated_img, label, (coords[0], coords[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save the annotated version
    final_pil = Image.fromarray(annotated_img)
    final_pil.save(output_path)
    return annotated_img

def generate_semantic_audit(model, processor, image_path):
    """Runs the Semantic Guardian audit on the selected frame."""
    print("--- Running Semantic Audit ---")
    pil_image = Image.open(image_path).convert("RGB")

    prompt = (
        "<|user|>\n"
        "<|image_1|>\n"
        "You are an Autonomous Driving Safety Auditor. Analyze this scene. "
        "Identify any rare traffic participants, complex social interactions, or "
        "ambiguous objects that a standard perception stack might struggle to classify. "
        "Explain why this frame should be 'uplifted' for human labeling.\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

    try:
        result = generate(
            model=model,
            processor=processor,
            image=pil_image,
            prompt=prompt,
            max_tokens=400,
            temperature=0.2
        )
        return result
    except Exception as e:
        print(f"VLM generation failed: {str(e)}")
        return None

def main():
    if not os.path.exists(DATA_ROOT):
        print(f"[Error] DATA_ROOT not found: {DATA_ROOT}")
        return

    nuim = NuImages(dataroot=DATA_ROOT, version=VERSION, verbose=False)
    detector, vlm_model, processor = initialize_models()

    img_path, results = find_uplift_candidate(nuim, detector)

    if img_path:
        # 1. Create Visual Evidence first
        output_dir = "./portfolio_results"
        os.makedirs(output_dir, exist_ok=True)
        filename_base = os.path.basename(img_path).split('.')[0]
        viz_path = os.path.join(output_dir, f"{filename_base}_sieve_analysis.jpg")
        
        annotated_img = create_visual_evidence(img_path, results, viz_path)
        print(f"[Visual] Sieve analysis saved to {viz_path}")

        # 2. Run Reasoning
        generation_result = generate_semantic_audit(vlm_model, processor, img_path)
        
        if generation_result:
            report_text = generation_result.text
            report_file = os.path.join(output_dir, f"{filename_base}_audit.txt")
            
            with open(report_file, "w") as f:
                f.write(f"IMAGE: {img_path}\nREASONING:\n{report_text}")
                
            print(f"\n[Success] Portfolio assets generated in {output_dir}")
            
            # 3. Final Showdown: Display the visual and the text together
            plt.figure(figsize=(12, 8))
            plt.imshow(annotated_img)
            plt.title("Smart Sieve Analysis (Red = High Uncertainty Uplift)")
            plt.axis('off')
            plt.figtext(0.5, 0.01, f"GUARDIAN VERDICT: {report_text[:200]}...", 
                        wrap=True, horizontalalignment='center', fontsize=10, 
                        bbox={'facecolor':'orange', 'alpha':0.2, 'pad':10})
            plt.show()
        else:
            print("[Error] Semantic audit failed.")
    else:
        print("[Abort] No uncertainty candidates found. Try increasing scan_limit.")

if __name__ == "__main__":
    main()