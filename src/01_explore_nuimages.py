import os
from nuimages import NuImages
import matplotlib.pyplot as plt
from PIL import Image

# --- CONFIGURATION ---
DATA_ROOT = '/Users/kumarapurv/Projects/AV-semantic-sieve/data/nuimages' 
VERSION = 'v1.0-train' # Start with the training set metadata

def main():
    print(f"--- Initializing nuImages DevKit from {DATA_ROOT} ---")
    
    try:
        # 1. Initialize the dataset
        # This loads the JSON metadata into memory
        nuim = NuImages(dataroot=DATA_ROOT, version=VERSION, verbose=True, lazy=False)
        
        # 2. Print Dataset Statistics
        print(f"\n[Stats] Total Images: {len(nuim.sample)}")
        print(f"[Stats] Total Categories: {len(nuim.category)}")
        print(f"[Stats] Total Attributes: {len(nuim.attribute)}")

        # 3. Identify "Long-Tail" Attributes
        # In the VW ADMT project, these are our "Hard Triggers"
        interesting_attributes = [
            'vehicle.emergency.ambulance',
            'vehicle.emergency.police',
            'pedestrian.stroller',
            'pedestrian.wheelchair',
            'cycle.with_rider'
        ]
        
        print("\n--- Searching for 'Long-Tail' Candidates ---")
        
        found_sample = None
        target_attr_name = ""

        # Search for a sample containing one of our target attributes
        for attr in nuim.attribute:
            if attr['name'] in interesting_attributes:
                # Find annotations that have this attribute
                ann_tokens = [ann['token'] for ann in nuim.object_ann if attr['token'] in ann['attribute_tokens']]
                
                if ann_tokens:
                    # Get the first sample associated with this annotation
                    sample_token = nuim.get('object_ann', ann_tokens[0])['sample_data_token']
                    found_sample = nuim.get('sample_data', sample_token)
                    target_attr_name = attr['name']
                    print(f"[Found] Found a match for '{target_attr_name}' in image: {found_sample['filename']}")
                    break

        if found_sample:
            # 4. Visualize the "High-Value" Frame
            img_path = os.path.join(DATA_ROOT, found_sample['filename'])
            image = Image.open(img_path)
            
            plt.figure(figsize=(12, 7))
            plt.imshow(image)
            plt.title(f"High-Value Uplift Candidate\nDetected Attribute: {target_attr_name}")
            plt.axis('off')
            plt.show()
            
            print("\n[Success] Visualization complete. This image would be a 'Smart Sieve' trigger.")
        else:
            print("[Note] No specific long-tail attributes found in the current metadata subset.")

    except Exception as e:
        print(f"\n[Error] {str(e)}")
        print("\nCheck if your DATA_ROOT path is correct and contains the 'v1.0-train' folder.")

if __name__ == "__main__":
    main()