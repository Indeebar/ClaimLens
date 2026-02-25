import os
import shutil
import random

def distribute_images():
    base_dir = r"f:\ClaimLens\data\raw\damage_images\data1a"
    dest_dir = r"f:\ClaimLens\data\raw\damage_images"
    
    # Create target directories
    for label in ['minor', 'moderate', 'severe']:
        os.makedirs(os.path.join(dest_dir, label), exist_ok=True)
    
    # Collect all 'whole' (minor) images
    whole_images = []
    for split in ['training', 'validation']:
        whole_dir = os.path.join(base_dir, split, '01-whole')
        if os.path.exists(whole_dir):
            for f in os.listdir(whole_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    whole_images.append(os.path.join(whole_dir, f))
    
    # Collect all 'damage' images
    damage_images = []
    for split in ['training', 'validation']:
        damage_dir = os.path.join(base_dir, split, '00-damage')
        if os.path.exists(damage_dir):
            for f in os.listdir(damage_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    damage_images.append(os.path.join(damage_dir, f))
                    
    print(f"Found {len(whole_images)} whole images and {len(damage_images)} damage images.")
    
    # Shuffle for random distribution
    random.shuffle(damage_images)
    
    # Split damage images into moderate and severe
    mid = len(damage_images) // 2
    moderate_images = damage_images[:mid]
    severe_images = damage_images[mid:]
    
    # Copy images to final destinations
    print("Copying minor...")
    for i, img in enumerate(whole_images):
        shutil.copy(img, os.path.join(dest_dir, 'minor', f'minor_{i}.jpg'))
        
    print("Copying moderate...")
    for i, img in enumerate(moderate_images):
        shutil.copy(img, os.path.join(dest_dir, 'moderate', f'moderate_{i}.jpg'))
        
    print("Copying severe...")
    for i, img in enumerate(severe_images):
        shutil.copy(img, os.path.join(dest_dir, 'severe', f'severe_{i}.jpg'))
        
    print("Done distributing images.")

if __name__ == "__main__":
    distribute_images()
