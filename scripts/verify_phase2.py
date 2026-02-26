import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.damage_classifier.predict import load_model, predict_damage

def verify():
    print("Loading model...")
    load_model('models/damage_classifier/best_model.pt')
    
    # Use one of the minor images for testing
    test_img = 'data/raw/damage_images/minor/minor_0.jpg'
    
    if not os.path.exists(test_img):
        print(f"Error: Could not find test image at {test_img}")
        return
        
    print(f"Running prediction on {test_img}...")
    result = predict_damage(test_img)
    print("\nPrediction Result:")
    print(result)
    print("\nPhase 2 Verification Complete.")

if __name__ == '__main__':
    verify()
