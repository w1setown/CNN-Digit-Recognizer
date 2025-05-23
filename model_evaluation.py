import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model_ensemble import ModelEnsemble

def evaluate_test_images(folder_path="test_digits"):
    """Evaluate model performance on a folder of test images"""
    # Load ensemble
    print("Loading model ensemble...")
    ensemble = ModelEnsemble()
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found. Creating it...")
        os.makedirs(folder_path)
        print(f"Please add test digit images to {folder_path} folder and run again.")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {folder_path}. Please add some test images.")
        return
    
    print(f"Found {len(image_files)} test images.")
    
    # Create figure for displaying results
    num_images = len(image_files)
    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=(cols*3, rows*3))
    
    for i, img_file in enumerate(image_files):
        # Load and preprocess image
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error loading image: {img_file}")
            continue
        
        # Convert to binary using thresholding
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours to detect the digit
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find bounding rectangle of all contours combined
            all_points = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            
            # Add padding to maintain aspect ratio
            padding = max(w, h) // 4
            
            # Calculate new boundaries with padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            
            # Extract the digit with padding
            digit_roi = binary[y1:y2, x1:x2]
            
            # Create a square image with the digit centered
            square_size = max(digit_roi.shape[0], digit_roi.shape[1])
            squared_img = np.zeros((square_size, square_size), dtype=np.uint8)
            
            # Center the digit in the square
            offset_x = (square_size - digit_roi.shape[1]) // 2
            offset_y = (square_size - digit_roi.shape[0]) // 2
            squared_img[offset_y:offset_y+digit_roi.shape[0], offset_x:offset_x+digit_roi.shape[1]] = digit_roi
            
            # Resize to 20x20 preserving aspect ratio (MNIST standard)
            squared_img = cv2.resize(squared_img, (20, 20), interpolation=cv2.INTER_AREA)
            
            # Add 4 pixels of padding around the digit (MNIST standard 28x28)
            processed_img = np.zeros((28, 28), dtype=np.uint8)
            processed_img[4:24, 4:24] = squared_img
        else:
            # If no contours found, just resize the image
            processed_img = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values
        norm_img = processed_img.astype('float32') / 255.0
        input_img = np.expand_dims(norm_img, axis=0)
        input_img = np.expand_dims(input_img, axis=-1)
        
        # Make prediction
        prediction = ensemble.predict(input_img)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Display results
        plt.subplot(rows, cols, i+1)
        plt.imshow(processed_img, cmap='gray')
        plt.title(f"Pred: {predicted_digit} ({confidence:.1f}%)")
        plt.axis('off')
        
        print(f"Image {img_file}: Predicted {predicted_digit} with {confidence:.2f}% confidence")
    
    plt.tight_layout()
    plt.savefig("prediction_results.png")
    plt.show()
    print(f"Results visualization saved to prediction_results.png")

if __name__ == "__main__":
    evaluate_test_images()