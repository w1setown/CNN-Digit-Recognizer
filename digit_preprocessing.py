import numpy as np
import cv2
from PIL import Image, ImageTk

def preprocess_digit_image(img, preview_size=(140, 140)):
    """
    Preprocess a digit image for model prediction/training and GUI preview.
    Returns (processed_img, preview_img)
    - processed_img: shape (1, 28, 28, 1), float32, normalized
    - preview_img: ImageTk.PhotoImage for display
    """
    # Convert to binary using thresholding
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        padding = max(w, h) // 4
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        digit_roi = binary[y1:y2, x1:x2]
        square_size = max(digit_roi.shape[0], digit_roi.shape[1])
        squared_img = np.zeros((square_size, square_size), dtype=np.uint8)
        offset_x = (square_size - digit_roi.shape[1]) // 2
        offset_y = (square_size - digit_roi.shape[0]) // 2
        squared_img[offset_y:offset_y+digit_roi.shape[0], offset_x:offset_x+digit_roi.shape[1]] = digit_roi
        squared_img = cv2.resize(squared_img, (20, 20), interpolation=cv2.INTER_AREA)
        processed_img = np.zeros((28, 28), dtype=np.uint8)
        processed_img[4:24, 4:24] = squared_img
    else:
        processed_img = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
    # For model
    norm_img = processed_img.astype('float32') / 255.0
    norm_img = np.expand_dims(norm_img, axis=-1)
    norm_img = np.expand_dims(norm_img, axis=0)
    # For preview
    pil_img = Image.fromarray(processed_img)
    pil_img = pil_img.resize(preview_size)
    preview_img = ImageTk.PhotoImage(pil_img)
    return norm_img, preview_img
