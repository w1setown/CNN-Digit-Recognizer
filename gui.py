import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
from PIL import Image, ImageTk

# Add project directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from model_ensemble import ModelEnsemble
from widgets import DrawingCanvas, PredictionDisplay, TrainingPanel
from digit_preprocessing import preprocess_digit_image  # NEW IMPORT

class DigitRecognitionApp(tk.Tk):
    """Main application window for digit recognition"""
    def __init__(self):
        super().__init__()
        self.title("Digit Recognition")
        self.geometry("800x600")
        self.resizable(True, True)

        # Style
        style = ttk.Style()
        style.configure('Big.TButton', padding=(10, 10))
        style.configure('TCombobox', padding=(5, 5))

        # Main frames
        self.left_frame = tk.Frame(self)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Preview frame for preprocessed image
        self.preview_frame = tk.LabelFrame(self.left_frame, text="Preprocessed Digit")
        self.preview_frame.pack(fill=tk.X, padx=10, pady=5)
        self.preview_canvas = tk.Canvas(self.preview_frame, width=140, height=140, bg="white", highlightthickness=1)
        self.preview_canvas.pack(padx=10, pady=5)

        # Drawing canvas
        self.canvas_frame = tk.LabelFrame(self.left_frame, text="Draw a digit")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas = DrawingCanvas(self.canvas_frame, width=280, height=280)
        self.canvas.pack(padx=10, pady=10)

        # Canvas controls
        self.canvas_controls = tk.Frame(self.left_frame)
        self.canvas_controls.pack(fill=tk.X, padx=10, pady=5)
        self.digit_var = tk.IntVar(value=0)
        digit_frame = tk.Frame(self.canvas_controls)
        digit_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(digit_frame, text="Current Digit:", font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT, padx=25)
        self.digit_selector = ttk.Combobox(
            digit_frame, textvariable=self.digit_var, values=list(range(10)),
            width=10, font=('TkDefaultFont', 10, 'bold'), state="readonly"
        )
        self.digit_selector.pack(side=tk.LEFT)
        self.digit_selector.current(0)

        self.clear_btn = ttk.Button(self.canvas_controls, text="Clear", command=self.clear_canvas, width=10, style='Big.TButton')
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        self.predict_btn = ttk.Button(self.canvas_controls, text="Predict", command=self.predict_digit, width=10, style='Big.TButton')
        self.predict_btn.pack(side=tk.LEFT, padx=5)

        # Training controls
        self.training_controls = tk.Frame(self.left_frame)
        self.training_controls.pack(fill=tk.X, padx=10, pady=5)
        self.add_btn = ttk.Button(
            self.training_controls, text="Add Current Drawing to Training Set",
            command=lambda: self.training_panel.add_to_training(),
            width=40, style='Big.TButton'
        )
        self.add_btn.pack(fill=tk.X, pady=5)

        # Prediction display
        self.prediction_frame = tk.LabelFrame(self.right_frame, text="Prediction")
        self.prediction_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.prediction_display = PredictionDisplay(self.prediction_frame)
        self.prediction_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Training panel
        self.training_frame = tk.LabelFrame(self.right_frame, text="Training")
        self.training_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.training_panel = TrainingPanel(self, self.predict_digit, self.digit_var)
        self.training_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Keyboard shortcuts
        self.bind("<Control-z>", lambda event: self.clear_canvas())
        self.bind("<Control-x>", lambda event: self.predict_digit())
        self.bind("<Control-v>", lambda event: self.training_panel.add_to_training())
        self.bind("<Control-n>", lambda event: self.training_panel.retrain_model())

        # Initialize
        self.ensemble = None
        self.initialize()

    def initialize(self):
        """Initialize the application and load the model ensemble"""
        self.status_var.set("Loading model...")
        thread = threading.Thread(target=self._load_model)
        thread.daemon = True
        thread.start()

    def _load_model(self):
        """Load the model ensemble in a background thread"""
        try:
            self.ensemble = ModelEnsemble()
            self.after(0, lambda: self.status_var.set("Model ensemble loaded successfully"))
        except Exception as e:
            self.after(0, lambda: self.status_var.set(f"Error loading models: {e}"))

    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.clear()

    def get_current_drawing(self):
        """Get the current drawing as a numpy array, or None if empty"""
        img = self.canvas.get_image()
        if np.max(img) == 0:
            return None
        return img

    def predict_digit(self):
        """Predict the drawn digit using the model ensemble"""
        img = self.get_current_drawing()
        if img is None:
            messagebox.showwarning("Empty Canvas", "Please draw a digit first!")
            return
        if self.ensemble is None:
            messagebox.showwarning("Models Not Ready", "Models are still loading, please wait")
            return

        # Use utility function for preprocessing
        processed_img, preview_img = preprocess_digit_image(img, preview_size=(140, 140))

        # Show preprocessed image in preview
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(70, 70, image=preview_img)
        self.preview_canvas.image = preview_img  # Prevent garbage collection

        # Predict
        prediction = self.ensemble.predict(processed_img)
        self.prediction_display.update_prediction(prediction)

        # Update status bar
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        model_counts = self.ensemble.get_model_counts()
        total_models = model_counts['mnist'] + model_counts['emnist']
        self.status_var.set(
            f"Predicted digit: {predicted_digit} (confidence: {confidence:.2f}%) "
            f"using {total_models} model{'s' if total_models != 1 else ''}"
        )

def main():
    app = DigitRecognitionApp()
    app.mainloop()

if __name__ == "__main__":
    main()