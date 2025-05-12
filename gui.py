import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Add this import at the top
from model_ensemble import ModelEnsemble

# Import from other modules
from model import load_or_train_model
from data_utils import preprocess_image
from training import retrain_model

class DrawingCanvas(tk.Canvas):
    """Canvas widget for drawing digits"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.width = kwargs.get('width', 280)
        self.height = kwargs.get('height', 280)
        
        # Create PIL image for drawing with white background
        self.image = Image.new("L", (self.width, self.height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Setup canvas with white background
        self.configure(bg="white", highlightthickness=0)
        self.bind("<B1-Motion>", self.paint)
        self.bind("<ButtonRelease-1>", self.reset_last_point)
        
        self.last_x = None
        self.last_y = None
        self.line_width = 12
        
    def paint(self, event):
        """Draw on canvas when mouse is dragged"""
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # Draw black lines on white background
            self.create_line((self.last_x, self.last_y, x, y), 
                             fill="black", width=self.line_width, 
                             capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], 
                          fill=0, width=self.line_width)
        
        self.last_x = x
        self.last_y = y
    
    def reset_last_point(self, event):
        """Reset last point to avoid lines between separate strokes"""
        self.last_x = None
        self.last_y = None
    
    def clear(self):
        """Clear the canvas"""
        self.delete("all")
        self.image = Image.new("L", (self.width, self.height), color=255)
        self.draw = ImageDraw.Draw(self.image)
    
    def get_image(self):
        """Return the current image as a numpy array"""
        return np.array(self.image)

    def show_preprocessed(self, processed_img):
        """Display the preprocessed image"""
        # Convert numpy array to PIL Image
        display_size = (140, 140)  # Half the size of drawing canvas
        img = Image.fromarray(processed_img)
        img = img.resize(display_size, Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        
        # Update canvas
        self.delete("preprocessed")
        self.create_image(self.width//2, self.height//2, image=img, tags="preprocessed")
        self.preprocessed_image = img  # Keep reference to prevent garbage collection

class PredictionDisplay(tk.Frame):
    """Frame to display prediction results and confidence"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create figure for bar chart
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty chart
        self.digits = list(range(10))
        self.probabilities = [0] * 10
        self.bars = self.ax.bar(self.digits, self.probabilities)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Digit')
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Prediction Probabilities')
        self.figure.tight_layout()
        
    def update_prediction(self, prediction_array):
        """Update chart with new prediction probabilities"""
        for bar, prob in zip(self.bars, prediction_array):
            bar.set_height(prob)
        
        self.ax.set_title(f'Predicted: {np.argmax(prediction_array)} ({max(prediction_array)*100:.1f}%)')
        self.canvas.draw()

class TrainingProgressDisplay(tk.Frame):
    """Display training progress with metrics"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create progress indicators
        self.progress_frame = ttk.LabelFrame(self, text="Training Progress")
        self.progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Epoch counter
        self.epoch_var = tk.StringVar(value="Epoch: 0/0")
        ttk.Label(self.progress_frame, textvariable=self.epoch_var).pack(fill=tk.X, padx=5, pady=2)
        
        # Accuracy
        self.accuracy_var = tk.StringVar(value="Accuracy: 0.0%")
        ttk.Label(self.progress_frame, textvariable=self.accuracy_var).pack(fill=tk.X, padx=5, pady=2)
        
        # Loss
        self.loss_var = tk.StringVar(value="Loss: 0.0")
        ttk.Label(self.progress_frame, textvariable=self.loss_var).pack(fill=tk.X, padx=5, pady=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, 
                                          variable=self.progress_var,
                                          mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)
        
    def update_progress(self, epoch, max_epochs, accuracy, loss):
        """Update the progress display with new metrics"""
        self.epoch_var.set(f"Epoch: {epoch}/{max_epochs}")
        self.accuracy_var.set(f"Accuracy: {accuracy*100:.2f}%")
        self.loss_var.set(f"Loss: {loss:.4f}")
        self.progress_var.set((epoch / max_epochs) * 100)

class TrainingPanel(tk.Frame):
    """Panel for retraining the model"""
    def __init__(self, parent, prediction_callback, digit_var, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.prediction_callback = prediction_callback
        self.digit_var = digit_var  # Store the digit_var reference
        self.training_images = []
        self.training_labels = []
        
        # Training stats
        self.stats_frame = tk.LabelFrame(self, text="Training Data Statistics")
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.stats_text = tk.Text(self.stats_frame, height=5, width=40)  # Adjust width + height
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.config(state=tk.DISABLED)
        
        # Add retrain button to the panel
        self.retrain_btn = ttk.Button(self, text="Retrain Model",
                                     command=self.retrain_model,
                                     width=40, style='Big.TButton')
        self.retrain_btn.pack(fill=tk.X, pady=5)
        
        # Training progress display
        self.progress_display = TrainingProgressDisplay(self)
        self.progress_display.pack(fill=tk.X, pady=5)
        self.progress_display.pack_forget()  # Hide initially
        
        # Update stats
        self.update_stats()
    
    def add_to_training(self):
        """Add current drawing to training set"""
        drawing = self.parent.get_current_drawing()
        if drawing is None:
            messagebox.showwarning("Empty Drawing", "Please draw a digit first!")
            return
        
        # Preprocess the image for training
        digit = self.digit_var.get()
        
        # Process the image to 28x28
        img = cv2.resize(drawing, (28, 28))
        
        # Normalize and add channel dimension
        img = img.astype('float32') / 255
        img = np.expand_dims(img, axis=-1)
        
        # Add to training set
        self.training_images.append(img)
        self.training_labels.append(digit)
        
        # Update stats
        self.update_stats()
        messagebox.showinfo("Success", f"Added drawing as digit {digit} to training set")
        
        # Clear canvas for next drawing
        self.parent.clear_canvas()
    
    def update_stats(self):
        """Update training statistics display"""
        # Count examples per digit
        stats = {}
        for label in self.training_labels:
            if label not in stats:
                stats[label] = 0
            stats[label] += 1
        
        # Display stats
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        self.stats_text.insert(tk.END, "Images per digit:\n")
        for digit in range(10):
            count = stats.get(digit, 0)
            self.stats_text.insert(tk.END, f"Digit {digit}: {count} images\n")
        
        self.stats_text.insert(tk.END, f"\nTotal training images: {len(self.training_images)}")
        self.stats_text.config(state=tk.DISABLED)
    
    def retrain_model(self):
        """Retrain the model with collected examples"""
        if len(self.training_images) == 0:
            messagebox.showwarning("No Data", "No training data available!")
            return
        
        # Convert lists to numpy arrays
        training_images_np = np.array(self.training_images)
        training_labels_np = np.array(self.training_labels)
        
        # Start retraining in a separate thread
        self.retrain_btn.config(state=tk.DISABLED)
        self.retrain_btn.config(text="Retraining...")
        
        thread = threading.Thread(
            target=self._do_retraining,
            args=(training_images_np, training_labels_np)
        )
        thread.daemon = True
        thread.start()
    
    # In TrainingPanel class, modify _do_retraining method
    def _do_retraining(self, images, labels):
        """Create new model with training data"""
        try:
            # Show progress display
            self.progress_display.pack(fill=tk.X, pady=5)
            
            # Create new model in ensemble
            self.parent.ensemble.create_new_model(images, labels)
            
            # Update UI on main thread
            self.parent.after(0, self._retraining_complete, True)
            
            # Clear training data after successful model creation
            self.training_images = []
            self.training_labels = []
            self.update_stats()
            
        except Exception as e:
            # Update UI with error
            self.parent.after(0, self._retraining_complete, False, str(e))
        finally:
            # Hide progress display
            self.parent.after(0, self.progress_display.pack_forget)
    
    def _retraining_complete(self, success, error_msg=None):
        """Handle completion of retraining"""
        self.retrain_btn.config(state=tk.NORMAL)
        self.retrain_btn.config(text="Retrain Model")
        
        if success:
            messagebox.showinfo("Success", "Model retrained successfully!")
            # Update prediction for current drawing
            self.prediction_callback()
        else:
            messagebox.showerror("Error Retraining", f"Error: {error_msg}")


class DigitRecognitionApp(tk.Tk):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Digit Recognition")
        self.geometry("800x600")
        self.resizable(True, True)
        
        # Bind key combinations to actions
        self.bind("<Control-z>", lambda event: self.clear_canvas()) # Clear canvas
        self.bind("<Control-x>", lambda event: self.predict_digit()) # Predict digit
        self.bind("<Control-v>", lambda event: self.training_panel.add_to_training()) # Add to training set
        self.bind("<Control-n>", lambda event: self.training_panel.retrain_model()) # Retrain model
        
        # Load model
        self.model = None
        
        style = ttk.Style()
        style.configure('Big.TButton', padding=(10, 10))  # Increase padding make button bigger
        style.configure('TCombobox', padding=(5, 5))
        
        # Create main frames
        self.left_frame = tk.Frame(self)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Preview frame for preprocessed image
        self.preview_frame = tk.LabelFrame(self.left_frame, text="Preprocessed Digit")
        self.preview_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.preview_canvas = tk.Canvas(self.preview_frame, width=140, height=140,
                                      bg="white", highlightthickness=1)
        self.preview_canvas.pack(padx=10, pady=5)
        
        # Drawing canvas
        self.canvas_frame = tk.LabelFrame(self.left_frame, text="Draw a digit")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = DrawingCanvas(self.canvas_frame, width=280, height=280, 
                                  bg="black", highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        
        # Canvas controls
        self.canvas_controls = tk.Frame(self.left_frame)
        self.canvas_controls.pack(fill=tk.X, padx=10, pady=5)
        
        # Digit selector
        self.digit_var = tk.IntVar(value=0)
        digit_frame = tk.Frame(self.canvas_controls)
        digit_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(digit_frame, text="Current Digit:", font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT, padx=25)
        
        self.digit_selector = ttk.Combobox(digit_frame, 
                                          textvariable=self.digit_var,
                                          values=list(range(10)),
                                          width=10,
                                          font=('TkDefaultFont', 10, 'bold'),
                                          state="readonly")
        self.digit_selector.pack(side=tk.LEFT)
        self.digit_selector.current(0)
        
        # Clear and Predict buttons
        self.clear_btn = ttk.Button(self.canvas_controls, text="Clear", 
                                    command=self.clear_canvas,
                                    width=10, style='Big.TButton')
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = ttk.Button(self.canvas_controls, text="Predict", 
                                      command=self.predict_digit,
                                      width=10, style='Big.TButton')
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Training controls under canvas
        self.training_controls = tk.Frame(self.left_frame)
        self.training_controls.pack(fill=tk.X, padx=10, pady=5)
        
        # Add to training set button
        self.add_btn = ttk.Button(self.training_controls, text="Add Current Drawing to Training Set", 
                                 command=lambda: self.training_panel.add_to_training(),
                                 width=40, style='Big.TButton')
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
        self.status_bar = tk.Label(self, textvariable=self.status_var, 
                                 bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize
        self.initialize()
    
    def initialize(self):
        """Initialize the application"""
        self.status_var.set("Loading model...")
        
        # Load model in a separate thread
        thread = threading.Thread(target=self._load_model)
        thread.daemon = True
        thread.start()
    
    # Modify _load_model method
    def _load_model(self):
        """Load the model ensemble in a background thread"""
        try:
            self.ensemble = ModelEnsemble()
            self.after(0, lambda: self.status_var.set("Model ensemble loaded successfully"))
        except Exception as e:
            self.after(0, lambda e=e: self.status_var.set(f"Error loading models: {e}"))
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.clear()
    
    def get_current_drawing(self):
        """Get the current drawing as a numpy array"""
        img = self.canvas.get_image()
        if np.max(img) == 0:  # Check if canvas is empty
            return None
        return img
    
    # Modify predict_digit method
    def predict_digit(self):
        """Predict the drawn digit using model ensemble"""
        img = self.get_current_drawing()
        if img is None:
            messagebox.showwarning("Empty Canvas", "Please draw a digit first!")
            return
        
        if self.ensemble is None:
            messagebox.showwarning("Models Not Ready", "Models are still loading, please wait")
            return
        
        # Enhanced preprocessing
        processed_img = cv2.resize(img, (28, 28))
        # Add thresholding to make digits more distinct
        _, processed_img = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY)
        # Invert the image to match MNIST format (white digits on black background)
        processed_img = cv2.bitwise_not(processed_img)
        
        # Show preprocessed image
        self.preview_canvas.delete("all")
        preview_img = Image.fromarray(processed_img)
        preview_img = preview_img.resize((140, 140), Image.Resampling.LANCZOS)
        preview_img = ImageTk.PhotoImage(preview_img)
        self.preview_canvas.create_image(70, 70, image=preview_img)
        self.preview_canvas.image = preview_img  # Keep reference
        
        # Prepare for prediction
        processed_img = processed_img.astype('float32') / 255
        processed_img = np.expand_dims(processed_img, axis=-1)
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # Make prediction using ensemble
        prediction = self.ensemble.predict(processed_img)
        self.prediction_display.update_prediction(prediction)
        
        # Update status bar with number of models used
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