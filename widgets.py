import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2

class DrawingCanvas(tk.Canvas):
    """Canvas widget for drawing digits"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.width = kwargs.get('width', 280)
        self.height = kwargs.get('height', 280)
        self.image = Image.new("L", (self.width, self.height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.configure(bg="white", highlightthickness=0)
        self.bind("<B1-Motion>", self.paint)
        self.bind("<ButtonRelease-1>", self.reset_last_point)
        self.last_x = None
        self.last_y = None
        self.line_width = 12

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.create_line((self.last_x, self.last_y, x, y),
                             fill="black", width=self.line_width,
                             capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=0, width=self.line_width)
        self.last_x = x
        self.last_y = y

    def reset_last_point(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.delete("all")
        self.image = Image.new("L", (self.width, self.height), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def get_image(self):
        return np.array(self.image)

    def show_preprocessed(self, processed_img):
        display_size = (140, 140)
        img = Image.fromarray(processed_img)
        img = img.resize(display_size, Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.delete("preprocessed")
        self.create_image(self.width//2, self.height//2, image=img, tags="preprocessed")
        self.preprocessed_image = img  # Prevent garbage collection

class PredictionDisplay(tk.Frame):
    """Frame to display prediction results and confidence"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.digits = list(range(10))
        self.probabilities = [0] * 10
        self.bars = self.ax.bar(self.digits, self.probabilities)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Digit')
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Prediction Probabilities')
        self.figure.tight_layout()

    def update_prediction(self, prediction_array):
        for bar, prob in zip(self.bars, prediction_array):
            bar.set_height(prob)
        self.ax.set_title(f'Predicted: {np.argmax(prediction_array)} ({max(prediction_array)*100:.1f}%)')
        self.canvas.draw()

class TrainingProgressDisplay(tk.Frame):
    """Display training progress with metrics"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.progress_frame = ttk.LabelFrame(self, text="Training Progress")
        self.progress_frame.pack(fill=tk.X, padx=5, pady=5)
        self.epoch_var = tk.StringVar(value="Epoch: 0/0")
        ttk.Label(self.progress_frame, textvariable=self.epoch_var).pack(fill=tk.X, padx=5, pady=2)
        self.accuracy_var = tk.StringVar(value="Accuracy: 0.0%")
        ttk.Label(self.progress_frame, textvariable=self.accuracy_var).pack(fill=tk.X, padx=5, pady=2)
        self.loss_var = tk.StringVar(value="Loss: 0.0")
        ttk.Label(self.progress_frame, textvariable=self.loss_var).pack(fill=tk.X, padx=5, pady=2)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame,
                                            variable=self.progress_var,
                                            mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)

    def update_progress(self, epoch, max_epochs, accuracy, loss):
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
        self.digit_var = digit_var
        self.training_images = []
        self.training_labels = []

        self.stats_frame = tk.LabelFrame(self, text="Training Data Statistics")
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.stats_text = tk.Text(self.stats_frame, height=5, width=40)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.config(state=tk.DISABLED)

        self.retrain_btn = ttk.Button(self, text="Retrain Model",
                                      command=self.retrain_model,
                                      width=40, style='Big.TButton')
        self.retrain_btn.pack(fill=tk.X, pady=5)

        self.progress_display = TrainingProgressDisplay(self)
        self.progress_display.pack(fill=tk.X, pady=5)
        self.progress_display.pack_forget()

        self.update_stats()

    def add_to_training(self):
        drawing = self.parent.get_current_drawing()
        if drawing is None:
            messagebox.showwarning("Empty Drawing", "Please draw a digit first!")
            return
        digit = self.digit_var.get()
        
        # Use the same preprocessing as in predict_digit
        import cv2
        import numpy as np
        
        # Convert to binary using thresholding
        _, binary = cv2.threshold(drawing, 200, 255, cv2.THRESH_BINARY_INV)
        
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
            x2 = min(drawing.shape[1], x + w + padding)
            y2 = min(drawing.shape[0], y + h + padding)
            
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
        processed_img = processed_img.astype('float32') / 255.0
        processed_img = np.expand_dims(processed_img, axis=-1)
        
        self.training_images.append(processed_img)
        self.training_labels.append(digit)
        self.update_stats()
        messagebox.showinfo("Success", f"Added drawing as digit {digit} to training set")
        self.parent.clear_canvas()

    def update_stats(self):
        stats = {}
        for label in self.training_labels:
            if label not in stats:
                stats[label] = 0
            stats[label] += 1
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "Images per digit:\n")
        for digit in range(10):
            count = stats.get(digit, 0)
            self.stats_text.insert(tk.END, f"Digit {digit}: {count} images\n")
        self.stats_text.insert(tk.END, f"\nTotal training images: {len(self.training_images)}")
        self.stats_text.config(state=tk.DISABLED)

    def retrain_model(self):
        if len(self.training_images) == 0:
            messagebox.showwarning("No Data", "No training data available!")
            return
        training_images_np = np.array(self.training_images)
        training_labels_np = np.array(self.training_labels)
        self.retrain_btn.config(state=tk.DISABLED)
        self.retrain_btn.config(text="Retraining...")
        thread = threading.Thread(
            target=self._do_retraining,
            args=(training_images_np, training_labels_np)
        )
        thread.daemon = True
        thread.start()

    def _do_retraining(self, images, labels):
        try:
            self.progress_display.pack(fill=tk.X, pady=5)
            self.parent.ensemble.create_new_model(images, labels)
            self.parent.after(0, self._retraining_complete, True)
            self.training_images = []
            self.training_labels = []
            self.update_stats()
        except Exception as e:
            self.parent.after(0, self._retraining_complete, False, str(e))
        finally:
            self.parent.after(0, self.progress_display.pack_forget)

    def _retraining_complete(self, success, error_msg=None):
        self.retrain_btn.config(state=tk.NORMAL)
        self.retrain_btn.config(text="Retrain Model")
        if success:
            messagebox.showinfo("Success", "Model retrained successfully!")
            self.prediction_callback()
        else:
            messagebox.showerror("Error Retraining", f"Error: {error_msg}")