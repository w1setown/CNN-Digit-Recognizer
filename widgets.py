import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from digit_preprocessing import preprocess_digit_image

class DrawingCanvas(tk.Canvas):
    """Canvas widget for drawing digits"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.width = kwargs.get('width', 280)
        self.height = kwargs.get('height', 280)
        self.image = Image.new("L", (self.width, self.height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.configure(bg="white", highlightthickness=2, bd=2, relief=tk.GROOVE)
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
        self.last_x = None  # Ensure drawing state is reset
        self.last_y = None  # Ensure drawing state is reset

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
        self.bars = self.ax.bar(self.digits, self.probabilities, color="#4a90e2")
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Digit', fontsize=13)
        self.ax.set_ylabel('Probability', fontsize=13)
        self.ax.set_title('Prediction Probabilities', fontsize=15)
        self.ax.set_xticks(self.digits)  # Show all digits 0-9
        self.ax.set_xticklabels([str(d) for d in self.digits])
        self.figure.tight_layout()

    def update_prediction(self, prediction_array):
        for bar, prob in zip(self.bars, prediction_array):
            bar.set_height(prob)
        self.ax.set_title(f"AI thinks it's a {np.argmax(prediction_array)} ({max(prediction_array)*100:.1f}%)", fontsize=15)
        self.canvas.draw()

class TrainingProgressDisplay(tk.Frame):
    """Display training progress with metrics"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.progress_frame = ttk.LabelFrame(self, text="Training Progress")
        self.progress_frame.pack(fill=tk.X, padx=5, pady=5)
        self.epoch_var = tk.StringVar(value="Epoch: 0/0")
        ttk.Label(self.progress_frame, textvariable=self.epoch_var, font=('Arial', 12)).pack(fill=tk.X, padx=5, pady=2)
        self.accuracy_var = tk.StringVar(value="Accuracy: 0.0%")
        ttk.Label(self.progress_frame, textvariable=self.accuracy_var, font=('Arial', 12)).pack(fill=tk.X, padx=5, pady=2)
        self.loss_var = tk.StringVar(value="Loss: 0.0")
        ttk.Label(self.progress_frame, textvariable=self.loss_var, font=('Arial', 12)).pack(fill=tk.X, padx=5, pady=2)
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

        self.stats_frame = tk.LabelFrame(self, text="Your Training Data", font=('Arial', 13, 'bold'))
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.stats_text = tk.Text(self.stats_frame, height=6, width=42, font=('Arial', 12))
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.config(state=tk.DISABLED)

        self.retrain_btn = ttk.Button(self, text="Teach the AI with my drawings",
                                      command=self.retrain_model,
                                      width=40, style='Big.TButton')
        self.retrain_btn.pack(fill=tk.X, pady=8)

        self.progress_display = TrainingProgressDisplay(self)
        self.progress_display.pack(fill=tk.X, pady=5)
        self.progress_display.pack_forget()

        self.update_stats()

    def add_to_training(self):
        drawing = self.parent.get_current_drawing()
        if drawing is None:
            messagebox.showwarning("No Drawing", "Please draw a digit in the box before adding it!")
            return
        digit = self.digit_var.get()
        processed_img, _ = preprocess_digit_image(drawing)
        processed_img = processed_img[0]  # Remove batch dimension for training set

        self.training_images.append(processed_img)
        self.training_labels.append(digit)
        self.update_stats()
        messagebox.showinfo("Thank you!", f"Your drawing as digit {digit} was added to help the AI learn!")
        self.parent.clear_canvas()

    def update_stats(self):
        stats = {}
        for label in self.training_labels:
            if label not in stats:
                stats[label] = 0
            stats[label] += 1
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "How many drawings you added for each digit:\n\n")
        for digit in range(10):
            count = stats.get(digit, 0)
            self.stats_text.insert(tk.END, f"Digit {digit}: {count} drawing{'s' if count != 1 else ''}\n")
        self.stats_text.insert(tk.END, f"\nTotal drawings: {len(self.training_images)}")
        self.stats_text.config(state=tk.DISABLED)

    def retrain_model(self):
        if len(self.training_images) == 0:
            messagebox.showwarning("No Data", "You haven't added any drawings yet!")
            return
        training_images_np = np.array(self.training_images)
        training_labels_np = np.array(self.training_labels)
        self.retrain_btn.config(state=tk.DISABLED)
        self.retrain_btn.config(text="Teaching in progress...")
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
        self.retrain_btn.config(text="Teach the AI with my drawings")
        if success:
            messagebox.showinfo("Success!", "Thank you! The AI has learned from your drawings.")
            self.prediction_callback()
        else:
            messagebox.showerror("Oops!", f"Something went wrong: {error_msg}")