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


class ToolTip:
    """Simple tooltip for widgets"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, _, cy = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0,0,0,0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("TkDefaultFont", 10))
        label.pack(ipadx=6, ipady=2)

    def hide_tip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

    def set_text(self, text):
        self.text = text

class DigitRecognitionApp(tk.Tk):
    """Main application window for digit recognition"""
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognition   ")
        
        icon_path = os.path.join(script_dir, "..", "assets", "logo.png") 
        if os.path.exists(icon_path):
            icon_img = Image.open(icon_path)
            self.iconphoto(False, ImageTk.PhotoImage(icon_img))
            
        self.geometry("900x650+0+0") 
        self.resizable(True, True)
        self.configure(bg="#f0f4ff")

        # Style
        style = ttk.Style()
        style.configure('Big.TButton', padding=(12, 12), font=('Arial', 13, 'bold'))
        style.configure('TCombobox', padding=(5, 5), font=('Arial', 12))
        style.configure('TLabel', font=('Arial', 12))
        style.configure('TLabelframe.Label', font=('Arial', 13, 'bold'))
        style.configure('TFrame', background="#f0f4ff")

        # Language state
        self.language = "en"  # "en" or "da"
        self.texts = {
            "en": {
                "title": "Welcome to the Handwritten Digit Recognizer!",
                "subtitle": "Draw a digit below and let the AI guess what it is.",
                "instructions_title": "How to use:",
                "instructions": [
                    "1. Use your mouse or tablet to draw a single digit (0-9) in the box below.",
                    "2. Click the 'Guess my digit!' button to see the AI's prediction.",
                    "3. If the AI guesses wrong, you can help improve the model by adding your attempt."
                ],
                "preview": "What the AI sees",
                "draw_here": "Draw your digit here",
                "canvas_tip": "Tip: Click here and start drawing! Press Enter or click 'Guess my digit!'.",
                "enter_number": "Enter the number you have drawn:",
                "clear_btn": "Clear Drawing (Ctrl+Z)",
                "clear_tip": "Erase your drawing and start over (Ctrl+Z)",
                "guess_btn": "Guess my digit!",
                "guess_tip": "Let the AI guess your digit (Enter or Ctrl+X)",
                "add_btn": "Add my drawing to help the AI learn (Ctrl+V)",
                "add_tip": "If the AI guessed wrong, add your drawing to improve it! (Ctrl+V)",
                "ai_guess": "AI's Guess",
                "teach_ai": "Teach the AI",
                "empty_canvas": "Please draw a digit first!",
                "models_not_ready": "Models are still loading, please wait",
                "predicted": "Predicted digit: {digit} (confidence: {conf:.2f}%) using {n} model{s}",
            },
            "da": {
                "title": "Velkommen til Håndskrevet Talgenkendelse!",
                "subtitle": "Tegn et tal nedenfor og lad AI gætte hvilket det er.",
                "instructions_title": "Sådan gør du:",
                "instructions": [
                    "1. Brug musen eller tablet til at tegne et tal (0-9) i boksen nedenfor.",
                    "2. Tryk på knappen 'Guess my digit!' for at se AI'ens gæt.",
                    "3. Hvis AI'en gætter forkert, kan du hjælpe med at forbedre modellen ved at tilføje dit forsøg."
                ],
                "preview": "Sådan ser AI'en dit tal",
                "draw_here": "Tegn dit tal her",
                "canvas_tip": "Tip: Klik her og begynd at tegne! Tryk Enter eller 'Gæt mit tal!'.",
                "enter_number": "Indtast tallet du har tegnet:",
                "clear_btn": "Ryd tegning",
                "clear_tip": "Slet din tegning og start forfra (Ctrl+Z)",
                "guess_btn": "Gæt mit tal!",
                "guess_tip": "Lad AI gætte dit tal (Enter eller Ctrl+X)",
                "add_btn": "Tilføj min tegning for at hjælpe AI'en",
                "add_tip": "Hvis AI'en gættede forkert, kan du hjælpe ved at tilføje din tegning! (Ctrl+V)",
                "ai_guess": "AI'ens Gæt",
                "teach_ai": "Lær AI'en",
                "empty_canvas": "Tegn venligst et tal først!",
                "models_not_ready": "Modellerne indlæses stadig, vent venligst",
                "predicted": "Gættet tal: {digit} (sikkerhed: {conf:.2f}%) med {n} model{s}",
            }
        }

        # --- Language Switcher (Flags) ---
        flag_frame = tk.Frame(self, bg="#f0f4ff")
        flag_frame.pack(side=tk.TOP, anchor="nw", padx=18, pady=(2, 0))  # Reduce top padding
        self.flag_imgs = {
            "en": ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "..", "assets", "flag_uk.png")).resize((48, 30))),  # Larger
            "da": ImageTk.PhotoImage(Image.open(os.path.join(script_dir, "..", "assets", "flag_dk.png")).resize((48, 30))),
        }
        self.flag_btn_en = tk.Button(flag_frame, image=self.flag_imgs["en"], bd=0, command=lambda: self.set_language("en"), cursor="hand2")
        self.flag_btn_en.pack(side=tk.LEFT, padx=4)
        self.flag_btn_da = tk.Button(flag_frame, image=self.flag_imgs["da"], bd=0, command=lambda: self.set_language("da"), cursor="hand2")
        self.flag_btn_da.pack(side=tk.LEFT, padx=4)

        # --- Title and subtitle ---
        self.title_frame = tk.Frame(self, bg="#f0f4ff")
        self.title_frame.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))  # Reduce top padding
        self.title_label = tk.Label(self.title_frame, font=('Arial', 22, 'bold'), bg="#f0f4ff", fg="#2a3a6c")
        self.title_label.pack()
        self.subtitle_label = tk.Label(self.title_frame, font=('Arial', 15), bg="#f0f4ff", fg="#444")
        self.subtitle_label.pack(pady=(2, 10))

        # Main frames
        self.left_frame = tk.Frame(self, bg="#f0f4ff")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame = tk.Frame(self, bg="#f0f4ff")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Instructions
        self.instr_frame = tk.Frame(self.left_frame, bg="#eaf1ff", bd=2, relief=tk.GROOVE)
        self.instr_frame.pack(fill=tk.X, padx=18, pady=(2, 0))  # Reduce top padding
        self.instr_title = tk.Label(self.instr_frame, font=('Arial', 13, 'bold'), bg="#eaf1ff", fg="#2a3a6c")
        self.instr_title.pack(anchor="w", padx=8, pady=(4,0))
        self.instr_labels = []
        for _ in range(3):
            lbl = tk.Label(self.instr_frame, font=('Arial', 12), bg="#eaf1ff", fg="#333", justify="left")
            lbl.pack(anchor="w", padx=16, pady=(0,2))
            self.instr_labels.append(lbl)

        # Preview frame for preprocessed image
        self.preview_frame = tk.LabelFrame(self.left_frame, bg="#f0f4ff")
        self.preview_frame.pack(fill=tk.X, padx=18, pady=(2, 0))  # Reduce top padding
        self.preview_canvas = tk.Canvas(self.preview_frame, width=140, height=140, bg="white", highlightthickness=1)
        self.preview_canvas.pack(padx=10, pady=5)

        # Drawing canvas frame
        self.canvas_frame = tk.LabelFrame(self.left_frame, bg="#f0f4ff")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=(2, 0))  # Reduce top padding
        self.canvas_frame.config(text=self.texts[self.language]["draw_here"])

        # Drawing canvas
        self.canvas = DrawingCanvas(self.canvas_frame, width=280, height=280)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.focus_set()  # Set initial focus to canvas

        # Subtle instruction below canvas
        self.canvas_instruction = tk.Label(self.canvas_frame, font=('Arial', 11, 'italic'), bg="#f0f4ff", fg="#888")
        self.canvas_instruction.pack(pady=(0, 6))

        # --- Guess button below canvas, blue border, centered ---
        self.guess_btn_frame = tk.Frame(self.canvas_frame, bg="#f0f4ff", highlightbackground="#4a90e2", highlightthickness=3, bd=0)
        self.guess_btn_frame.pack(pady=(0, 10))
        self.predict_btn = ttk.Button(self.guess_btn_frame, width=15, style='Big.TButton', command=self.predict_digit)
        self.predict_btn.pack(padx=8, pady=4)
        self.predict_btn_tooltip = ToolTip(self.predict_btn, "Let the AI guess your digit (Enter or Ctrl+X)")

        # Canvas controls (digit selector and clear button only)
        self.canvas_controls = tk.Frame(self.left_frame, bg="#f0f4ff")
        self.canvas_controls.pack(fill=tk.X, padx=18, pady=(2, 0))  # Reduce top padding
        self.digit_var = tk.IntVar(value=0)
        digit_frame = tk.Frame(self.canvas_controls, bg="#f0f4ff")
        digit_frame.pack(side=tk.LEFT, padx=5)
        self.enter_number_label = tk.Label(digit_frame, font=('Arial', 12, 'bold'), bg="#f0f4ff", fg="#2a3a6c")
        self.enter_number_label.pack(side=tk.LEFT, padx=10)
        self.digit_selector = ttk.Combobox(
            digit_frame, textvariable=self.digit_var, values=list(range(10)),
            width=10, font=('Arial', 12, 'bold'), state="readonly"
        )
        self.digit_selector.pack(side=tk.LEFT)
        self.digit_selector.current(0)
        self.digit_selector_tooltip = ToolTip(self.digit_selector, "Select the digit you just drew (for training)")

        self.clear_btn = ttk.Button(self.canvas_controls, width=13, style='Big.TButton', command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=8)
        self.clear_btn_tooltip = ToolTip(self.clear_btn, "Erase your drawing and start over (Ctrl+Z)")

        # Training controls
        self.training_controls = tk.Frame(self.left_frame, bg="#f0f4ff")
        self.training_controls.pack(fill=tk.X, padx=18, pady=(2, 0))  # Reduce top padding
        self.add_btn = ttk.Button(
            self.training_controls, width=40, style='Big.TButton', command=lambda: self.training_panel.add_to_training()
        )
        self.add_btn.pack(fill=tk.X, pady=5)
        self.add_btn_tooltip = ToolTip(self.add_btn, "If the AI guessed wrong, add your drawing to improve it! (Ctrl+V)")

        # Prediction display
        self.prediction_frame = tk.LabelFrame(self.right_frame, bg="#f0f4ff")
        self.prediction_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=(2, 0))  # Reduce top padding
        self.prediction_display = PredictionDisplay(self.prediction_frame)
        self.prediction_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Training panel
        self.training_frame = tk.LabelFrame(self.right_frame, bg="#f0f4ff")
        self.training_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=(2, 0))  # Reduce top padding
        self.training_panel = TrainingPanel(self, self.predict_digit, self.digit_var)
        self.training_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(self, textvariable=self.status_var, bd=2, relief=tk.RIDGE, anchor=tk.W,
                                   font=('Arial', 12, 'bold'), bg="#eaf1ff", fg="#2a3a6c")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 0))  # No extra bottom padding

        # Keyboard shortcuts
        self.bind("<Control-z>", lambda event: self.clear_canvas())
        self.bind("<Control-x>", lambda event: self.predict_digit())
        self.bind("<Control-v>", lambda event: self.training_panel.add_to_training())
        self.bind("<Control-n>", lambda event: self.training_panel.retrain_model())
        self.bind("<Escape>", lambda event: self.clear_canvas())

        # Highlight Guess button when canvas is focused
        self.canvas.bind("<FocusIn>", lambda e: self._highlight_guess_button(True))
        self.canvas.bind("<FocusOut>", lambda e: self._highlight_guess_button(False))

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
            print("[GUI] Starting model ensemble initialization...")
            self.ensemble = ModelEnsemble()
            counts = self.ensemble.get_model_counts()
            print(f"[GUI] Model ensemble initialized: MNIST={counts['mnist']}, EMNIST={counts['emnist']}")
            self.after(0, lambda: self.status_var.set("Model ensemble loaded successfully"))
        except Exception as e:
            print(f"[GUI] Error loading models: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self.status_var.set(f"Error loading models: {e}"))

    def _on_canvas_focus(self, event=None):
        # Animate canvas border color on focus
        orig_color = "#4a90e2"
        flash_color = "#ffb347"
        self.canvas.configure(highlightbackground=flash_color)
        self.after(120, lambda: self.canvas.configure(highlightbackground=orig_color))

    def _on_canvas_unfocus(self, event=None):
        self.canvas.configure(highlightbackground="#4a90e2")

    def _highlight_guess_button(self, highlight=True):
        # Change style to visually highlight the Guess button
        style = ttk.Style()
        if highlight:
            style.configure('Big.TButton', background="#ffb347")
            self.predict_btn.configure(style='Big.TButton')
        else:
            style.configure('Big.TButton', background="#e1eaff")
            self.predict_btn.configure(style='Big.TButton')

    def set_language(self, lang):
        self.language = lang
        t = self.texts[lang]
        # Title and subtitle
        self.title_label.config(text=t["title"])
        self.subtitle_label.config(text=t["subtitle"])
        # Instructions
        self.instr_title.config(text=t["instructions_title"])
        for lbl, txt in zip(self.instr_labels, t["instructions"]):
            lbl.config(text=txt)
        # Preview frame
        self.preview_frame.config(text=t["preview"])
        # Canvas frame
        self.canvas_frame.config(text=t["draw_here"])
        self.canvas_instruction.config(text=t["canvas_tip"])
        # Canvas controls
        self.enter_number_label.config(text=t["enter_number"])
        self.clear_btn.config(text=t["clear_btn"])
        self.clear_btn_tooltip.set_text(t["clear_tip"])
        self.predict_btn.config(text=t["guess_btn"])
        self.predict_btn_tooltip.set_text(t["guess_tip"])
        self.add_btn.config(text=t["add_btn"])
        self.add_btn_tooltip.set_text(t["add_tip"])
        # Prediction and training frames
        self.prediction_frame.config(text=t["ai_guess"])
        self.training_frame.config(text=t["teach_ai"])

    def clear_canvas(self):
        self.canvas.clear()
        self.canvas.focus_set()

    def get_current_drawing(self):
        img = self.canvas.get_image()
        if np.max(img) == 0:
            return None
        return img

    def predict_digit(self):
        img = self.get_current_drawing()
        if img is None:
            messagebox.showwarning("", self.texts[self.language]["empty_canvas"])
            self.canvas.focus_set()
            return
        if self.ensemble is None:
            messagebox.showwarning("", self.texts[self.language]["models_not_ready"])
            return

        processed_img, preview_img = preprocess_digit_image(img, preview_size=(140, 140))
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(70, 70, image=preview_img)
        self.preview_canvas.image = preview_img

        prediction = self.ensemble.predict(processed_img)
        self.prediction_display.update_prediction(prediction)

        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        model_counts = self.ensemble.get_model_counts()
        total_models = model_counts['mnist'] + model_counts['emnist']
        s = "" if total_models == 1 else "s"
        self.status_var.set(
            self.texts[self.language]["predicted"].format(
                digit=predicted_digit, conf=confidence, n=total_models, s=s
            )
        )
        self._highlight_guess_button(True)
        self.after(200, lambda: self._highlight_guess_button(False))

def main():
    app = DigitRecognitionApp()
    app.mainloop()

if __name__ == "__main__":
    main()
