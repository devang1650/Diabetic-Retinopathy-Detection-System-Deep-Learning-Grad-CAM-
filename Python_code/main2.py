import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.cm as cm
import sys
from ai_helper import AIHelper  # Make sure this exists


class DRDetectionApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.load_components()

    def setup_ui(self):
        self.root.title("Diabetic Retinopathy Detection")
        self.root.geometry("1100x800")
        self.root.config(bg="#f0f0f0")

        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        header_frame = tk.Frame(main_frame, bg="#007acc")
        header_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(header_frame, text="Diabetic Retinopathy Detection",
                 font=("Arial", 20, "bold"), fg="white", bg="#007acc").pack(pady=10)

        # Image display frame
        img_frame = tk.Frame(main_frame, bg="#f0f0f0")
        img_frame.pack(fill=tk.X, pady=10)

        # Original image
        orig_frame = tk.Frame(img_frame, bg="#f0f0f0")
        orig_frame.pack(side=tk.LEFT, expand=True)
        tk.Label(orig_frame, text="Original Image", font=("Arial", 12), bg="#f0f0f0").pack()
        self.original_label = tk.Label(orig_frame, bg="black", width=300, height=200)
        self.original_label.pack()

        # Grad-CAM image
        grad_frame = tk.Frame(img_frame, bg="#f0f0f0")
        grad_frame.pack(side=tk.RIGHT, expand=True)
        tk.Label(grad_frame, text="Grad-CAM Heatmap", font=("Arial", 12), bg="#f0f0f0").pack()
        self.gradcam_label = tk.Label(grad_frame, bg="black", width=300, height=200)
        self.gradcam_label.pack()

        # Prediction and button
        self.prediction_label = tk.Label(main_frame, text="Upload an image to analyze",
                                         font=("Arial", 14), bg="#f0f0f0")
        self.prediction_label.pack(pady=10)

        self.upload_btn = tk.Button(main_frame, text="Upload Image",
                                    command=self.upload_and_predict,
                                    bg="#007acc", fg="white",
                                    font=("Arial", 12), width=15)
        self.upload_btn.pack(pady=10)

        # Analysis frame
        analysis_frame = tk.Frame(main_frame, bg="#f0f0f0")
        analysis_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(analysis_frame, text="AI Analysis Summary",
                 font=("Arial", 12, "bold"), bg="#f0f0f0").pack(anchor=tk.W)

        self.summary_text = scrolledtext.ScrolledText(analysis_frame,
                                                      wrap=tk.WORD,
                                                      width=100,
                                                      height=10,
                                                      font=("Arial", 10),
                                                      bg="white")
        self.summary_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Initialize placeholders
        self.display_placeholder_images()

    def display_placeholder_images(self):
        """Show placeholder black images before any upload"""
        placeholder = Image.new('RGB', (300, 200), color='black')
        placeholder_tk = ImageTk.PhotoImage(placeholder)

        self.original_label.config(image=placeholder_tk)
        self.original_label.image = placeholder_tk

        self.gradcam_label.config(image=placeholder_tk)
        self.gradcam_label.image = placeholder_tk

    def load_components(self):
        """Load model and AI components"""
        try:
            self.model = load_model('Final_year.h5', compile=False)
            self.class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
            print("Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.model = None

        try:
            self.ai = AIHelper()
            print("AI Helper initialized")
        except Exception as e:
            messagebox.showwarning("Warning", f"AI features disabled:\n{str(e)}")
            self.ai = None

    def upload_and_predict(self):
        """Handle image upload and prediction"""
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp")]
        file_path = filedialog.askopenfilename(title="Select Retina Image", filetypes=filetypes)

        if not file_path or not self.model:
            return

        try:
            # Clear previous results
            self.summary_text.delete(1.0, tk.END)
            self.prediction_label.config(text="Processing...", fg="blue")
            self.root.update()  # Force UI update

            # Preprocess image
            img_array = self.preprocess_image(file_path)

            # Get prediction
            prediction = self.model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = self.class_labels[predicted_class]

            # Generate Grad-CAM
            heatmap = self.make_gradcam_heatmap(img_array)

            # Display results
            self.display_results(file_path, heatmap, predicted_label)

            # Get AI analysis if available
            if self.ai:
                ai_summary = self.ai.get_dr_analysis(predicted_label)
                self.summary_text.insert(tk.END, ai_summary)
            else:
                self.summary_text.insert(tk.END, "AI analysis unavailable")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
            self.prediction_label.config(text="Error processing image", fg="red")
            print(f"Error: {str(e)}")

    def preprocess_image(self, filepath):
        """Preprocess image for model input"""
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Could not read image file")

        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize
        return np.expand_dims(img, axis=0)  # Add batch dimension

    def make_gradcam_heatmap(self, img_array):
        """Generate Grad-CAM heatmap"""
        # Find last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:  # 4D output means conv layer
                last_conv_layer = layer.name
                break

        if not last_conv_layer:
            raise ValueError("Could not find convolutional layer in model")

        # Create gradient model
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(last_conv_layer).output, self.model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Generate heatmap
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy()

    def display_results(self, file_path, heatmap, predicted_label):
        """Display images and prediction results"""
        # Load and resize original image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (300, 200))  # Resize for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create Grad-CAM overlay
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

        # Convert to PhotoImage
        original_img = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        gradcam_img = ImageTk.PhotoImage(Image.fromarray(superimposed_img))

        # Update labels
        self.original_label.config(image=original_img)
        self.original_label.image = original_img
        self.gradcam_label.config(image=gradcam_img)
        self.gradcam_label.image = gradcam_img
        self.prediction_label.config(
            text=f"Prediction: {predicted_label}",
            fg="green"
        )


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = DRDetectionApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application failed to start:\n{str(e)}")
        sys.exit(1)