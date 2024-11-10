import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from deepface import DeepFace
import os
import numpy as np

# Disable oneDNN options for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

root = tk.Tk()
root.title("Nationality Detection")
root.geometry("600x600")

# Global variable to store the image path
image_path = None

# Label to display the result
result_label = tk.Label(root, text="Results will be displayed here", font=("Arial", 12), wraplength=500)
result_label.pack(pady=10)

# Label to display the image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Dress Color Detection Function
def detect_dress_color(img):
    # Assuming that the lower part of the image is the dress, crop and analyze
    height, width, _ = img.shape
    dress_region = img[int(height * 0.5):, :]
    avg_color_per_row = np.average(dress_region, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    color_name = 'Unknown'

    # Simple color thresholding (you can expand it for more colors)
    if avg_color[0] > 150:  # Blueish
        color_name = "Blue"
    elif avg_color[1] > 150:  # Greenish
        color_name = "Green"
    elif avg_color[2] > 150:  # Reddish
        color_name = "Red"

    return color_name

# Function to handle image selection
def select_image():
    global image_path
    image_path = filedialog.askopenfilename(initialdir="/", title="Select an Image",
                                            filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("all files", "*.*")))
    if image_path:
        try:
            img = Image.open(image_path)
            img = img.resize((250, 250), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img
        except Exception as e:
            result_label.config(text=f"Error loading image: {e}")

# Function to predict nationality, age, emotion, and dress color
def predict_nationality_details():
    global image_path
    if image_path:
        try:
            # Load and analyze image
            img = cv2.imread(image_path)
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform ethnicity and emotion detection
            predictions = DeepFace.analyze(color_img, actions=['race', 'age', 'emotion'])
            nationality = predictions[0]['dominant_race']
            emotion = predictions[0]['dominant_emotion']
            age = predictions[0]['age']

            # Determine nationality-based logic
            result_text = f"Nationality: {nationality}\nEmotion: {emotion}"

            # Conditional logic for nationality
            if nationality == "indian":
                dress_color = detect_dress_color(img)
                result_text += f"\nAge: {age}\nDress Color: {dress_color}"
            elif nationality == "white":  # Assuming white is American
                result_text += f"\nAge: {age}"
            elif nationality == "black":  # Assuming black is African
                dress_color = detect_dress_color(img)
                result_text += f"\nDress Color: {dress_color}"

            # Display the result
            result_label.config(text=result_text)

        except Exception as e:
            result_label.config(text=f"Error during analysis: {e}")
    else:
        result_label.config(text="Please select an image first.")

# Button to select an image (left side)
button_select_image = tk.Button(root, text="Select Image", command=select_image,
                                activebackground="blue", activeforeground="white",
                                anchor="center", bd=3, bg="lightgray", cursor="hand2",
                                fg="black", font=("Arial", 12), height=2, width=15)
button_select_image.pack(side=tk.LEFT, padx=20, pady=20)

# Button to get nationality and other predictions (right side)
button_get_output = tk.Button(root, text="Get Output", command=predict_nationality_details,
                              activebackground="blue", activeforeground="white",
                              anchor="center", bd=3, bg="lightgray", cursor="hand2",
                              fg="black", font=("Arial", 12), height=2, width=15)
button_get_output.pack(side=tk.RIGHT, padx=20, pady=20)

root.mainloop()
