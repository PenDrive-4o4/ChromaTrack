import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import cv2

# Parameters for color detection
lower_bound = np.array([0, 50, 50])  # HSV lower bound for color detection
upper_bound = np.array([10, 255, 255])  # HSV upper bound for color detection

# Machine Learning Model for Color Classification
class ColorClassifier:
    def __init__(self):
        self.model = SVC(kernel='linear')
        self.colors = ['Red', 'Green', 'Blue', 'Yellow']
        self.color_data = []
        self.labels = []

    def train(self):
        # Example: Train with some sample data
        self.color_data = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]  # RGB values
        self.labels = [0, 1, 2, 3]  # Corresponding labels
        self.model.fit(self.color_data, self.labels)

    def predict(self, color):
        return self.colors[self.model.predict([color])[0]]


color_classifier = ColorClassifier()
color_classifier.train()

# Function for adaptive color detection
def adaptive_color_detection(frame):
    # Convert frame to HSV (using OpenCV)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    return mask

# Function for multi-color detection using K-Means
def multi_color_detection(frame, num_colors=3):
    pixels = frame.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# Function to update the GUI
def update_gui():
    ret, frame = cap.read()
    if not ret:
        return

    # Resize the frame for display
    frame = cv2.resize(frame, (640, 480))

    # Adaptive Color Detection
    mask = adaptive_color_detection(frame)

    # Multi-Color Detection
    dominant_colors = multi_color_detection(frame)
    print("Dominant Colors:", dominant_colors)

    # Machine Learning Color Classification
    for color in dominant_colors:
        predicted_color = color_classifier.predict(color)
        print(f"Predicted Color: {predicted_color}")

    # Display the frame in the GUI
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=frame)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, update_gui)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# GUI Setup
root = tk.Tk()
root.title("Advanced Color Detection")
label = tk.Label(root)
label.pack()

# Start the GUI update loop
update_gui()
root.mainloop()

# Release the webcam when the program exits
cap.release()
cv2.destroyAllWindows()
