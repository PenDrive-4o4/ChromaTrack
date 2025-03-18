import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Scale, Button, Label, StringVar, Checkbutton, IntVar, Canvas
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import threading
import time

# Enhanced SVM Color Classifier with more blue data
class ColorClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True, C=1.0)
        self.scaler = StandardScaler()
        self.colors = ['Red', 'Green', 'Blue', 'Yellow', 'White', 'Black', 'Orange', 'Purple']
        self.color_data = []
        self.labels = []

    def train(self):
        self.color_data = [
            [255, 0, 0], [200, 50, 50],  # Red
            [0, 255, 0], [50, 200, 50],  # Green
            [0, 0, 255], [50, 50, 200], [0, 150, 255], [70, 130, 180],  # More Blue variations
            [255, 255, 0], [200, 200, 50],  # Yellow
            [255, 255, 255], [240, 240, 240],  # White
            [0, 0, 0], [20, 20, 20],  # Black
            [255, 165, 0], [220, 140, 0],  # Orange
            [128, 0, 128], [100, 0, 100]  # Purple
        ]
        self.labels = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
        scaled_data = self.scaler.fit_transform(self.color_data)
        self.model.fit(scaled_data, self.labels)

    def predict(self, color):
        scaled_color = self.scaler.transform([color])
        prediction = self.model.predict(scaled_color)[0]
        confidence = max(self.model.predict_proba(scaled_color)[0])
        return self.colors[prediction], confidence

# Global variables
lower_bounds = [np.array([0, 50, 50])]
upper_bounds = [np.array([10, 255, 255])]
track_window = None
color_classifier = ColorClassifier()
color_classifier.train()
running = True
current_frame = None
track_enabled = False
selected_rgb = None

# Adaptive color detection with improved lighting adjustment
def adaptive_color_detection(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Enhance lighting robustness
    h, s, v = cv2.split(hsv_frame)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    
    masks = []
    highlighted = frame.copy()
    for lower, upper in zip(lower_bounds, upper_bounds):
        mask = cv2.inRange(hsv_eq, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        masks.append(mask)
        highlighted = cv2.bitwise_and(highlighted, highlighted, mask=mask)
    return masks, highlighted

# Multi-color detection with K-Means
def multi_color_detection(frame, num_colors=3):
    pixels = frame.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init=5, max_iter=100)
    kmeans.fit(pixels[::10])
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.predict(pixels)
    segmented = centers[labels].reshape(frame.shape)
    return centers, segmented

# CamShift object tracking
def track_object(frame, mask):
    global track_window
    if track_window is None:
        return frame, None
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    ret, track_window = cv2.CamShift(mask, track_window, term_crit)
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    return cv2.polylines(frame, [pts], True, (0, 255, 0), 2), track_window

# Mouse callback for selecting colors and tracking
def mouse_callback(event):
    global track_window, lower_bounds, upper_bounds, track_enabled, selected_rgb
    x, y = event.x, event.y
    frame = current_frame.copy()
    rgb_color = frame[y, x]  # BGR
    hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_BGR2HSV)[0][0]
    selected_rgb = rgb_color[::-1]  # RGB
    
    if event.num == 1:  # Left-click
        track_window = (x - 50, y - 50, 100, 100)  # Larger ROI for better tracking
        track_enabled = track_var.get() == 1
        tolerance = int(tolerance_slider.get())
        # Adjusted HSV bounds for better blue detection
        lower_bounds[0] = np.array([max(0, hsv_color[0] - tolerance), max(50, hsv_color[1] - 60), max(50, hsv_color[2] - 60)])
        upper_bounds[0] = np.array([min(179, hsv_color[0] + tolerance), 255, 255])
        print(f"Clicked HSV: {hsv_color}, Lower Bound: {lower_bounds[0]}, Upper Bound: {upper_bounds[0]}")
    
    elif event.num == 3:  # Right-click
        tolerance = int(tolerance_slider.get())
        lower_bounds.append(np.array([max(0, hsv_color[0] - tolerance), max(50, hsv_color[1] - 60), max(50, hsv_color[2] - 60)]))
        upper_bounds.append(np.array([min(179, hsv_color[0] + tolerance), 255, 255]))
    
    predicted_color, confidence = color_classifier.predict(selected_rgb)
    info_text.set(f"RGB: {selected_rgb}\nHSV: {hsv_color}\nPredicted: {predicted_color} ({confidence:.2f})")
    update_swatches()

# Reset secondary colors
def reset_secondary_colors():
    global lower_bounds, upper_bounds
    lower_bounds = [lower_bounds[0]]
    upper_bounds = [upper_bounds[0]]
    update_swatches()

# Update color swatches
def update_swatches():
    if selected_rgb is not None:
        selected_swatch.configure(bg=f'#{selected_rgb[0]:02x}{selected_rgb[1]:02x}{selected_rgb[2]:02x}')
    
    for i, swatch in enumerate(dominant_swatches):
        if i < len(dominant_colors):
            r, g, b = dominant_colors[i]
            swatch.configure(bg=f'#{r:02x}{g:02x}{b:02x}')
        else:
            swatch.configure(bg='gray')

# GUI update function
def update_gui():
    global current_frame, dominant_colors
    dominant_colors = []
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))
        current_frame = frame.copy()

        # Adaptive color detection
        masks, highlighted = adaptive_color_detection(frame)
        
        # Multi-color detection and segmentation
        dominant_colors, segmented = multi_color_detection(frame, num_colors=int(num_colors_slider.get()))
        color_names = []
        for color in dominant_colors:
            name, confidence = color_classifier.predict(color)
            color_names.append(f"{name} ({confidence:.2f})")
        colors_text.set("Dominant Colors:\n" + "\n".join(color_names))
        root.after(0, update_swatches)

        # Object tracking
        display_frame = frame.copy()
        if track_enabled and track_window:
            tracked_frame, track_window = track_object(display_frame, masks[0])
            display_frame = tracked_frame if tracked_frame is not None else display_frame

        # Combine visualizations
        display_frame = cv2.addWeighted(display_frame, 0.7, highlighted, 0.3, 0)
        if segment_var.get():
            display_frame = cv2.addWeighted(display_frame, 0.7, segmented, 0.3, 0)
        
        # Convert for Tkinter
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=frame_img)
        root.after(0, lambda: label.configure(image=imgtk) or setattr(label, 'imgtk', imgtk))
        time.sleep(0.5)

# Stop the application
def stop_app():
    global running
    running = False
    cap.release()
    root.quit()

# GUI Setup
root = tk.Tk()
root.title("Advanced Color Detection and Tracking")

cap = cv2.VideoCapture(0)
label = tk.Label(root)
label.pack()

# Mouse bindings
def on_mouse(event):
    mouse_callback(type('Event', (), {'x': event.x, 'y': event.y, 'num': 1 if event.num == 1 else 3})())
label.bind("<Button-1>", on_mouse)
label.bind("<Button-3>", on_mouse)

# Text displays
info_text = StringVar()
colors_text = StringVar()
info_label = Label(root, textvariable=info_text, justify="left")
info_label.pack(side=tk.LEFT, padx=10)
colors_label = Label(root, textvariable=colors_text, justify="left")
colors_label.pack(side=tk.RIGHT, padx=10)

# Color swatches
swatch_frame = tk.Frame(root)
swatch_frame.pack()
Label(swatch_frame, text="Selected Color:").pack(side=tk.LEFT)
selected_swatch = Canvas(swatch_frame, width=30, height=30, bg='gray')
selected_swatch.pack(side=tk.LEFT, padx=5)
Label(swatch_frame, text="Dominant Colors:").pack(side=tk.LEFT)
dominant_swatches = [Canvas(swatch_frame, width=30, height=30, bg='gray') for _ in range(5)]
for swatch in dominant_swatches:
    swatch.pack(side=tk.LEFT, padx=2)

# Controls
tolerance_slider = Scale(root, from_=5, to=30, label="HSV Tolerance", orient=tk.HORIZONTAL)
tolerance_slider.set(10)
tolerance_slider.pack()
num_colors_slider = Scale(root, from_=1, to=5, label="Number of Colors", orient=tk.HORIZONTAL)
num_colors_slider.set(3)
num_colors_slider.pack()
track_var = IntVar()
track_check = Checkbutton(root, text="Enable Tracking", variable=track_var)
track_check.pack()
segment_var = IntVar()
segment_check = Checkbutton(root, text="Show Segmentation", variable=segment_var)
segment_check.pack()
reset_button = Button(root, text="Reset Secondary Colors", command=reset_secondary_colors)
reset_button.pack()
stop_button = Button(root, text="Stop", command=stop_app)
stop_button.pack()

# Start GUI in thread
dominant_colors = []
thread = threading.Thread(target=update_gui, daemon=True)
thread.start()

root.mainloop()
cv2.destroyAllWindows()