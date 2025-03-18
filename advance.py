import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import time

# KNN Color Classifier using X11's 140 colors with HSV
class ColorClassifier:
    def __init__(self):
        self.model = KNeighborsClassifier(
            n_neighbors=3,  # Default, will tune
            metric='euclidean'
        )
        self.scaler = StandardScaler()
        self.colors = [
            'Alice Blue', 'Antique White', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque', 'Black',
            'Blanched Almond', 'Blue', 'Blue Violet', 'Brown', 'Burlywood', 'Cadet Blue', 'Chartreuse',
            'Chocolate', 'Coral', 'Cornflower Blue', 'Cornsilk', 'Crimson', 'Cyan', 'Dark Blue',
            'Dark Cyan', 'Dark Goldenrod', 'Dark Gray', 'Dark Green', 'Dark Khaki', 'Dark Magenta',
            'Dark Olive Green', 'Dark Orange', 'Dark Orchid', 'Dark Red', 'Dark Salmon', 'Dark Sea Green',
            'Dark Slate Blue', 'Dark Slate Gray', 'Dark Turquoise', 'Dark Violet', 'Deep Pink',
            'Deep Sky Blue', 'Dim Gray', 'Dodger Blue', 'Firebrick', 'Floral White', 'Forest Green',
            'Fuchsia', 'Gainsboro', 'Ghost White', 'Gold', 'Goldenrod', 'Gray', 'Green', 'Green Yellow',
            'Honeydew', 'Hot Pink', 'Indian Red', 'Indigo', 'Ivory', 'Khaki', 'Lavender', 'Lavender Blush',
            'Lawn Green', 'Lemon Chiffon', 'Light Blue', 'Light Coral', 'Light Cyan', 'Light Goldenrod Yellow',
            'Light Gray', 'Light Green', 'Light Pink', 'Light Salmon', 'Light Sea Green', 'Light Sky Blue',
            'Light Slate Gray', 'Light Steel Blue', 'Light Yellow', 'Lime', 'Lime Green', 'Linen', 'Magenta',
            'Maroon', 'Medium Aquamarine', 'Medium Blue', 'Medium Orchid', 'Medium Purple', 'Medium Sea Green',
            'Medium Slate Blue', 'Medium Spring Green', 'Medium Turquoise', 'Medium Violet Red', 'Midnight Blue',
            'Mint Cream', 'Misty Rose', 'Moccasin', 'Navajo White', 'Navy', 'Old Lace', 'Olive', 'Olive Drab',
            'Orange', 'Orange Red', 'Orchid', 'Pale Goldenrod', 'Pale Green', 'Pale Turquoise', 'Pale Violet Red',
            'Papaya Whip', 'Peach Puff', 'Peru', 'Pink', 'Plum', 'Powder Blue', 'Purple', 'Red', 'Rosy Brown',
            'Royal Blue', 'Saddle Brown', 'Salmon', 'Sandy Brown', 'Sea Green', 'Seashell', 'Sienna', 'Silver',
            'Sky Blue', 'Slate Blue', 'Slate Gray', 'Snow', 'Spring Green', 'Steel Blue', 'Tan', 'Teal', 'Thistle',
            'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White', 'White Smoke', 'Yellow', 'Yellow Green'
        ]
        self.color_data_rgb = []
        self.labels = []

    def generate_variations(self, base_rgb, num_samples=20):
        """Generate variations with a smaller range."""
        variations = []
        r, g, b = base_rgb
        for _ in range(num_samples):
            r_var = max(0, min(255, r + np.random.randint(-10, 11)))  # Reduced to ±10
            g_var = max(0, min(255, g + np.random.randint(-10, 11)))
            b_var = max(0, min(255, b + np.random.randint(-10, 11)))
            variations.append([r_var, g_var, b_var])
        return variations

    def train(self):
        # X11's 140 colors with their standard RGB values
        x11_base_colors = [
            [240, 248, 255], [250, 235, 215], [0, 255, 255], [127, 255, 212], [240, 255, 255],
            [245, 245, 220], [255, 228, 196], [0, 0, 0], [255, 235, 205], [0, 0, 255],
            [138, 43, 226], [165, 42, 42], [222, 184, 135], [95, 158, 160], [127, 255, 0],
            [210, 105, 30], [255, 127, 80], [100, 149, 237], [255, 248, 220], [220, 20, 60],
            [0, 255, 255], [0, 0, 139], [0, 139, 139], [184, 134, 11], [169, 169, 169],
            [0, 100, 0], [189, 183, 107], [139, 0, 139], [85, 107, 47], [255, 140, 0],
            [153, 50, 204], [139, 0, 0], [233, 150, 122], [143, 188, 143], [72, 61, 139],
            [47, 79, 79], [0, 206, 209], [148, 0, 211], [255, 20, 147], [0, 191, 255],
            [105, 105, 105], [30, 144, 255], [178, 34, 34], [255, 250, 240], [34, 139, 34],
            [255, 0, 255], [220, 220, 220], [248, 248, 255], [255, 215, 0], [218, 165, 32],
            [128, 128, 128], [0, 128, 0], [173, 255, 47], [240, 255, 240], [255, 105, 180],
            [205, 92, 92], [75, 0, 130], [255, 245, 240], [240, 230, 140], [230, 230, 250],
            [255, 240, 245], [124, 252, 0], [255, 250, 205], [173, 216, 230], [240, 128, 128],
            [224, 255, 255], [238, 221, 130], [211, 211, 211], [144, 238, 144], [255, 182, 193],
            [255, 160, 122], [32, 178, 170], [135, 206, 250], [119, 136, 153], [176, 196, 222],
            [255, 255, 224], [0, 255, 0], [50, 205, 50], [250, 240, 230], [255, 0, 255],
            [128, 0, 0], [102, 205, 170], [0, 0, 205], [186, 85, 211], [147, 112, 219],
            [60, 179, 113], [123, 104, 238], [0, 250, 154], [72, 209, 204], [199, 21, 133],
            [25, 25, 112], [245, 245, 220], [255, 228, 225], [255, 228, 181], [255, 222, 173],
            [0, 0, 128], [253, 245, 230], [128, 128, 0], [107, 142, 35], [255, 165, 0],
            [255, 69, 0], [218, 112, 214], [238, 232, 170], [152, 251, 152], [175, 238, 238],
            [219, 112, 147], [255, 239, 213], [255, 218, 185], [205, 133, 63], [255, 192, 203],
            [221, 160, 221], [176, 224, 230], [128, 0, 128], [255, 0, 0], [188, 143, 143],
            [65, 105, 225], [139, 69, 19], [250, 128, 114], [244, 164, 96], [46, 139, 87],
            [255, 245, 238], [160, 82, 45], [192, 192, 192], [135, 206, 235], [106, 90, 205],
            [112, 128, 144], [255, 250, 250], [0, 255, 127], [70, 130, 180], [210, 180, 140],
            [0, 128, 128], [216, 191, 216], [255, 99, 71], [64, 224, 208], [238, 130, 238],
            [245, 222, 179], [255, 255, 255], [245, 245, 245], [255, 255, 0], [154, 205, 50]
        ]

        # Generate 20 variations for each of the 140 X11 colors
        self.color_data_rgb = []
        self.labels = []
        for idx, base_rgb in enumerate(x11_base_colors):
            variations = self.generate_variations(base_rgb, num_samples=20)
            self.color_data_rgb.extend(variations)
            self.labels.extend([idx] * 20)

        # Convert RGB to HSV
        self.color_data_hsv = [cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
                               for rgb in self.color_data_rgb]
        scaled_data = self.scaler.fit_transform(self.color_data_hsv)

        # Tune k and select the best
        best_k = 3
        best_accuracy = 0
        for k in [1, 3, 5, 7, 10, 15]:
            model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            scores = cross_val_score(model, scaled_data, self.labels, cv=5)
            accuracy = scores.mean()
            print(f"k={k}: Accuracy: {accuracy:.2f} (+/- {scores.std() * 2:.2f})")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        # Train with the best k
        print(f"Training with best k={best_k}")
        self.model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
        self.model.fit(scaled_data, self.labels)

    def predict(self, color):
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
        scaled_color = self.scaler.transform([hsv_color])
        prediction = self.model.predict(scaled_color)[0]
        confidence = max(self.model.predict_proba(scaled_color)[0])
        return self.colors[prediction], confidence

# Multi-color detection with K-Means in HSV space
def multi_color_detection(frame, num_colors=3):
    # Convert frame to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    pixels = frame_hsv.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init=5, max_iter=100)
    kmeans.fit(pixels[::10])
    centers = kmeans.cluster_centers_.astype(int)
    return centers

# Process and annotate a single frame
def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (640, 480))
    dominant_colors = multi_color_detection(frame_rgb, num_colors=3)  # Now in HSV
    annotations = []
    for i, color in enumerate(dominant_colors, 1):
        color_name, confidence = color_classifier.predict(color)  # Predict using HSV directly
        # Convert HSV back to RGB for display
        rgb_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0]
        annotations.append((rgb_color, color_name, confidence))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for i, (color, name, confidence) in enumerate(annotations):
        y_pos = 30 + i * 40
        text = f"{i + 1}. {name} ({confidence:.2f}) - RGB: {color}"
        cv2.putText(frame_bgr, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
        circle_color = tuple(int(c) for c in color[::-1])  # BGR for display
        cv2.circle(frame_bgr, (300, y_pos - 10), 15, circle_color, -1)
    return frame_bgr

# Initialize and train classifier
color_classifier = ColorClassifier()
color_classifier.train()

# Start video capture and processing
cap = cv2.VideoCapture(0)
last_update_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break

        annotated_frame = process_frame(frame)
        cv2.imshow('Webcam - Dominant Colors', annotated_frame)

        if time.time() - last_update_time > 5:
            dominant_colors = multi_color_detection(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print("Three Most Dominant Colors:")
            for i, color in enumerate(dominant_colors, 1):
                color_name, confidence = color_classifier.predict(color)
                rgb_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0]
                print(f"{i}. {color_name} ({confidence:.2f}) - RGB: {rgb_color}")
            last_update_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    cap.release()
    cv2.destroyAllWindows()
