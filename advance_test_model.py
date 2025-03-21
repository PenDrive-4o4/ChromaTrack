# webcam_app_with_csv.py
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

### Color Classifier Class
class ColorClassifier:
    """KNN-based color classifier using a CSV dataset."""
    def __init__(self, csv_path):
        self.model = KNeighborsClassifier(n_neighbors=9, metric='euclidean')
        self.scaler = StandardScaler()
        self.csv_path = csv_path
        self.colors = None
        self.color_data_rgb = []
        self.labels = []

    def train(self):
        """Load CSV data, split into train/test sets, train the classifier, and compute accuracy."""
        print(f"Loading dataset from {self.csv_path}...")
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

        required_columns = ['red', 'green', 'blue', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        self.colors = sorted(df['label'].unique())
        print(f"Found {len(self.colors)} unique colors: {self.colors}")

        self.color_data_rgb = df[['red', 'green', 'blue']].values.tolist()
        self.labels = [self.colors.index(label) for label in df['label']]

        # Convert RGB to HSV for all data
        print(f"Converting {len(self.color_data_rgb)} RGB points to HSV...")
        color_data_hsv = [cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
                          for rgb in self.color_data_rgb]

        # Split the data into training and test sets (80% train, 20% test)
        print("Splitting dataset into training and test sets (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            color_data_hsv, self.labels, test_size=0.2, random_state=42
        )

        # Scale the training data
        print("Scaling the training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model on the training set
        print("Training with k=7...")
        self.model.fit(X_train_scaled, y_train)
        print("Training complete!")

        # Evaluate the model on the test set
        print("Evaluating model accuracy on test set...")
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy on Test Set: {accuracy:.4f} ({accuracy*100:.2f}%)")

    def predict(self, color):
        """Predict the color name and confidence for a given HSV color."""
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
        scaled_color = self.scaler.transform([hsv_color])
        prediction = self.model.predict(scaled_color)[0]
        confidence = max(self.model.predict_proba(scaled_color)[0])
        return self.colors[prediction], confidence

### Frame Processing Functions
def multi_color_detection(frame, num_colors=3):
    """Detect dominant colors in the frame using K-Means in HSV space."""
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    pixels = frame_hsv.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init=5, max_iter=100)
    kmeans.fit(pixels[::10])  # Subsample for speed
    centers = kmeans.cluster_centers_.astype(int)
    return centers

def process_frame(frame):
    """Process a frame by detecting colors and annotating it."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (640, 480))
    dominant_colors = multi_color_detection(frame_rgb, num_colors=3)
    annotations = []
    for i, color in enumerate(dominant_colors, 1):
        color_name, confidence = color_classifier.predict(color)
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
    return frame_bgr, dominant_colors

### Main Execution
if __name__ == "__main__":
    # Path to your CSV file
    csv_path = "./final_data.csv"  # Update this to the path of your CSV file

    # Initialize and train the classifier
    color_classifier = ColorClassifier(csv_path=csv_path)
    try:
        color_classifier.train()
    except Exception as e:
        print(f"Error training classifier: {e}")
        exit()

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Starting webcam... Press 'q' to quit.")

    last_update_time = time.time()
    last_frame = None
    last_dominant_colors = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            # Process and display the frame
            annotated_frame, dominant_colors = process_frame(frame)
            cv2.imshow('Webcam - Dominant Colors', annotated_frame)

            # Store the last frame and its dominant colors
            last_frame = frame
            last_dominant_colors = dominant_colors

            # Print dominant colors every 5 seconds
            if time.time() - last_update_time > 5:
                print("Three Most Dominant Colors:")
                for i, color in enumerate(dominant_colors, 1):
                    color_name, confidence = color_classifier.predict(color)
                    rgb_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0]
                    print(f"{i}. {color_name} ({confidence:.2f}) - RGB: {rgb_color}")
                last_update_time = time.time()

            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopping... (User pressed 'q')")
                break

    except Exception as e:
        print(f"Error during webcam operation: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped and resources cleaned up.")

    # Post-webcam step: Summarize the last frame's dominant colors
    print("\n=== Post-Webcam Step ===")
    if last_frame is not None and last_dominant_colors is not None:
        print("Summary of the last captured frame's dominant colors:")
        for i, color in enumerate(last_dominant_colors, 1):
            color_name, confidence = color_classifier.predict(color)
            rgb_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0]
            print(f"{i}. {color_name} ({confidence:.2f}) - RGB: {rgb_color}")
    else:
        print("No frames were captured successfully.")