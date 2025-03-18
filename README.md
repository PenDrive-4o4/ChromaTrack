Advanced Color Detection and Tracking
Welcome to the Advanced Color Detection and Tracking project! This application uses computer vision techniques to detect and track colors in real-time using a webcam. It combines adaptive color detection, multi-color segmentation, object tracking, and machine learning-based color classification, all wrapped in an interactive GUI.

Features
Real-Time Color Detection: Detects colors using HSV thresholding and highlights them on the video feed.
Multi-Color Segmentation: Uses K-Means clustering to segment and identify dominant colors in the frame.
Object Tracking: Implements CamShift tracking to follow objects of a specific color.
Adaptive Color Detection: Adjusts to varying lighting conditions with histogram equalization.
Machine Learning Classification: Utilizes an SVM classifier to predict color names with confidence scores.
Interactive GUI: Features sliders, checkboxes, color swatches, and a reset button for user control and visualization.
Prerequisites
Python: Version 3.6 or higher.
Webcam: A connected webcam is required for real-time video input.
Installation
Dependencies
Install the required Python libraries using pip:

bash

Collapse

Wrap

Copy
pip install numpy Pillow opencv-python scikit-learn
Tkinter: Usually included with Python, but on Linux, you may need to install it separately:
bash

Collapse

Wrap

Copy
sudo apt-get install python3-tk  # For Debian/Ubuntu
All-in-One Installation
To install all dependencies at once:

bash

Collapse

Wrap

Copy
pip install numpy Pillow opencv-python scikit-learn
Usage
Save the Script:
Copy the provided Python code into a file named color_detection.py.
Navigate to the Directory:
Open a terminal or command prompt and change to the directory containing color_detection.py:
bash

Collapse

Wrap

Copy
cd /path/to/your/directory
Run the Script:
Execute the script using:
bash

Collapse

Wrap

Copy
python color_detection.py
or, on systems requiring python3:
bash

Collapse

Wrap

Copy
python3 color_detection.py
Interact with the GUI:
Left-Click: Select a pixel to set the primary color and initialize tracking (if enabled).
Right-Click: Add a secondary color for multi-color detection.
Sliders: Adjust "HSV Tolerance" (5–30) and "Number of Colors" (1–5).
Checkboxes: Toggle "Enable Tracking" or "Show Segmentation".
Reset Button: Clear secondary colors.
Stop Button: Exit the application.
Color Swatches: Visualize the selected color and dominant colors detected.
Troubleshooting
No Webcam Found: Ensure your webcam is connected and not in use by another application. Test with:
python

Collapse

Wrap

Copy
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(ret)
cap.release()
Module Not Found: Re-run the pip install command for the missing library (e.g., pip install opencv-python).
Performance Lag: Increase time.sleep(0.5) to time.sleep(1.0) in the code or reduce the resolution in cv2.resize(frame, (640, 480)) to (320, 240).
Color Detection Issues: Adjust the "HSV Tolerance" slider or check the printed HSV bounds in the console (added for debugging) to fine-tune the detection.
How It Works
HSV Thresholding: Detects colors by converting the frame to HSV and applying a range-based mask, enhanced with histogram equalization for lighting robustness.
K-Means Clustering: Segments the image into dominant colors, visualized as an overlay when "Show Segmentation" is enabled.
CamShift Tracking: Tracks the object of the primary color, outlined with a green bounding box.
SVM Classification: Predicts color names (e.g., "Blue", "Green") with confidence scores based on RGB values.
GUI: Built with Tkinter, displaying real-time video, text info, and interactive controls.
Customization
HSV Bounds: Modify the lower_bounds and upper_bounds in mouse_callback to better match specific colors (e.g., blue: Hue 100–130).
Training Data: Expand the color_data in ColorClassifier.train() with more color variations for improved classification.
Update Speed: Adjust time.sleep(0.5) in update_gui() for faster or slower updates.
Contributing
Feel free to fork this repository, submit issues, or send pull requests to improve the project. Suggestions for adding deep learning trackers (e.g., DeepSORT) or GPU acceleration are welcome!

License
This project is open-source and available under the MIT License. See the LICENSE file for details (if you choose to add one).

Acknowledgments
Built with the help of OpenCV, scikit-learn, and Tkinter communities.
Inspired by xAI's Grok AI assistance in developing this code.
How to Use the README
Save the File:
Copy the content above into a file named README.md in the same directory as color_detection.py.
View in GitHub:
If you push this to a GitHub repository, the Markdown will render nicely with formatted text, code blocks, and lists.
Local Viewing:
Open README.md in any text editor or Markdown viewer to see the formatted version. Alternatively, use a command-line tool like grip:
bash

Collapse

Wrap

Copy
pip install grip
grip
Then open http://localhost:6419 in your browser.
