### Objective:
## Develop an automated pipeline to track hand movements in a video by leveraging computer vision techniques and state-of-the-art tools.


### Tools & Libraries Used:

# cv2: OpenCV for video and image processing.
# Google MediaPipe: For detecting and localizing hand positions.
# SAM 2 (Segment Anything Model v2): For generating masks for detected hands in video frames.

### Key References:

# Google MediaPipe for hand detection.
# SAM 2 repository and Video Predictor example for segmentation tasks.

### INSTALLATION STEPS:

# For CPU-only systems (if no GPU):

pip3 install torch torchvision torchaudio

# OpenCV (for video/image processing)
pip install opencv-python

# MediaPipe (for hand detection)
pip install mediapipe

# Segment Anything Model (SAM)
pip install git+https://github.com/facebookresearch/segment-anything.git

# NumPy (for array operations)
pip install numpy

# Run the model

python app.py