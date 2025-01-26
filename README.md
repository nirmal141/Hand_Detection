# Automatic Hand Tracking Pipeline

A pipeline to detect and track hand movements in videos using **Google MediaPipe** for hand detection and **SAM (Segment Anything Model)** for segmentation and tracking.

## Features
- Detect hands in the first frame using MediaPipe.
- Track hands across all frames using SAM 2.
- Generate an output video with masked hand regions.

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nirmal141/Hand-Detection.git
   cd Hand_Detection

2. **Create Conda Environment**:
   ```bash
   conda create -n sam2 python=3.11
   conda activate sam2

3. **Install Dependencies**:
   ```bash
   # Install PyTorch (MPS support for Apple Silicon)
   pip3 install torch torchvision torchaudio

   # Install other libraries
   pip install opencv-python mediapipe numpy
   
   # Install Segment Anything Model (SAM)
   pip install git+https://github.com/facebookresearch/segment-anything.git

4. **Run the Pipeline**:
   ```bash
   python app.py --input test.mp4 --output output.mp4
