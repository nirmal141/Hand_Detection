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
   git clone https://github.com/yourusername/hand-tracking.git
   cd hand-tracking
