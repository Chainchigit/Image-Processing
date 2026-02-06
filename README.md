# Image Processing Toolkit  
**Dual Camera Capture · Leaf Image Processing · Pipeline Experimentation (Python + OpenCV)**

A practical Python toolkit for:

- Capturing synchronized images from **two USB cameras**
- Performing **leaf-focused image preprocessing**
- Experimenting with **image-processing pipelines** for research and machine-learning preparation

This repository demonstrates a complete real-world workflow:

> **Data Collection → Image Processing → Pipeline Experimentation → ML Readiness**

---

# 1. Project Overview (Simple Explanation 👴)

Imagine your computer has **two digital eyes** (two USB cameras).

This project allows you to:

1. **Open both cameras** and see live video  
2. **Press one key** to take photos from both cameras at the same time  
3. **Save images automatically** with timestamps  
4. **Auto-upload to Google Drive** if the save folder is inside Drive Desktop  

After collecting images, additional scripts help:

- Clean and transform images  
- Detect edges or important regions  
- Prepare datasets for **data analysis or machine learning**

So the full idea is:

> **Take photos → Improve photos → Test ideas**

---

# 2. Project Structure



Image-Processing/
│
├── capture_to_drive_folder.py
│ Dual USB camera capture script.
│ Opens two cameras simultaneously, shows live preview,
│ and saves synchronized timestamped images to a local
│ or Google Drive–synced directory.
│
├── leaf_image_processing(training).py
│ Experimental preprocessing module focused on leaf/plant data.
│ Includes resizing, grayscale conversion, denoising,
│ thresholding, segmentation, and edge detection
│ for analysis or machine-learning preparation.
│
├── refer_and_pipeline_testing_vscode.py
│ Sandbox script for testing end-to-end processing pipelines.
│ Loads images, applies sequential transformations,
│ visualizes intermediate outputs, and enables rapid
│ prototyping of computer-vision workflows.
│
├── data/ (optional)
│ Storage directory for datasets.
│
│ ├── raw/
│ │ Original captured images from the dual-camera system.
│ │
│ └── processed/
│ Images after preprocessing or analysis.
│
└── README.md
Project documentation, setup instructions, and technical overview.


---

# 3. Key Features

- 📷 **Synchronized dual-camera image capture**
- 🕒 **Automatic timestamped file naming**
- ☁️ **Optional Google Drive auto-synchronization**
- 🌿 **Leaf-focused preprocessing workflow**
- 🧪 **Pipeline experimentation environment**
- 🐍 **Pure Python + OpenCV implementation**
- 🧠 **Foundation for computer-vision / ML datasets**

---

# 4. Requirements

## Hardware
- Windows PC or laptop  
- **Two USB cameras**

## Software
- Python **3.8+** (3.10 recommended)

### Python Libraries
```bash
pip install opencv-python numpy


Optional (for visualization or analysis):

pip install matplotlib pandas

5. Installation
5.1 Clone the repository
git clone https://github.com/Chainchigit/Image-Processing.git
cd Image-Processing

5.2 Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

5.3 Install dependencies
pip install opencv-python numpy

6. Usage Guide
6.1 Dual Camera Capture

Run:

python capture_to_drive_folder.py


Two preview windows will appear.

Keyboard Controls
Key	Action
c	Capture images from both cameras
q	Quit program

Images are saved automatically with timestamps.

Google Drive Sync (Optional)

If the save folder is inside:

Google Drive Desktop → My Drive → <folder>


captured images will sync to the cloud automatically.

6.2 Leaf Image Processing
python "leaf_image_processing(training).py"


Typical preprocessing steps:

Resize

Grayscale conversion

Noise reduction

Thresholding / segmentation

Edge detection

Region extraction

Ensure input/output paths inside the script are correct.

6.3 Pipeline Testing
python refer_and_pipeline_testing_vscode.py


Used for:

Rapid experimentation

Debugging transformation chains

Trying new computer-vision algorithms

Update image paths if required.

7. Example Workflow

Connect two USB cameras

Run capture script

Press c to collect synchronized images

Images saved with timestamps

Run preprocessing script to clean/prepare data

Use pipeline script to test new ideas

8. Troubleshooting
Camera not opening

Another application is using the camera (Zoom, Teams, etc.)

Replug USB cable or change port

Camera IDs may be swapped (0 ↔ 1)

Both windows show the same camera

Swap camera IDs in the script

Slow or laggy preview

Reduce camera resolution

Close heavy applications

Google Drive not syncing

Ensure Google Drive for Desktop is running

Confirm the save folder is inside the synced Drive path

9. Future Improvements

🎥 Video recording support

🤖 Integration with AI segmentation / ML inference

🖥 GUI interface for non-technical users

🔄 Camera synchronization calibration

☁️ Direct cloud upload via API

📊 Dataset annotation & training pipeline integration

10. License

This project may be distributed under the MIT License.

11. Author

Chainchigit
GitHub: https://github.com/Chainchigit/Image-Processing
