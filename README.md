# Image Processing Toolkit  
**Dual Camera Capture · Leaf Image Processing · Pipeline Experimentation (Python + OpenCV)**

A practical Python-based toolkit for:

- Capturing synchronized images from **two USB cameras**
- Performing **basic image processing experiments** focused on **leaf/plant data**
- Testing **image-processing pipelines** for research and machine-learning preparation

This repository demonstrates a complete mini-workflow:

> **Collect images → Process images → Experiment with pipelines**

---

# Project Overview (Explained Simply 👴)

Imagine your computer has **two digital eyes** (two USB cameras).

This project lets you:

1. **Open both cameras** and view them live  
2. **Press one key** to capture photos from both cameras at the same moment  
3. **Automatically save** photos with timestamps  
4. **Auto-sync to Google Drive** if the save folder is inside Drive Desktop  

After collecting images, other scripts help:

- Clean and transform images  
- Detect edges or regions of interest  
- Prepare datasets for **data analysis or machine learning**

So the full concept is:

> **Take photos → Improve photos → Test ideas**

---

# Project Structure

Image-Processing/
├── capture_to_drive_folder.py # Dual-camera image capture
├── leaf_image_processing(training).py # Leaf image processing experiments
├── refer_and_pipeline_testing_vscode.py # Pipeline testing sandbox
├── data/ # (optional) image storage
│ ├── raw/
│ └── processed/
└── README.md

yaml
Copy code

---

# Key Features

- 📷 **Synchronized dual-camera capture**
- 🕒 **Timestamped image saving**
- ☁️ **Optional Google Drive auto-sync**
- 🌿 **Leaf-focused preprocessing workflow**
- 🧪 **Pipeline experimentation environment**
- 🐍 **Pure Python + OpenCV implementation**

---

# Requirements

## Hardware
- Windows PC or laptop  
- **Two USB cameras**

## Software
- Python **3.8+** (3.10 recommended)

### Python Libraries
```bash
pip install opencv-python numpy
Optional:

bash
Copy code
pip install matplotlib pandas
Installation
1. Clone the repository
bash
Copy code
git clone https://github.com/Chainchigit/Image-Processing.git
cd Image-Processing
2. Create a virtual environment (recommended)
bash
Copy code
python -m venv venv
venv\Scripts\activate
3. Install dependencies
bash
Copy code
pip install opencv-python numpy
Usage Guide
1️⃣ Dual Camera Capture
Run:

bash
Copy code
python capture_to_drive_folder.py
Two preview windows will appear.

Keyboard Controls
Key	Action
c	Capture images from both cameras
q	Quit program

Images are saved automatically with timestamps.

Google Drive Sync (Optional)
If the save folder is inside:

php-template
Copy code
Google Drive Desktop → My Drive → <folder>
then captured images will sync to the cloud automatically.

2️⃣ Leaf Image Processing
bash
Copy code
python "leaf_image_processing(training).py"
Typical processing steps may include:

Resize

Grayscale conversion

Noise reduction

Thresholding / segmentation

Edge detection

Region extraction

Ensure input/output paths inside the script are correct.

3️⃣ Pipeline Testing
bash
Copy code
python refer_and_pipeline_testing_vscode.py
Used for:

Rapid experimentation

Debugging processing chains

Trying new algorithms

Update image paths if needed.

Example Workflow
Connect two USB cameras

Run capture script

Press c to collect paired images

Images saved with timestamps

Run processing script to clean/prepare data

Use pipeline script to test new ideas

Troubleshooting
Camera not opening
Another app is using the camera (Zoom, Teams, etc.)

Replug USB cable or change port

Camera IDs may be swapped (0 ↔ 1)

Both windows show the same camera
Swap camera IDs in the script

Slow or laggy preview
Reduce camera resolution

Close heavy programs

Google Drive not syncing
Ensure Google Drive for Desktop is running

Confirm save folder is inside the synced Drive path

Future Improvements
🎥 Video recording support

🤖 Integration with AI segmentation / ML models

🖥 GUI interface for non-technical users

🔄 Camera synchronization calibration

☁️ Direct cloud upload via API

License
This project may be released under the MIT License.

Author
Chainchigit
GitHub: https://github.com/Chainchigit/Image-Processing
