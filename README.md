# Image-Processing (Dual Camera Capture & Leaf Image Experiments)

This repository contains simple and practical Python tools for:

1. **Capturing images from two USB cameras at the same time**
2. **Running basic image processing experiments (focused on leaf/plant images)**
3. **Testing small image-processing pipelines in a sandbox script**

The goal of this project is to make **image collection → image processing → experimentation**
easy to understand and easy to run, even for beginners.

---

# 1. Simple explanation (like explaining to Grandpa 👴)

Imagine your computer has **two digital eyes** (two USB cameras).

This project can:

- Open both eyes so you can **see through both cameras**
- Let you **press one key** to take a photo from both cameras together
- Save the photos into a folder on your computer
- If that folder is inside **Google Drive Desktop**,  
  the photos will **automatically upload to the cloud**

After collecting many photos, other scripts in this repo can:

- Clean the images  
- Convert colors  
- Detect shapes or edges  
- Prepare images for training or analysis  

So the full idea is:

**Take photos → Improve photos → Test processing ideas**

---

# 2. Files in this repository

## 2.1 `capture_to_drive_folder.py`
**Purpose:**  
Capture images from **two USB cameras simultaneously** and save them with timestamps.

**Think of it as:**  
Press a button → both cameras take a picture → save into an album folder.

**Common uses:**
- Building image datasets  
- Monitoring plants or leaves  
- Recording experiments from two angles  

---

## 2.2 `leaf_image_processing(training).py`
**Purpose:**  
Run **image-processing experiments** on captured leaf images.

Typical operations in this type of script may include:

- Resize images  
- Convert to grayscale  
- Blur or denoise  
- Threshold / segmentation  
- Edge detection  
- Crop region of interest  

This helps prepare images for **analysis or machine-learning training**.

---

## 2.3 `refer_and_pipeline_testing_vscode.py`
**Purpose:**  
A **testing playground** for trying image-processing steps in sequence.

**Think of it as:**  
Load image → apply step A → apply step B → see result → adjust → repeat.

Useful for:
- Rapid experimentation  
- Debugging processing logic  
- Trying new algorithms quickly  

---

# 3. Requirements

## Hardware
- Windows PC or laptop  
- **Two USB cameras** (for capture script)

## Software
- Python **3.8 or newer** (3.10+ recommended)
- Required Python libraries:
  - `opencv-python`
  - `numpy`
  - *(optional)* `matplotlib`, `pandas`

---

# 4. Installation (step-by-step)

## Step 1 — Download this project

### Option A: Download ZIP from GitHub
Click **Code → Download ZIP**, then unzip.

### Option B: Clone with Git (recommended)

```bash
git clone https://github.com/Chainchigit/Image-Processing.git
cd Image-Processing

