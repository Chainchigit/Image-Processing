# Image-Processing Project

This repository contains multiple Python scripts and experiments for **image processing tasks**, including dual USB camera capture, basic image transformations, and pipeline testing.

📁 All code is written in **Python** and designed to explore practical image processing workflows and utilities.

---

## 🧠 Project Overview

The repository includes:

### 📸 Camera Capture Module
- **`capture_to_drive_folder.py`**  
  A script to capture images from two USB cameras simultaneously and save them to a specified folder (e.g., Google Drive sync).  
  Useful for collecting datasets or automating multi-camera image capture workflows.

### 🌿 Image Processing Experiments
- **`leaf_image_processing(training).py`**  
  Experimental image processing code — likely aimed at training or analyzing leaf images.  
  This may include filtering, transformation, segmentation, or other computer vision experiments.

### 🧪 Pipeline Testing
- **`refer_and_pipeline_testing_vscode.py`**  
  Utility/testing script for experimenting with processing pipelines and integrating various functions.

---

## 🚀 Getting Started

### 📌 Prerequisites

Make sure you have:

- Python 3.8 or newer installed
- Camera(s) connected (for capture script)
- Required libraries installed (see below)

---

### 📦 Install dependencies

Create a virtual environment and install necessary packages:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

pip install opencv-python numpy

