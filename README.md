# 📸 Image Processing Suite

A comprehensive collection of digital image processing algorithms and techniques implemented using **Python** and **OpenCV**. This repository serves as a practical guide for understanding how pixels are manipulated to enhance, filter, and analyze visual data.

---

## 📑 Table of Contents
* [Overview](#-overview)
* [Core Features](#-core-features)
* [Technologies Used](#-technologies-used)
* [Mathematical Foundations](#-mathematical-foundations)
* [Project Structure](#-project-structure)
* [Installation & Setup](#-installation--setup)
* [Usage Examples](#-usage-examples)

---

## 🧐 Overview
The goal of this project is to implement standard image processing pipeline stages, moving from raw data acquisition to high-level feature extraction. It covers essential concepts used in Medical Imaging, Satellite Analysis, and Computer Vision.



---

## 🚀 Core Features

### 1. Image Enhancement & Transformation
* **Histogram Equalization:** Improving contrast by stretching the intensity range.
* **Geometric Transforms:** Scaling, rotation, and affine transformations.
* **Color Space Conversion:** Switching between RGB, HSV, Lab, and Grayscale for specific analysis needs.

### 2. Noise Reduction & Blurring
* **Linear Filters:** Mean and Gaussian blurring to remove high-frequency noise.
* **Non-Linear Filters:** Median filtering (excellent for salt-and-pepper noise) and Bilateral filtering (edge-preserving smoothing).

### 3. Feature Detection (Edge & Corner)
* **Gradient Operators:** Sobel, Prewitt, and Roberts Cross.
* **Advanced Detection:** Canny Edge Detection and Harris Corner Detection for structural analysis.

### 4. Morphological Operations
* Fundamental operations like **Erosion** and **Dilation**.
* Compound operations like **Opening** (noise removal) and **Closing** (hole filling).

---

## 🛠️ Technologies Used
* **Python 3.x:** Core programming language.
* **OpenCV:** For optimized image processing functions.
* **NumPy:** For high-performance matrix and array manipulations.
* **Matplotlib:** For visualizing histograms and side-by-side comparisons.

---

## 🔬 Mathematical Foundations
Most operations in this repo rely on **Convolution**, defined as:

$$g(x,y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s,t) f(x+s, y+t)$$

Where $f$ is the input image, $w$ is the kernel (mask), and $g$ is the output image.

---

## 📂 Project Structure
```text
Image-Processing/
├── assets/             # Input images for testing
├── src/                # Implementation source code
│   ├── preprocessing.py
│   ├── filters.py
│   └── morphology.py
├── notebooks/          # Interactive Jupyter Notebooks (.ipynb)
├── output/             # Processed results
├── requirements.txt    # Library dependencies
└── README.md
