📸 Image Processing SuiteA comprehensive collection of digital image processing algorithms and techniques implemented using Python and OpenCV. This repository serves as a practical guide for understanding how pixels are manipulated to enhance, filter, and analyze visual data.📑 Table of ContentsOverviewCore FeaturesTechnologies UsedMathematical FoundationsInstallation & SetupUsage ExamplesProject Roadmap🧐 OverviewThe goal of this project is to implement standard image processing pipeline stages, moving from raw data acquisition to high-level feature extraction. It covers essential concepts used in Medical Imaging, Satellite Analysis, and Computer Vision.🚀 Core Features1. Image Enhancement & TransformationHistogram Equalization: Improving contrast by stretching the intensity range.Geometric Transforms: Scaling, rotation, and affine transformations.Color Space Conversion: Switching between RGB, HSV, Lab, and Grayscale for specific analysis needs.2. Noise Reduction & BlurringLinear Filters: Mean and Gaussian blurring to remove high-frequency noise.Non-Linear Filters: Median filtering (excellent for salt-and-pepper noise) and Bilateral filtering (edge-preserving smoothing).3. Feature Detection (Edge & Corner)Gradient Operators: Sobel, Prewitt, and Roberts Cross.Advanced Detection: Canny Edge Detection and Harris Corner Detection for structural analysis.4. Morphological OperationsFundamental operations like Erosion and Dilation.Compound operations like Opening (noise removal) and Closing (hole filling).🛠️ Technologies UsedPython 3.x: Core programming language.OpenCV (Open Source Computer Vision Library): For optimized image processing functions.NumPy: For high-performance matrix and array manipulations.Matplotlib: For visualizing histograms and side-by-side comparisons.🔬 Mathematical FoundationsMost operations in this repo rely on Convolution, defined as:$$g(x,y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s,t) f(x+s, y+t)$$Where $f$ is the input image, $w$ is the kernel (mask), and $g$ is the output image.📂 Project StructurePlaintextImage-Processing/
├── assets/             # Input images for testing
├── src/                # Implementation source code
│   ├── preprocessing.py
│   ├── filters.py
│   └── morphology.py
├── notebooks/          # Interactive Jupyter Notebooks (.ipynb)
├── output/             # Processed results
├── requirements.txt    # Library dependencies
└── README.md
⚙️ Installation & SetupClone the repository:Bashgit clone https://github.com/Chainchigit/Image-Processing.git
cd Image-Processing
Create a virtual environment (Recommended):Bashpython -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate  # Windows
Install dependencies:Bashpip install -r requirements.txt
💻 Usage ExamplesTo run a specific processing script, use:Pythonimport cv2
from src.filters import apply_gaussian

# Load image
img = cv2.imread('assets/sample.jpg')

# Apply 5x5 Gaussian Blur
blurred = apply_gaussian(img, kernel_size=5)

# Show result
cv2.imshow('Result', blurred)
cv2.waitKey(0)
📈 Project Roadmap[ ] Add Frequency Domain processing (Fast Fourier Transform).[ ] Implement simple Object Tracking (Centroid-based).[ ] Add a Batch Processing script for large datasets.🤝 ContributingContributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.Developed by: Chainchigit
