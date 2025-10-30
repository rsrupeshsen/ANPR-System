# Automatic Number Plate Recognition (ANPR) System

![Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)

A high-performance ANPR system built with a two-stage deep learning pipeline. This project uses a custom-trained YOLOv8 model for robust license plate detection and EasyOCR for accurate text extraction.

---

## 🚀 How It Works

This project uses a hybrid, two-model pipeline for maximum accuracy:

1.  **Stage 1: Plate Detection (YOLOv8)**
    * A custom **YOLOv8n** object detection model is trained on a dataset of car images to find and crop the license plate from the main image.
    * This custom model (`models/best.pt`) is lightweight and fast.

2.  **Stage 2: Text Recognition (EasyOCR)**
    * The cropped plate image from Stage 1 is passed to an **EasyOCR** model.
    * EasyOCR is a pre-trained OCR model that handles the difficult tasks of segmenting (separating) and recognizing the individual characters, even on plates with tight spacing or different fonts.

---

## 📸 Results & Demo

Here is the pipeline running on two test images. The system first finds the plate (red box) and then extracts the text.

*(You can upload your result images to the GitHub repo later and add the links here)*

| Test Image 1 (Scorpio) | Test Image 2 (UK Plate) |
| :---: | :---: |
| **Prediction: `MI2RNA005`** | **Prediction: `UK17C09`** |
| ![Scorpio Result](test_images/scorpio_result.png) | ![UK Plate Result](test_images/uk_plate_result.png) |


---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/ANPR-System.git](https://github.com/YOUR_USERNAME/ANPR-System.git)
    cd ANPR-System
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚡ How to Run

To run the detector on a new image, use the `detect.py` script from your command line.

```bash
python detect.py --image "test_images/scorpio_img.jpeg"
