# Landmark Detection using VGG19 (Deep Learning)

This project implements a **landmark image classification system** using a
deep learning approach. A **VGG19-based Convolutional Neural Network (CNN)**
is used to classify landmark images into their respective landmark IDs.

The implementation focuses on dataset preprocessing, model training,
validation, and evaluation using **Keras with TensorFlow backend**.

---

## Features

- Landmark image classification using CNN
- Automatic image path generation from image IDs
- Image preprocessing and normalization
- Label encoding for landmark classes
- VGG19-based model with transfer learning
- Training using mini-batch processing
- Model evaluation with correct and incorrect predictions tracking
- Visualization of random sample images

---

## Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- scikit-learn
- VGG19 (ImageNet pretrained model)

---

## Project Structure

Landmark-Detection/
│
├── LandmarkDetection.py # Main training and evaluation script
├── README.md # Project documentation
├── .gitignore


---

## Dataset Description

- Dataset information is provided through a CSV file containing:
  - `id` → image identifier
  - `landmark_id` → class label
- Images are stored in a **nested directory structure** derived from
  the first three characters of the image ID.

### Example image path format

- images/0/0/0/000abcd.jpg

> **Dataset files are not included in this repository due to size constraints.**

---

## Model Architecture

- Base Model: **VGG19** (`include_top=False`, ImageNet weights)
- Global Average Pooling layer
- Dropout layer for regularization
- Dense output layer with Softmax activation
- Loss Function: `sparse_categorical_crossentropy`
- Optimizer: `RMSprop`

---

## How to Run the Project

### 1️⃣ Install dependencies
```bash
- pip install tensorflow keras numpy pandas opencv-python matplotlib scikit-learn pillow

2️⃣ Prepare dataset

- Place train.csv in the project root

- Place the image dataset inside the images/ folder

3️⃣ Run the script
python LandmarkDetection.py
