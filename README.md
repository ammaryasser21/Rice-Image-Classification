

---

# Rice Image Classification Project

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
  - [Pretrained Model](#pretrained-model)
  - [Custom CNN Model](#custom-cnn-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Confusion Matrices](#confusion-matrices)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)

## Overview
This project focuses on rice image classification using machine learning models. The goal is to classify rice types into five categories: **Arborio, Basmati, Ipsala, Jasmine, and Karacadag**. Two models were trained and evaluated:
1. **Pretrained Model** (ResNet-50).
2. **Custom Convolutional Neural Network (CNN)**.

## Dataset
- **Source**: [Rice Images Dataset](https://www.kaggle.com/datasets/mbsoroush/rice-images-dataset).
- **Size**: 
  - Training set: 60,000 images.
  - Validation set: 7,500 images.
  - Test set: 7,500 images.
- **Classes**: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.

## Preprocessing
The images were preprocessed to ensure compatibility with the models:
- Resized images to fit model input dimensions.
- Normalized pixel values to improve model training.
- Split dataset into training, validation, and test sets.

## Models

### Pretrained Model
- **Architecture**: ResNet-50 without the top layer (transfer learning).
- **Training Details**:
  - Epochs: 2.
  - Accuracy: 
    - Training: 92.37%.
    - Validation: 96.81%.
  - Loss: 
    - Training: 0.2143.
    - Validation: 0.1124.

### Custom CNN Model
- **Architecture**: Custom-designed CNN for rice image classification.
- **Training Details**:
  - Epochs: 2.
  - Accuracy: 
    - Training: 96.31%.
    - Validation: 97.36%.
  - Loss: 
    - Training: 0.1187.
    - Validation: 0.1038.

## Evaluation
### Pretrained Model
- **Test Accuracy**: 95.94%.
- **Classification Report**:
  - Precision: 96%.
  - Recall: 96%.
  - F1-Score: 96%.

### Custom CNN Model
- **Test Accuracy**: 95.68%.
- **Classification Report**:
  - Precision: 97%.
  - Recall: 97%.
  - F1-Score: 97%.

## Results
- Both models performed exceptionally well in classifying rice images.
- The custom CNN model slightly outperformed the pretrained ResNet-50 model in accuracy and f1-score.

## Confusion Matrices

### Pretrained Model
![image](https://github.com/user-attachments/assets/9c99739d-d701-4b09-b1f4-d25272172a00)


### Custom CNN Model
![image](https://github.com/user-attachments/assets/26cf189f-3456-4c3a-aef0-f31bce6db71d)


## How to Use
1. Clone this repository.
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script.
   ```bash
   python data_science.py
   ```

## Dependencies
- Python 3.10
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

---

