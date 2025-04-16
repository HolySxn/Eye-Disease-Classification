# Eye Disease Classification with Deep Learning (ODIR-5K)

This project focuses on the classification of eye diseases based on retinal images using a Convolutional Neural Network (CNN) with residual connections. The model was trained and evaluated on the **ODIR-5K** dataset and achieved strong performance across multiple classes.

---

## 📦 Dataset

- **ODIR-5K**: A large-scale dataset of fundus (retinal) images labeled with multiple eye disease categories.
- Each image is labeled with one or more diseases. In this project, we used the **first label** for single-label classification.

---

## 🧪 Project Features

- **Custom CNN with Residual Blocks** for deep feature extraction.
- **Image Augmentation** using `ImageDataGenerator` to improve model generalization.
- **Oversampling** with `RandomOverSampler` to handle class imbalance.
- **Label Encoding** for categorical disease classes.
- **Performance Evaluation** using accuracy, precision, recall, F1-score.
- **Visualization** of training metrics and classification results.

---

## 🛠 Tech Stack

- Python, NumPy, Pandas
- TensorFlow / Keras
- Scikit-learn
- imbalanced-learn (RandomOverSampler)
- Matplotlib, Seaborn
- PIL

---

## 🧠 Model Architecture

- Input: 224x224 RGB fundus images
- CNN layers with Batch Normalization and ReLU activation
- Residual connections (ResNet-style blocks)
- MaxPooling for downsampling
- Dense layers with Dropout for regularization
- Output: Softmax layer with 8 classes

---

## 🔄 Image Augmentation

Performed image augmentation using `ImageDataGenerator`, including:
- Rotation
- Width/height shift
- Shear and zoom
- Horizontal flipping
- Rescaling

This helped to reduce overfitting and improve robustness to variations in image capture.

---

## 📊 Dataset Stats

- **Total images loaded**: 8,972
- **Post-oversampling**:
  - **Training set**: 7,249 images
  - **Validation set**: 1,813 images
  - **Test set**: 2,266 images

---

## ✅ Results (on test set)

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| N     | 0.81      | 0.92   | 0.86     |
| D     | 0.96      | 0.99   | 0.98     |
| G     | 0.47      | 0.46   | 0.47     |
| C     | 0.89      | 0.95   | 0.92     |
| A     | 0.86      | 0.96   | 0.91     |
| H     | 0.99      | 1.00   | 0.99     |
| M     | 0.49      | 0.52   | 0.50     |
| O     | 0.59      | 0.36   | 0.45     |

- **Overall Accuracy**: **76%**
- **Macro Avg F1-score**: **0.76**

---

## 📈 Training Curves

- Includes plots of training/validation **accuracy** and **loss** over epochs.
- Implemented `EarlyStopping` and `ReduceLROnPlateau` for stable convergence.

---
