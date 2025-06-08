#  Breast Cancer Prediction using Machine Learning

This project focuses on predicting breast cancer diagnoses (Benign vs Malignant) using several classification models. Given the **discrete** nature of the target variable and the **medical importance** of prediction quality, we prioritize **precision** and **F1-score** over simple accuracy.

---

##  Dataset

- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Target Variable**: `Diagnosis` (0 = Benign, 1 = Malignant)
- **Features**: 30 numerical attributes derived from cell nuclei images

---

##  Why Classification Models (not Regression)?

Although initial experiments were made with regression models (linear, multiple, polynomial), they were dropped because:

- The output is **categorical** (not continuous).
- Regression is not suitable for discrete binary classification tasks.
- The goal is not just to predict a value but to **classify correctly** with a strong emphasis on **minimizing false negatives**.

Thus, the following classification models were adopted instead:

---

##  Models Used

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM)**
4. **Neural Network (PyTorch)**  
   - Hidden layers: [64 → 32 → 1]  
   - Activations: ReLU and Sigmoid (compared experimentally)  
   - Loss: Binary Cross-Entropy  
   - Regularization: Dropout (0.3)

---

## ⚙ Preprocessing Steps

- Standardized the input features using `StandardScaler`
- Stratified splitting: Train (80%), Test (20%), Validation (optional)
- Converted labels to binary (0 for Benign, 1 for Malignant)

---

##  Evaluation Metrics

Given that this is a **medical** dataset where **false negatives** are dangerous (i.e., saying a malignant case is benign), we prioritize:

- **Precision**: How many predicted positives are truly positive.
- **Recall**: Sensitivity — how many actual positives were caught.
- **F1-score**: Harmonic mean of precision and recall.

All models are evaluated using:

- **Confusion Matrix**
- **Accuracy, Precision, Recall, F1-score**
- **Classification Report**
- **Heatmap visualizations (Seaborn)**

---

## 📈 Results Summary

| Model              | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| ✅        | ✅         | ✅     | ✅        |
| KNN                | ✅        | ✅         | ✅     | ✅        |
| SVM                | ✅        | ✅         | ✅     | ✅        |
| Neural Network     | ✅        | ✅         | ✅     | ✅        |

> ✅ All models were tuned to increase precision and F1-score, especially important due to the **imbalanced** nature of the data.
