# 🧠 Breast Cancer Analysis Dashboard

<img src="https://upload.wikimedia.org/wikipedia/commons/7/7e/Breast_cancer_awareness_pink_ribbon.svg" alt="Breast Cancer Awareness" width="120" style="margin-bottom: 20px;">

This Streamlit dashboard provides a comprehensive analysis of a **breast cancer dataset** using various machine learning techniques and data visualization tools.

---

## 🚀 Features

### 📋 Data Overview
- Displays basic statistics about the dataset: head, shape, missing values, data types, and summary stats.

### 📈 Feature Distributions
- Interactive histograms for numerical features to understand their distributions.

### 🔍 Pairplot
- Visualizes relationships between selected features using a pairplot, colored by diagnosis.

### 📊 Correlation Heatmap
- Shows a correlation heatmap to detect multicollinearity and relationships between features.

### 📉 Simple Linear Regression
- Performs simple linear regression using the feature most correlated with `radius_mean`.

### 📚 Multiple Linear Regression
- Uses all features to predict `perimeter_mean`.

### 🧮 Polynomial Regression
- Trains polynomial regression models of degrees 2 and 3 and compares their performance.

### 🧠 Logistic Regression
- Classifies tumors as **Benign (0)** or **Malignant (1)** using logistic regression.

### 📌 Summary
- Summarizes key takeaways from all the applied models.

---

## 📂 Dataset

The dashboard uses the **Wisconsin Breast Cancer Dataset** (`wdbc.csv`).  
By default, it loads the dataset from `.streamlit/wdbc.csv`, but you can upload your own CSV file from the sidebar.

---

## 📦 Libraries Used

- `streamlit`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`

---

## 🛠 How to Run

```bash
streamlit run app.py
