import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Breast Cancer Classification App", layout="wide")

# Title and description
st.title("Breast Cancer Classification App")
st.markdown("""
This app classifies breast cancer tumors as benign or malignant using machine learning models.
Upload a WDBC dataset (CSV) to proceed.
""")

# Function to load and preprocess data
@st.cache_data
def load_data(uploaded_file):
    columns = [
        'ID', 'Diagnosis', 
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    df = pd.read_csv(uploaded_file, header=None, names=columns)
    df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})
    df.drop(columns=["ID"], inplace=True)
    return df

# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, activation='relu'):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.activation(self.layer2(x))
        x = self.dropout(x)
        x = self.output_activation(self.layer3(x))
        return x

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    return fig

# Function to plot ROC curves
def plot_roc_curves(roc_data):
    fig, ax = plt.subplots(figsize=(10, 8))
    for model_name, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    return fig

# File uploader
uploaded_file = st.file_uploader("Upload WDBC dataset (CSV)", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Display dataset info
    st.subheader("Dataset Overview")
    st.write("First 5 rows:")
    st.dataframe(df.head())
    st.write(f"Dataset shape: {df.shape}")
    st.write("Missing values:")
    st.write(df.isnull().sum())
    st.write("Data types:")
    st.write(df.dtypes)

    # Data preprocessing
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']  # Already numeric (0 for 'B', 1 for 'M')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    st.write(f"Training set size: {X_train.shape}")
    st.write(f"Validation set size: {X_val.shape}")
    st.write(f"Test set size: {X_test.shape}")

    # Model selection
    st.subheader("Select Models to Train")
    models_to_train = st.multiselect(
        "Choose models:",
        ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)", "Neural Network"],
        default=["Logistic Regression"]
    )

    results = {}
    roc_data = {}

    if st.button("Train Models"):
        progress_bar = st.progress(0)
        total_steps = len(models_to_train) + (2 if "Support Vector Machine (SVM)" in models_to_train else 0)
        current_step = 0

        # Logistic Regression
        if "Logistic Regression" in models_to_train:
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(max_iter=10000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            results['Logistic Regression'] = {
                'accuracy': acc,
                'confusion_matrix': cm,
                'classification_report': report
            }
            roc_data['Logistic Regression'] = (fpr, tpr, roc_auc)
            
            st.write(f"Accuracy: {acc:.4f}")
            st.pyplot(plot_confusion_matrix(cm, "Confusion Matrix - Logistic Regression"))
            st.write("Classification Report:")
            st.dataframe(pd.DataFrame(report).transpose())
            current_step += 1
            progress_bar.progress(current_step / total_steps)

        # KNN
        if "K-Nearest Neighbors (KNN)" in models_to_train:
            st.subheader("K-Nearest Neighbors Results")
            k_values = range(1, 21)
            val_accuracies = []
            best_k = 1
            best_val_accuracy = 0
            
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                val_pred = knn.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_pred)
                val_accuracies.append(val_accuracy)
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_k = k
            
            # Train final model
            knn_final = KNeighborsClassifier(n_neighbors=best_k)
            knn_final.fit(X_train, y_train)
            y_pred = knn_final.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            y_pred_proba = knn_final.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            results['KNN'] = {
                'accuracy': acc,
                'confusion_matrix': cm,
                'classification_report': report
            }
            roc_data['KNN'] = (fpr, tpr, roc_auc)
            
            st.write(f"Best K: {best_k}")
            st.write(f"Test Accuracy: {acc:.4f}")
            st.pyplot(plot_confusion_matrix(cm, f"Confusion Matrix - KNN (K={best_k})"))
            st.write("Classification Report:")
            st.dataframe(pd.DataFrame(report).transpose())
            current_step += 1
            progress_bar.progress(current_step / total_steps)

        # SVM
        if "Support Vector Machine (SVM)" in models_to_train:
            st.subheader("Support Vector Machine Results")
            kernels = ['linear', 'rbf']  # Removed 'poly' to reduce runtime
            for kernel in kernels:
                svm = SVC(kernel=kernel, random_state=42, probability=True)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                y_pred_proba = svm.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                results[f'SVM {kernel}'] = {
                    'accuracy': acc,
                    'confusion_matrix': cm,
                    'classification_report': report
                }
                roc_data[f'SVM {kernel}'] = (fpr, tpr, roc_auc)
                
                st.write(f"SVM with {kernel} kernel:")
                st.write(f"Accuracy: {acc:.4f}")
                st.pyplot(plot_confusion_matrix(cm, f"Confusion Matrix - SVM {kernel}"))
                st.write("Classification Report:")
                st.dataframe(pd.DataFrame(report).transpose())
                current_step += 1
                progress_bar.progress(current_step / total_steps)

        # Neural Network
        if "Neural Network" in models_to_train:
            st.subheader("Neural Network Results")
            try:
                # Convert to PyTorch tensors
                X_train_torch = torch.FloatTensor(X_train)
                y_train_torch = torch.FloatTensor(y_train.values).reshape(-1, 1)
                X_val_torch = torch.FloatTensor(X_val)
                y_val_torch = torch.FloatTensor(y_val.values).reshape(-1, 1)
                X_test_torch = torch.FloatTensor(X_test)
                y_test_torch = torch.FloatTensor(y_test.values).reshape(-1, 1)
                
                activation = 'relu'
                model = NeuralNetwork(input_size=X_train.shape[1], activation=activation)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Training loop
                epochs = 100
                batch_size = 32
                train_losses = []
                val_losses = []
                
                for epoch in range(epochs):
                    model.train()
                    for i in range(0, len(X_train_torch), batch_size):
                        batch_X = X_train_torch[i:i+batch_size]
                        batch_y = y_train_torch[i:i+batch_size]
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                    
                    model.eval()
                    with torch.no_grad():
                        train_outputs = model(X_train_torch)
                        train_loss = criterion(train_outputs, y_train_torch)
                        val_outputs = model(X_val_torch)
                        val_loss = criterion(val_outputs, y_val_torch)
                        train_losses.append(train_loss.item())
                        val_losses.append(val_loss.item())
                
                # Predict on test set
                model.eval()
                with torch.no_grad():
                    y_pred_proba = model(X_test_torch).cpu().numpy()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                results[f'Neural Network {activation}'] = {
                    'accuracy': accuracy,
                    'confusion_matrix': cm,
                    'classification_report': report
                }
                roc_data[f'Neural Network {activation}'] = (fpr, tpr, roc_auc)
                
                st.write(f"Neural Network with {activation} activation:")
                st.write(f"Accuracy: {accuracy:.4f}")
                st.pyplot(plot_confusion_matrix(cm, f"Confusion Matrix - NN {activation}"))
                st.write("Classification Report:")
                st.dataframe(pd.DataFrame(report).transpose())
                
                # Plot loss curves
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(train_losses, label='Training Loss')
                ax.plot(val_losses, label='Validation Loss')
                ax.set_title(f'Loss Curve - {activation} Activation')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            except Exception as e:
                st.error(f"Neural Network training failed: {str(e)}")
                st.info("Ensure PyTorch is installed and compatible with your system. Try reducing epochs or batch size if resources are limited.")

        # Display ROC curves
        if roc_data:
            st.subheader("Model Comparison - ROC Curves")
            st.pyplot(plot_roc_curves(roc_data))
else:
    st.info("Please upload a CSV file to proceed.")
