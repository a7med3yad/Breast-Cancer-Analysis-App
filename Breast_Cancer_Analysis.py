import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

st.title("Breast Cancer Classification Models Comparison")

# Define feature names explicitly
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Load data
uploaded_file = st.file_uploader("Upload a CSV file (optional)")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    st.info("Using default breast cancer dataset (wdbc.csv)")
    data = pd.read_csv("wdbc.csv")

# Select only the features columns from the dataset
X = data[feature_names]
y = data['diagnosis'].map({'M': 1, 'B': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Rest of your code (models, training, evaluation) remains the same...

# ================================
# 1. Logistic Regression
# ================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# ================================
# 2. KNN
# ================================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# ================================
# 3. SVM
# ================================
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# ================================
# 4. Neural Network (PyTorch)
# ================================
class BreastCancerNN(nn.Module):
    def __init__(self, input_size):
        super(BreastCancerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return self.sigmoid(out)

# Convert data to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Initialize and train the model
input_size = X_train.shape[1]
model = BreastCancerNN(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

with st.spinner("Training neural network model..."):
    for epoch in range(200):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Predict with the neural network
model.eval()
with torch.no_grad():
    y_pred_nn_proba = model(X_test_tensor).numpy().flatten()
    y_pred_nn = (y_pred_nn_proba >= 0.5).astype(int)

# Evaluate all models
models = {
    "Logistic Regression": y_pred_lr,
    "KNN": y_pred_knn,
    "SVM": y_pred_svm,
    "Neural Network": y_pred_nn
}

results = []

for name, preds in models.items():
    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results.append({
        "Model": name,
        "Accuracy": f"{acc:.4f}",
        "Recall": f"{recall:.4f}",
        "Precision": f"{prec:.4f}",
        "F1 Score": f"{f1:.4f}"
    })

st.subheader("Model Evaluation Metrics")
st.table(results)
