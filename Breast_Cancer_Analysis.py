# Breast Cancer ML App with Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# Load Data Function
def load_data():
    data = pd.read_csv('wdbc.csv')
    data.drop('id', axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def preprocess_data(data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Neural Network Class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, activation='relu'):
        super(NeuralNetwork, self).__init__()
        act = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), act,
            nn.Linear(64, 32), act,
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Main App
st.title('Breast Cancer Diagnosis ML App')
data = load_data()
st.subheader('Raw Data')
st.dataframe(data.head())

# Visualization
if st.checkbox('Show EDA Plots'):
    st.subheader('Correlation Heatmap')
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), cmap='coolwarm')
    st.pyplot(plt)

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(data)

# Model Selection
model_choice = st.selectbox('Select Model', ['Logistic Regression', 'KNN', 'SVM', 'Neural Network'])

if model_choice == 'Logistic Regression':
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

elif model_choice == 'KNN':
    k = st.slider('K value', 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

elif model_choice == 'SVM':
    kernel = st.selectbox('Kernel', ['linear', 'rbf', 'poly'])
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

elif model_choice == 'Neural Network':
    act_func = st.selectbox('Activation Function', ['relu', 'sigmoid'])
    model = NeuralNetwork(X_train.shape[1], activation=act_func)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    for epoch in range(100):
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        preds = model(X_test_tensor).round().numpy().astype(int).flatten()

# Evaluation
st.subheader('Model Performance')
st.write('Accuracy:', accuracy_score(y_test, preds))
st.text('Classification Report')
st.text(classification_report(y_test, preds))

st.text('Confusion Matrix')
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', ax=ax)
st.pyplot(fig)
