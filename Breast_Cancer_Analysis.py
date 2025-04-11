import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Streamlit setup
st.title("Breast Cancer Data Analysis")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    
    # Column names
    columns = [
        'ID', 'Diagnosis', 
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    df.columns = columns

    # Drop ID column and encode Diagnosis
    df.drop('ID', axis=1, inplace=True)
    df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

    # Sidebar options
    st.sidebar.header("Options")
    task = st.sidebar.radio("Choose a task", ["Data Overview", "Visualizations", "Modeling"])

    # Data Overview
    if task == "Data Overview":
        st.subheader("Data Overview")
        st.write(df.head())
        st.write("Dataset shape:", df.shape)
        st.write("Missing values:", df.isnull().sum())

    # Visualizations
    elif task == "Visualizations":
        st.subheader("Visualizations")
        chart = st.selectbox("Choose a plot", ["Histograms", "Pairplot", "Correlation Heatmap", "Box Plot"])

        if chart == "Histograms":
            st.write("Feature Distributions")
            fig, ax = plt.subplots(figsize=(20, 15))
            df.hist(bins=20, figsize=(20, 15), edgecolor='black', ax=ax)
            st.pyplot(fig)

        elif chart == "Pairplot":
            sample_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'Diagnosis']
            st.write("Scatter Plots by Diagnosis")
            sns.pairplot(df[sample_features], hue='Diagnosis', palette='coolwarm')
            st.pyplot()

        elif chart == "Correlation Heatmap":
            st.write("Correlation Heatmap")
            corr = df.corr()
            fig, ax = plt.subplots(figsize=(18, 16))
            sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
            st.pyplot(fig)

        elif chart == "Box Plot":
            feature = st.selectbox("Choose a feature", df.columns[1:])
            st.write(f"Box Plot for {feature}")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=df[feature], color='skyblue', ax=ax)
            st.pyplot(fig)

    # Modeling
    elif task == "Modeling":
        st.subheader("Modeling")
        model_choice = st.selectbox("Choose a model", ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression", "Logistic Regression"])

        # Preprocessing
        X = df.drop('Diagnosis', axis=1)
        y = df['Diagnosis']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if model_choice == "Simple Linear Regression":
            correlations = df.corr()['radius_mean'].drop('radius_mean')
            top_feature = correlations.abs().idxmax()

            X_single = df[[top_feature]].values
            y_single = df['radius_mean'].values

            X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(X_single, y_single, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train_single, y_train_single)

            y_pred = model.predict(X_test_single)
            mse = mean_squared_error(y_test_single, y_pred)
            r2 = r2_score(y_test_single, y_pred)

            st.write(f"Top correlated feature: {top_feature}")
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"R² Score: {r2:.4f}")

            fig, ax = plt.subplots()
            ax.scatter(X_test_single, y_test_single, color='blue', label='Actual')
            ax.plot(X_test_single, y_pred, color='red', linewidth=2, label='Regression Line')
            ax.set_xlabel(top_feature)
            ax.set_ylabel("radius_mean")
            ax.legend()
            st.pyplot(fig)

        elif model_choice == "Multiple Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            st.write("Training R²:", r2_score(y_train, y_pred_train))
            st.write("Testing R²:", r2_score(y_test, y_pred_test))
            st.write("Training MSE:", mean_squared_error(y_train, y_pred_train))
            st.write("Testing MSE:", mean_squared_error(y_test, y_pred_test))

        elif model_choice == "Polynomial Regression":
            degree = st.slider("Select degree", 2, 4)
            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(X_train)
            X_poly_test = poly.transform(X_test)

            model = LinearRegression()
            model.fit(X_poly, y_train)

            y_poly_pred_train = model.predict(X_poly)
            y_poly_pred_test = model.predict(X_poly_test)

            st.write("Training R²:", r2_score(y_train, y_poly_pred_train))
            st.write("Testing R²:", r2_score(y_test, y_poly_pred_test))
            st.write("Training MSE:", mean_squared_error(y_train, y_poly_pred_train))
            st.write("Testing MSE:", mean_squared_error(y_test, y_poly_pred_test))

        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=10000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix")
            st.write(cm)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
            st.pyplot()
