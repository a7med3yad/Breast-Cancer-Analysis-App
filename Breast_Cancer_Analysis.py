import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

# --- App Config ---
st.set_page_config(page_title="Breast Cancer Analysis", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar Navigation ---
st.sidebar.title("🔎 Navigation")
section = st.sidebar.radio("Go to", [
    "📋 Data Overview",
    "📈 Histograms",
    "🔍 Pairplot",
    "📊 Correlation Heatmap",
    "📉 Simple Linear Regression",
    "📚 Multiple Linear Regression",
    "🧮 Polynomial Regression",
    "🧠 Logistic Regression",
    "📌 Summary"
])

# --- Load Data ---
@st.cache_data
def load_data():
    columns = [
        'ID', 'Diagnosis', 
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    file_path = "C:\\Users\\hazem\\OneDrive\\Documents\\archive\\New folder\\wdbc.csv"
    df = pd.read_csv(file_path, header=None, names=columns)
    df.drop('ID', axis=1, inplace=True)
    df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})
    return df

df = load_data()
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------ Sections ------------------------

if section == "📋 Data Overview":
    st.title("📋 Data Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing values:", df.isnull().sum().sum())
    st.write("Data types:", df.dtypes)
    st.write(df.describe())

elif section == "📈 Histograms":
    st.title("📈 Feature Distributions")
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 20))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.histplot(df[col], ax=axes[i], bins=20, kde=True, edgecolor='black')
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

elif section == "🔍 Pairplot":
    st.title("🔍 Pairplot")
    selected = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'Diagnosis']
    fig = sns.pairplot(df[selected], hue='Diagnosis', palette='coolwarm')
    st.pyplot(fig)

elif section == "📊 Correlation Heatmap":
    st.title("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap", fontsize=18)
    st.pyplot(fig)

elif section == "📉 Simple Linear Regression":
    st.title("📉 Simple Linear Regression")
    corr = df.corr()['radius_mean'].drop('radius_mean')
    top_feature = corr.abs().idxmax()
    st.write(f"Top correlated feature: `{top_feature}`")

    X_simple = df[[top_feature]].values
    y_simple = df['radius_mean'].values

    X_simple_scaled = scaler.fit_transform(X_simple)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple_scaled, y_simple, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train_s, y_train_s)
    y_pred_s = model.predict(X_test_s)

    st.write(f"MSE: {mean_squared_error(y_test_s, y_pred_s):.4f}")
    st.write(f"R²: {r2_score(y_test_s, y_pred_s):.4f}")

    fig, ax = plt.subplots()
    ax.scatter(X_test_s, y_test_s, color='blue', label='Actual')
    ax.plot(X_test_s, y_pred_s, color='red', label='Prediction')
    ax.set_xlabel(top_feature)
    ax.set_ylabel("radius_mean")
    ax.legend()
    st.pyplot(fig)

elif section == "📚 Multiple Linear Regression":
    st.title("📚 Multiple Linear Regression")
    A = df.drop(columns=['perimeter_mean'])
    B = df['perimeter_mean']
    A_scaled = scaler.fit_transform(A)
    B_scaled = scaler.fit_transform(B.values.reshape(-1, 1))

    A_train, A_test, B_train, B_test = train_test_split(A_scaled, B_scaled, test_size=0.2, random_state=42)

    mlr = LinearRegression()
    mlr.fit(A_train, B_train)
    B_pred_test = mlr.predict(A_test)

    st.write("R²:", r2_score(B_test, B_pred_test))
    st.write("MSE:", mean_squared_error(B_test, B_pred_test))

    fig, ax = plt.subplots()
    ax.scatter(range(len(B_test)), B_test, color='blue', label='Actual')
    ax.scatter(range(len(B_test)), B_pred_test, color='red', label='Predicted', alpha=0.6)
    ax.legend()
    st.pyplot(fig)

elif section == "🧮 Polynomial Regression":
    st.title("🧮 Polynomial Regression")
    A = df.drop(columns=['perimeter_mean'])
    B = df['perimeter_mean']
    A_scaled = scaler.fit_transform(A)
    A_train, A_test, B_train, B_test = train_test_split(A_scaled, B, test_size=0.2, random_state=42)

    degrees = [2, 3]
    for degree in degrees:
        poly = PolynomialFeatures(degree)
        A_poly_train = poly.fit_transform(A_train)
        A_poly_test = poly.transform(A_test)

        model_poly = LinearRegression()
        model_poly.fit(A_poly_train, B_train)
        B_pred_test = model_poly.predict(A_poly_test)

        st.subheader(f"Degree {degree}")
        st.write("R²:", r2_score(B_test, B_pred_test))
        st.write("MSE:", mean_squared_error(B_test, B_pred_test))

        fig, ax = plt.subplots()
        ax.scatter(B_test, B_pred_test, alpha=0.6)
        ax.plot([min(B_test), max(B_test)], [min(B_test), max(B_test)], '--', color='red')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Polynomial Regression (degree {degree})")
        st.pyplot(fig)

elif section == "🧠 Logistic Regression":
    st.title("🧠 Logistic Regression")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

elif section == "📌 Summary":
    st.title("📌 Summary")
    st.markdown("""
    - ✅ The best-performing feature for simple linear regression was **`perimeter_mean`**.
    - 📈 Polynomial regression of degree **2** gave the best performance.
    - 🧪 Logistic Regression performs well in classifying tumors into **Benign** and **Malignant**.
    """)

