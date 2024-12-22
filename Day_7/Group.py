import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Set page configuration
st.set_page_config(page_title="General Dataset Analysis and Classification", layout="wide")

# Streamlit app title
st.title("General Dataset Analysis and Classification")

# Sidebar for file upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Handle missing values
    st.sidebar.subheader("Missing Value Handling")
    missing_strategy = st.sidebar.selectbox("Select strategy for handling missing values", ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"])
    if missing_strategy == "Drop Rows":
        df = df.dropna()
    elif missing_strategy == "Fill with Mean":
        df = df.fillna(df.mean())
    elif missing_strategy == "Fill with Median":
        df = df.fillna(df.median())
    elif missing_strategy == "Fill with Mode":
        df = df.fillna(df.mode().iloc[0])

    st.write("### Dataset After Missing Value Handling")
    st.dataframe(df.head())

    # Preprocess the dataset
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])

    st.write("### Preprocessed Dataset")
    st.dataframe(df.head())

    # Data summary
    st.subheader("Data Summary")
    st.write(df.describe())

    # Correlation matrix
    st.subheader("Correlation Matrix")
    selected_columns = st.multiselect("Select columns for correlation matrix", df.columns.tolist(), default=df.columns.tolist())
    if selected_columns:
        correlation_matrix = df[selected_columns].corr()
        fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='Viridis', title="Correlation Matrix", width=800, height=800)
        st.plotly_chart(fig)

    # Boxplots
    st.subheader("Boxplots: Numerical Features")
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

    selected_boxplot_feature = st.selectbox("Select a feature for boxplot", numerical_features)
    if selected_boxplot_feature:
        fig = px.box(df, y=selected_boxplot_feature, title=f'Distribution of {selected_boxplot_feature}')
        st.plotly_chart(fig)

    # Model training and prediction
    st.subheader("Train a Classification Model")
    target_column = st.selectbox("Select the target column", df.columns.tolist())
    feature_columns = st.multiselect("Select feature columns", df.columns.tolist(), default=[col for col in df.columns if col != target_column])

    if feature_columns and target_column:
        X = df[feature_columns]
        y = df[target_column]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model_results = []

        classifiers = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "MLP Classifier": MLPClassifier(random_state=42, max_iter=1000),
            "SVM Classifier": SVC(random_state=42, probability=True),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42)
        }

        for model_name, model in classifiers.items():
            st.write(f"Training {model_name}...")
            progress = st.progress(0)
            start_time = time.time()
            for i in range(1, 101):
                time.sleep(0.01)
                progress.progress(i)
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            model_results.append((model_name, accuracy, pred, training_time))

        # Find the best model
        best_model = max(model_results, key=lambda x: x[1])
        st.write(f"## Best Model: {best_model[0]}")
        st.write("Accuracy:", best_model[1])
        st.write("Training Time (seconds):", round(best_model[3], 2))

        # Visualize accuracies of all models
        st.subheader("Model Comparison")
        model_names = [result[0] for result in model_results]
        accuracies = [result[1] for result in model_results]
        training_times = [result[3] for result in model_results]

        comparison_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracies,
            'Training Time (s)': training_times
        })

        st.dataframe(comparison_df)

        fig = px.bar(comparison_df, x='Model', y='Accuracy', color='Model', text='Accuracy', title='Model Accuracy Comparison')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)

        # Confusion Matrix Heatmap
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, best_model[2])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis', title=f'Confusion Matrix - {best_model[0]}', labels={'x': 'Predicted', 'y': 'Actual'})
        st.plotly_chart(fig)

        st.write("Classification Report:")
        st.text(classification_report(y_test, best_model[2]))

    # Download preprocessed dataset
    st.subheader("Download Preprocessed Dataset")
    csv = df.to_csv(index=False)
    st.download_button("Download as CSV", csv, "preprocessed_dataset.csv", "text/csv")

else:
    st.write("Please upload a CSV file to proceed.")
