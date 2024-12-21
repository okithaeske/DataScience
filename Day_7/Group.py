import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Set page configuration
st.set_page_config(page_title="Student Depression Analysis", layout="wide")

# Streamlit app title
st.title("Student Depression Analysis")

# Sidebar for file upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
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
        fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='Viridis', title="Correlation Matrix")
        st.plotly_chart(fig)

    # Boxplots
    st.subheader("Boxplots: Numerical Features by Depression")
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_features = [feature for feature in numerical_features if feature not in ['Gender', 'Depression']]

    selected_boxplot_feature = st.selectbox("Select a feature for boxplot", numerical_features)
    if selected_boxplot_feature:
        fig = px.box(df, x='Depression', y=selected_boxplot_feature, color='Depression', title=f'Depression vs {selected_boxplot_feature}')
        st.plotly_chart(fig)

    # Bar chart for men and women with depression
    st.subheader("Comparison of Men and Women with Depression")
    total_depression_count = len(df[df['Depression'] == 1])
    men_depression_count = len(df[(df['Gender'] == 1) & (df['Depression'] == 1)])
    women_depression_count = len(df[(df['Gender'] == 0) & (df['Depression'] == 1)])

    st.write("Total count of people with depression:", total_depression_count)
    st.write("Men with depression:", men_depression_count)
    st.write("Women with depression:", women_depression_count)

    categories = ['Men with Depression', 'Women with Depression']
    counts = [men_depression_count, women_depression_count]

    fig = px.bar(x=categories, y=counts, text=counts, title="Comparison of Men and Women with Depression", labels={'x': 'Category', 'y': 'Count'}, color=categories)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)

    # Model training and prediction
    st.subheader("Train a Classification Model")
    target_column = st.selectbox("Select the target column", df.columns.tolist(), index=df.columns.get_loc('Depression'))
    feature_columns = st.multiselect("Select feature columns", df.columns.tolist(), default=[col for col in df.columns if col != target_column])

    if feature_columns and target_column:
        X = df[feature_columns]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model_results = []

        # Random Forest Classifier
        st.write("Training Random Forest Classifier...")
        rf_progress = st.progress(0)
        rf_model = RandomForestClassifier(random_state=42)
        start_time = time.time()
        for i in range(1, 101):
            time.sleep(0.01)
            rf_progress.progress(i)
        rf_model.fit(X_train, y_train)
        rf_training_time = time.time() - start_time
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        model_results.append(("Random Forest", rf_accuracy, rf_pred, rf_training_time))

        # MLP Classifier
        st.write("Training MLP Classifier...")
        mlp_progress = st.progress(0)
        mlp_model = MLPClassifier(random_state=42, max_iter=1000)
        start_time = time.time()
        for i in range(1, 101):
            time.sleep(0.01)
            mlp_progress.progress(i)
        mlp_model.fit(X_train, y_train)
        mlp_training_time = time.time() - start_time
        mlp_pred = mlp_model.predict(X_test)
        mlp_accuracy = accuracy_score(y_test, mlp_pred)
        model_results.append(("MLP Classifier", mlp_accuracy, mlp_pred, mlp_training_time))

        # Support Vector Machine Classifier
        st.write("Training SVM Classifier...")
        svm_progress = st.progress(0)
        svm_model = SVC(random_state=42)
        start_time = time.time()
        for i in range(1, 101):
            time.sleep(0.01)
            svm_progress.progress(i)
        svm_model.fit(X_train, y_train)
        svm_training_time = time.time() - start_time
        svm_pred = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        model_results.append(("SVM Classifier", svm_accuracy, svm_pred, svm_training_time))

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
