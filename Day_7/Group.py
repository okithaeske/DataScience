import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        st.pyplot(plt)

    # Boxplots
    st.subheader("Boxplots: Numerical Features by Depression")
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_features = [feature for feature in numerical_features if feature not in ['Gender', 'Depression']]

    selected_boxplot_feature = st.selectbox("Select a feature for boxplot", numerical_features)
    if selected_boxplot_feature:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Depression', y=selected_boxplot_feature, data=df, palette='coolwarm')
        plt.title(f'Depression vs {selected_boxplot_feature}')
        st.pyplot(plt)

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

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=categories, y=counts, palette='coolwarm')
    for i, value in enumerate(counts):
        ax.text(i, value / 2, str(value), ha='center', va='center', fontsize=12, color='black')
    plt.xlabel('Gender with Depression')
    plt.ylabel('Count')
    plt.title('Comparison of Men and Women with Depression')
    st.pyplot(plt)

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
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        model_results.append(("Random Forest", rf_accuracy, rf_pred))

        # MLP Classifier
        mlp_model = MLPClassifier(random_state=42, max_iter=1000)
        mlp_model.fit(X_train, y_train)
        mlp_pred = mlp_model.predict(X_test)
        mlp_accuracy = accuracy_score(y_test, mlp_pred)
        model_results.append(("MLP Classifier", mlp_accuracy, mlp_pred))

        # Support Vector Machine Classifier
        svm_model = SVC(random_state=42)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        model_results.append(("SVM Classifier", svm_accuracy, svm_pred))

        # Find the best model
        best_model = max(model_results, key=lambda x: x[1])
        st.write(f"## Best Model: {best_model[0]}")
        st.write("Accuracy:", best_model[1])
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, best_model[2]))
        st.write("Classification Report:")
        st.text(classification_report(y_test, best_model[2]))

    # Download preprocessed dataset
    st.subheader("Download Preprocessed Dataset")
    csv = df.to_csv(index=False)
    st.download_button("Download as CSV", csv, "preprocessed_dataset.csv", "text/csv")

else:
    st.write("Please upload a CSV file to proceed.")
