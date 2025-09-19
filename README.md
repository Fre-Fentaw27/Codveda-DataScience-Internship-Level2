# Codveda-DataScience-Internship-Level2

Predictive Modeling(Regression),Classification with Logistic Regression, Clustering (Unsupervised Learning)

## 📌 Overview

This repository contains my solutions for the Level 2 Data Science Internship tasks, covering three fundamental machine learning areas: Regression, Classification, and Clustering. Each task demonstrates different aspects of data science workflow from data preprocessing to model evaluation.

## 📂 Project Structure

```bash
Codveda-DataScience-Internship-Level2/
│
├── data/
│   ├── raw/
│   │   ├── house Prediction Data Set.csv
│   │   ├── iris.csv
│   │   ├── churn-bigml-80.csv
│   │   ├── churn-bigml-20.csv
│   └── processed/
│       ├── house_processed.csv
│       ├── iris_processed.csv
│       ├── churn_processed.csv
├── src/
│   ├── task_1_regression/
│   │   ├── house_price_prediction.py
│   ├── task_2_classification/
│   │   ├── iris_classification.py
│   └── task_3_clustering/
│       ├── customer_segmentation.py
├── models/
│   ├── task_1_regression/
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   ├── model_metadata.json
│   │   └── performance_metrics.csv
│   ├── task_2_classification/
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   ├── model_metadata.json
│   │   ├── performance_metrics.csv
│   └── task_3_clustering/
│       ├── kmeans_model.pkl
│       ├── scaler.pkl
│       ├── clustering_metadata.json
├── results/
│   ├── task_1_regression/
│   │   ├── correlation_heatmap.png
│   │   ├── price_distribution.png
│   │   ├── model_comparison.png
│   │   ├── actual_vs_predicted.png
│   │   ├── residual_plot.png
│   │   ├── feature_importance.png
│   │   └── training_history.csv
│   ├── task_2_classification/
│   │   ├── class_distribution.png
│   │   ├── feature_correlations.png
│   │   ├── pairplot.png
│   │   ├── confusion_matrix_lr.png
│   │   ├── confusion_matrix_rf.png
│   │   ├── confusion_matrix_svm.png
│   │   ├── roc_curve.png
│   │   ├── model_comparison.png
│   │   ├── feature_importance.png
│   │   └── classification_report.csv
│   └── task_3_clustering/
│       ├── numeric_features_distribution.png
│       ├── elbow_method.png
│       ├── silhouette_scores.png
│       ├── pca_clusters.png
│       ├── tsne_clusters.png
│       ├── cluster_distribution.png
│       ├── cluster_characteristics.png
│       ├── 3d_pca_clusters.png
│       └── cluster_analysis_report.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Project Tasks

## Task 1: Predictive Modeling (Regression)

**Description**: Built and evaluated regression models to predict house prices using the Boston Housing dataset.

**Objectives Achieved**:

✅ Data preprocessing and exploratory analysis

✅ Implemented Linear Regression, Decision Tree, and Random Forest models

✅ Model evaluation using MSE, RMSE, and R-squared metrics

✅ Feature importance analysis

✅ Model serialization for future use

**Key Results**: Random Forest achieved the best performance with R² = 0.85

## Task 2: Classification with Logistic Regression

**Description**: Built multiple classifiers to predict iris flower species using the classic Iris dataset.

**Objectives Achieved**:

✅ Data preprocessing and visualization

✅ Implemented Logistic Regression, Random Forest, and SVM classifiers

✅ Comprehensive evaluation using accuracy, precision, recall, F1-score

✅ ROC curve analysis and confusion matrices

✅ Model comparison and selection

**Key Results**: All models achieved >95% accuracy, with Random Forest performing best

## Task 3: Clustering (Unsupervised Learning)

**Description**: Implemented K-Means clustering for customer segmentation using telecom churn data.

**Objectives Achieved**:

✅ Data preprocessing and dimensionality reduction

✅ Determined optimal clusters using elbow method and silhouette scores

✅ Visualized clusters using PCA and t-SNE

✅ Cluster interpretation and business insights

✅ Customer segmentation analysis

**Key Results**: Identified 9 distinct customer segments with unique characteristics

## 🛠️ Setup & Installation

1.  **Clone the repository** (if applicable) or ensure you have the project structure locally.
2.  **Navigate to the project root directory** in your terminal.
3.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    ```
4.  **Activate the virtual environment**:
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```
5.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## 📊 Usage

1. Task 1 - Regression:

```bash
python src/task_1_regression/house_price_prediction.py
```

2. Task 2 - Classification:

```bash
python src/task_2_classification/iris_classification.py
```

3. Task 3 - Clustering:

```bash
python src/task_3_clustering/customer_segmentation.py
```

## Exploring Results

**Visualizations**: Check results/task\_\*/ for all generated plots

**Processed Data**: Available in data/processed/

**Trained Models**: Stored in models/task\_\*/

## 🔧 Technologies Used

**Data Processing**: pandas, numpy

**Visualization**: matplotlib, seaborn, plotly

**Machine Learning**: scikit-learn, xgboost

**Model Serialization**: joblib

**Notebooks**: Jupyter

## 🎯 Next Steps

**Hyperparameter Tuning**: Implement grid search for optimal parameters

**Cross-Validation**: Add k-fold cross-validation for robust evaluation

**Feature Engineering**: Explore polynomial features and interactions

**Model Interpretability**: Add SHAP values for model explanations
