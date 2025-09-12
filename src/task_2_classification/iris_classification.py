import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

# Create directories if they don't exist
results_dir = '../../results/task_2_classification/'
processed_dir = '../../data/processed/'
models_dir = '../../models/task_2_classification/'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load the data
print("Loading Iris dataset...")
df = pd.read_csv('../../data/raw/iris.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())
print("\nClass distribution:")
print(df['species'].value_counts())

# EDA: Class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='species', data=df)
plt.title('Class Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.savefig(os.path.join(results_dir, 'class_distribution.png'))
plt.close()

# EDA: Pairplot to visualize relationships
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='species', palette='viridis')
plt.suptitle('Iris Dataset Pairplot', y=1.02)
plt.savefig(os.path.join(results_dir, 'pairplot.png'))
plt.close()

# EDA: Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'feature_correlations.png'))
plt.close()

# Preprocessing
X = df.drop(columns=['species'])
y = df['species']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save processed data
processed_data = X.copy()
processed_data['species_encoded'] = y_encoded
processed_data.to_csv(os.path.join(processed_dir, 'iris_processed.csv'), index=False)
print(f"\nProcessed data saved to {os.path.join(processed_dir, 'iris_processed.csv')}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
print(f"Scaler saved to {os.path.join(models_dir, 'scaler.pkl')}")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

# Train and evaluate models
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    trained_models[name] = model
    
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    # Classification report
    print(f"\nClassification Report - {name}:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Compare model performance
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)

# Plot model comparison
plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x_pos = np.arange(len(metrics))

for i, model in enumerate(results_df.index):
    values = [results_df.loc[model, metric] for metric in metrics]
    plt.bar(x_pos + i*0.2, values, width=0.2, label=model)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Comparison Across Metrics')
plt.xticks(x_pos + 0.2, metrics)
plt.legend()
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
plt.close()

# ROC Curve (for multiclass)
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']

for i, (name, model) in enumerate(models.items()):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for j in range(len(le.classes_)):
            fpr[j], tpr[j], _ = roc_curve(y_test == j, y_prob[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])
        
        # Plot ROC curves
        for j, color in zip(range(len(le.classes_)), colors):
            plt.plot(fpr[j], tpr[j], color=color, lw=2,
                     label='ROC curve of class {0} ({1}) (area = {2:0.2f})'
                     ''.format(le.classes_[j], name, roc_auc[j]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curves')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'roc_curves.png'))
plt.close()

# Identify the best model
best_model_name = results_df['Accuracy'].idxmax()
best_model = trained_models[best_model_name]
best_metrics = results[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"Best Accuracy: {best_metrics['Accuracy']:.4f}")
print(f"Best F1-Score: {best_metrics['F1-Score']:.4f}")

# Save the best model
model_path = os.path.join(models_dir, 'best_model.pkl')
joblib.dump(best_model, model_path)
print(f"Best model saved to {model_path}")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'model_type': type(best_model).__name__,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'performance_metrics': best_metrics,
    'feature_names': X.columns.tolist(),
    'target_classes': le.classes_.tolist(),
    'class_encoding': dict(zip(range(len(le.classes_)), le.classes_)),
    'dataset_shape': {
        'original': df.shape,
        'processed': processed_data.shape
    }
}

with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Model metadata saved to {os.path.join(models_dir, 'model_metadata.json')}")

# Feature importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance ({best_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close()

print(f"\nClassification task completed! Check the following directories:")
print(f"- Results: {results_dir}")
print(f"- Processed data: {processed_dir}")
print(f"- Saved models: {models_dir}")