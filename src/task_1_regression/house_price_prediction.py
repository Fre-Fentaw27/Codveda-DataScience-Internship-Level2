import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Create directories if they don't exist
results_dir = '../../results/task_1_regression/'
processed_dir = '../../data/processed/'
models_dir = '../../models/task_1_regression/'
raw_data_dir = '../../data/raw/'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load the data with the correct file name and proper parsing
print("Loading house price data...")
data_file_path = '../../data/raw/house Prediction Data Set.csv'

if not os.path.exists(data_file_path):
    # If the file doesn't exist, check what files are available
    print(f"File not found: {data_file_path}")
    print("Available files in data/raw/:")
    if os.path.exists(raw_data_dir):
        files = os.listdir(raw_data_dir)
        for file in files:
            print(f"  - {file}")
    raise FileNotFoundError(f"Could not find the data file: {data_file_path}")

# Let's first check the raw file content to understand its structure
with open(data_file_path, 'r') as f:
    first_few_lines = [next(f) for _ in range(5)]
print("First few lines of the file:")
for i, line in enumerate(first_few_lines):
    print(f"Line {i}: {line.strip()}")

# Try reading without header since the file doesn't have column names
try:
    # Read without header and with space delimiter
    df = pd.read_csv(data_file_path, header=None, delim_whitespace=True)
    print("Data loaded with header=None and delim_whitespace=True")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows with all data included:")
print(df.head())
print("\nColumn names (temporary):")
print(df.columns.tolist())

# Assign proper column names based on Boston Housing dataset
# CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

if len(column_names) == df.shape[1]:
    df.columns = column_names
    print("\nAssigned proper column names:")
    print(df.columns.tolist())
else:
    print(f"\nWarning: Expected {len(column_names)} columns but got {df.shape[1]}")
    # Use generic names if column count doesn't match
    df.columns = [f'feature_{i}' for i in range(df.shape[1]-1)] + ['price']
    print("Assigned generic column names:")
    print(df.columns.tolist())

print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Identify target variable (typically the last column for house price data)
target = df.columns[-1]  # Use the last column as target
print(f"\nUsing column '{target}' as target variable")

# EDA: Correlation heatmap
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'))
plt.close()

# Distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(df[target], kde=True)
plt.title(f'Distribution of {target}')
plt.xlabel(target)
plt.savefig(os.path.join(results_dir, 'target_distribution.png'))
plt.close()

# Preprocessing
X = df.drop(columns=[target])
y = df[target]

# Handle categorical variables if any (CHAS is typically binary)
X_processed = pd.get_dummies(X, drop_first=True)

# Save processed data
processed_data = X_processed.copy()
processed_data[target] = y
processed_data.to_csv(os.path.join(processed_dir, 'house_processed.csv'), index=False)
print(f"\nProcessed data saved to {os.path.join(processed_dir, 'house_processed.csv')}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
print(f"Scaler saved to {os.path.join(models_dir, 'scaler.pkl')}")

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
}

# Train and evaluate models
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    
    trained_models[name] = model
    
    print(f"{name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Compare model performance
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)

# Plot model performance comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x=results_df.index, y=results_df['R2'])
plt.title('R-squared Comparison')
plt.xticks(rotation=45)
plt.ylabel('R-squared')

plt.subplot(1, 2, 2)
sns.barplot(x=results_df.index, y=results_df['RMSE'])
plt.title('RMSE Comparison')
plt.xticks(rotation=45)
plt.ylabel('RMSE')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
plt.close()

# Identify the best model
best_model_name = results_df['R2'].idxmax()
best_model = trained_models[best_model_name]
best_metrics = results[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"Best R2 score: {best_metrics['R2']:.4f}")
print(f"Best RMSE: {best_metrics['RMSE']:.4f}")

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
    'feature_names': X_processed.columns.tolist(),
    'target_variable': target,
    'dataset_shape': {
        'original': df.shape,
        'processed': processed_data.shape
    },
    'data_file': data_file_path,
    'column_names': df.columns.tolist()
}

with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Model metadata saved to {os.path.join(models_dir, 'model_metadata.json')}")

# Actual vs Predicted values for the best model
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted Values ({best_model_name})')
plt.savefig(os.path.join(results_dir, 'actual_vs_predicted.png'))
plt.close()

# Residual plot
residuals = y_test - y_pred_best
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_best, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig(os.path.join(results_dir, 'residual_plot.png'))
plt.close()

# Feature importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Important Features ({best_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close()

print(f"\nRegression task completed! Check the following directories:")
print(f"- Results: {results_dir}")
print(f"- Processed data: {processed_dir}")
print(f"- Saved models: {models_dir}")