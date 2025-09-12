import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Create directories if they don't exist
results_dir = '../../results/task_3_clustering/'
processed_dir = '../../data/processed/'
models_dir = '../../models/task_3_clustering/'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load the data
print("Loading customer churn data...")
df = pd.read_csv('../../data/raw/churn-bigml-80.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Preprocessing for clustering
# We'll drop the target variable 'Churn' as this is unsupervised learning
if 'Churn' in df.columns:
    df_cluster = df.drop(columns=['Churn'])
    print("Dropped 'Churn' column for unsupervised learning")
else:
    df_cluster = df.copy()

# Handle categorical variables - we'll use one-hot encoding
df_processed = pd.get_dummies(df_cluster, drop_first=True)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_processed_imputed = pd.DataFrame(imputer.fit_transform(df_processed), 
                                   columns=df_processed.columns)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed_imputed)

# Save the scaler
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
print(f"Scaler saved to {os.path.join(models_dir, 'scaler.pkl')}")

# Save processed data
processed_data = df_processed_imputed.copy()
processed_data.to_csv(os.path.join(processed_dir, 'churn_processed.csv'), index=False)
print(f"Processed data saved to {os.path.join(processed_dir, 'churn_processed.csv')}")

# Determine optimal number of clusters using Elbow Method
print("\nCalculating optimal clusters using Elbow Method...")
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
plt.savefig(os.path.join(results_dir, 'elbow_method.png'))
plt.close()

# Calculate silhouette scores
print("\nCalculating silhouette scores...")
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, silhouette score is {silhouette_avg:.4f}")

# Plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different k Values')
plt.xticks(k_range)
plt.savefig(os.path.join(results_dir, 'silhouette_scores.png'))
plt.close()

# Choose optimal k based on silhouette score
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k} (based on silhouette score)")

# Apply K-Means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Save the model
joblib.dump(kmeans, os.path.join(models_dir, 'kmeans_model.pkl'))
print(f"K-Means model saved to {os.path.join(models_dir, 'kmeans_model.pkl')}")

# Add cluster labels to the original dataframe
df['Cluster'] = clusters
print("\nCluster distribution:")
cluster_distribution = df['Cluster'].value_counts().sort_index()
print(cluster_distribution)

# Plot cluster distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_distribution.index, y=cluster_distribution.values)
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.savefig(os.path.join(results_dir, 'cluster_distribution.png'))
plt.close()

# Dimensionality reduction for visualization
print("\nReducing dimensions for visualization...")

# Using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Customer Segments Visualized with PCA')
plt.savefig(os.path.join(results_dir, 'pca_clusters.png'))
plt.close()

# Using t-SNE (more computationally expensive but often better for visualization)
print("Running t-SNE (this may take a while)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Customer Segments Visualized with t-SNE')
plt.savefig(os.path.join(results_dir, 'tsne_clusters.png'))
plt.close()

# Analyze cluster characteristics - only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
if 'Cluster' in numeric_cols:
    numeric_cols = numeric_cols.drop('Cluster')
    
cluster_analysis = df.groupby('Cluster')[numeric_cols].mean()
print("\nCluster characteristics (mean values for numeric features):")
print(cluster_analysis)

# Visualize cluster characteristics for numeric features
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_analysis.T, cmap='coolwarm', center=0, annot=True, fmt='.2f')
plt.title('Cluster Characteristics Heatmap (Numeric Features)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'cluster_characteristics.png'))
plt.close()

# For categorical variables, we can look at value counts per cluster
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\nCategorical variable distribution by cluster:")
    for col in categorical_cols:
        cross_tab = pd.crosstab(df['Cluster'], df[col])
        print(f"\n{col} distribution:")
        print(cross_tab)

# Save clustering metadata
metadata = {
    'optimal_clusters': int(optimal_k),
    'silhouette_score': float(silhouette_scores[np.argmax(silhouette_scores)]),
    'cluster_distribution': cluster_distribution.to_dict(),
    'feature_names': df_processed_imputed.columns.tolist(),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_shape': {
        'original': df.shape,
        'processed': df_processed_imputed.shape
    }
}

with open(os.path.join(models_dir, 'clustering_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Clustering metadata saved to {os.path.join(models_dir, 'clustering_metadata.json')}")

# Interpret and summarize key findings
print("\n=== KEY FINDINGS ===")
print(f"1. Optimal number of clusters: {optimal_k}")
print(f"2. Best silhouette score: {silhouette_scores[np.argmax(silhouette_scores)]:.4f}")
print("3. Cluster distribution:")
for cluster, count in cluster_distribution.items():
    print(f"   Cluster {cluster}: {count} customers ({count/len(df)*100:.1f}%)")

print("\n4. Cluster characteristics summary:")
# Analyze what makes each cluster unique
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} profile:")
    # Find top 3 features that are most different from overall mean
    cluster_features = cluster_analysis.loc[cluster]
    overall_means = df[numeric_cols].mean()
    differences = (cluster_features - overall_means).abs().sort_values(ascending=False)
    top_features = differences.head(3)
    
    for feature in top_features.index:
        cluster_val = cluster_features[feature]
        overall_val = overall_means[feature]
        diff_pct = (cluster_val - overall_val) / overall_val * 100
        print(f"   - {feature}: {cluster_val:.2f} (vs overall {overall_val:.2f}, {diff_pct:+.1f}%)")

print(f"\nClustering task completed! Check the following directories:")
print(f"- Results: {results_dir}")
print(f"- Processed data: {processed_dir}")
print(f"- Saved models: {models_dir}")