import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_excel("telecomChurnData.xlsx", sheet_name='Worksheet')

# Preprocessing
# Convert TotalCharges to numeric and handle errors
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(0)  # Fill missing TotalCharges with 0

# Drop CustomerID as it is not useful for clustering
data.drop('CustomerID', axis=1, inplace=True)

# Define categorical and numerical columns
categorical_cols = [
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'Churn'
]
numerical_cols = ['SeniorCitizen', 'Tenure', 'MonthlyCharges', 'TotalCharges']

# One-hot encoding for categorical data and scaling for numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Create a pipeline with preprocessing and PCA
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=3))
])

# Transform the data using the pipeline
X_transformed = pipeline.fit_transform(data)

# Correlation Heatmap for Numerical Features
numerical_data = data[numerical_cols]
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# PCA Explained Variance Plot
# Extract PCA component information
pca = pipeline.named_steps['pca']
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, alpha=0.7, align='center')
plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio) * 100, where='mid', color='red')
plt.title('PCA Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.legend(['Cumulative Explained Variance', 'Individual Component Variance'])
plt.grid(True)
plt.show()

# PCA Feature Loadings
# Feature contribution to each principal component
if isinstance(preprocessor, ColumnTransformer):
    # Combine feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([numerical_cols, cat_feature_names])
else:
    feature_names = numerical_cols + categorical_cols

# Loadings (contributions of original features to PCs)
loadings = pca.components_.T

plt.figure(figsize=(12, 8))
for i in range(3):
    plt.bar(feature_names, loadings[:, i], alpha=0.7, label=f'PC{i+1}')
plt.xticks(rotation=90)
plt.title('PCA Feature Loadings')
plt.xlabel('Original Features')
plt.ylabel('Contribution to Principal Components')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Perform Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=5, linkage='ward')  # Adjust the number of clusters
clusters = agglo.fit_predict(X_transformed)

# Evaluate clustering performance
silhouette_avg = silhouette_score(X_transformed, clusters)
print(f"Agglomerative Clustering Silhouette Score: {silhouette_avg}")

# Visualize the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=clusters, cmap='viridis', s=50)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('3D Cluster Visualization')
plt.show()

# Add cluster labels to the original data
data['Cluster'] = clusters

# Summary for numerical features
numerical_summary = data.groupby('Cluster')[numerical_cols].agg(['mean', 'std', 'median'])

# Summary for categorical features
categorical_summary = {}
for col in categorical_cols:
    categorical_summary[col] = data.groupby('Cluster')[col].value_counts(normalize=True).unstack()

# Display numerical summaries
print("Numerical Feature Summary by Cluster:")
print(tabulate(numerical_summary, headers='keys', tablefmt='fancy_grid'))

# Display categorical summaries
print("\nCategorical Feature Summary by Cluster:")
for col, summary in categorical_summary.items():
    print(f"\nFeature: {col}")
    print(tabulate(summary, headers='keys', tablefmt='fancy_grid'))

# Save summaries to Excel for further review (if needed)
with pd.ExcelWriter("Cluster_Analysis_Summary.xlsx") as writer:
    numerical_summary.to_excel(writer, sheet_name='Numerical_Summary')
    for col, summary in categorical_summary.items():
        summary.to_excel(writer, sheet_name=f'{col}_Summary')
