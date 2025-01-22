import pandas as pd
import seaborn as sns
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.manifold import TSNE
import datetime as dt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

### Functions used ###

# Function to provide overview on data
def data_overview(df, head=5):
    print(" SHAPE OF DATASET ".center(125, '-'))
    print(f'Rows: {df.shape[0]}')
    print(f'Columns: {df.shape[1]}')
    print(" HEAD ".center(125, '-'))
    print(df.head(head))
    print(" DATA TYPES ".center(125, '-'))
    print(df.dtypes.value_counts())
    print(" MISSING VALUES ".center(125, '-'))
    print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))
    print(" DUPLICATED VALUES ".center(125, '-'))
    print(df.duplicated().sum())
    print(" STATISTICS OF DATA ".center(125, '-'))
    print(df.describe(include="all"))
    print(" DATA INFO ".center(125, '-'))
    print(df.info())

# Functions to handle outliers
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return up_limit, low_limit

def replace_with_threshold(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Load data
data = pd.read_csv("onlineretail/online_retail_II.csv", encoding='unicode_escape')
pd.set_option('display.max_columns', None)

data_overview(data)

# Cleaning data
print("Shape of data before removing NaN's CustomerID:", data.shape)
data.dropna(subset=["Customer ID"], axis=0, inplace=True)
print("Shape of data after removing NaN's CustomerID:", data.shape)

print("Missing values in each column after cleaning customerID:", data.isnull().sum())

data = data[~data.Invoice.str.contains('C', na=False)]
print("Dataset is free from cancelled products information")

data = data.drop_duplicates(keep="first")
print("Number of duplicates after cleaning:", data.duplicated().sum())

data = data[data.Quantity > 0] # Handle negative values
data = data[data.Price > 0]

# Remove outliers
replace_with_threshold(data, "Quantity")
replace_with_threshold(data, "Price")

# Create new feature Revenue
data["Revenue"] = data["Quantity"] * data["Price"]

# RFM Features
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
Latest_Date = dt.datetime(2011, 12, 10)

RFM = data.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (Latest_Date - x.max()).days,
    'Invoice': 'nunique',
    'Revenue': 'sum'
})

RFM.rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Revenue': 'Monetary'}, inplace=True)
RFM = RFM[RFM["Frequency"] > 1]

# Add Interpurchase_Time
data_grouped = data.groupby('Customer ID')
Shopping_Cycle = data_grouped.agg({'InvoiceDate': lambda x: (x.max() - x.min()).days})
RFM["Shopping_Cycle"] = Shopping_Cycle
RFM["Interpurchase_Time"] = RFM["Shopping_Cycle"] // RFM["Frequency"]

RFMT = RFM[["Recency", "Frequency", "Monetary", "Interpurchase_Time"]]

# Pairplot for visualizing relationships between features
sns.pairplot(RFMT, diag_kind='kde')
plt.suptitle("Pairplot of RFM Features", y=1.02)
plt.show()

# Histograms for each feature
RFMT.hist(figsize=(12, 10), bins=20)
plt.suptitle("Histograms of RFM Features", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(RFMT.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of RFM Features")
plt.show()

# Scale the features
scaler = StandardScaler()
RFMT_scaled = scaler.fit_transform(RFMT[["Recency", "Frequency", "Monetary", "Interpurchase_Time"]])

# Apply PCA to reduce dimensions
pca = PCA(n_components=2)
RFMT_pca = pca.fit_transform(RFMT_scaled)

# Fit K-Means with the best number of clusters
kmeans = KMeans(n_clusters=5, max_iter=50, random_state=42)
kmeans.fit(RFMT_pca)
RFMT["Cluster"] = kmeans.labels_

# Visualize clusters using PCA (2D)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=RFMT_pca[:, 0], y=RFMT_pca[:, 1], hue=RFMT["Cluster"], palette="tab10", s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black", marker="*", s=200, label="Centroids")
plt.title("Clusters Visualization (PCA Reduced to 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()

KMeansScore = silhouette_score(RFMT_pca, kmeans.labels_)
print(f"Silhouette Score for KMeans 5 clusters: {KMeansScore:.2f}")

# Cluster Summary
cluster_summary = RFMT.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "Interpurchase_Time": "mean",
    "Cluster": "size"
}).rename(columns={"Cluster": "Size"}).round(2)

print("Cluster Summary:")
print(tabulate(cluster_summary, headers="keys", tablefmt="psql"))

agglo = AgglomerativeClustering(n_clusters=5, linkage='ward')
agglo.fit(RFMT_pca)
RFMT["AggloCluster"] = agglo.labels_

plt.figure(figsize=(10, 6))
sns.scatterplot(x=RFMT_pca[:, 0], y=RFMT_pca[:, 1], hue=RFMT["AggloCluster"], palette="tab10", s=50)
plt.title("Clusters Visualization (PCA Reduced to 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()

AggloScore = silhouette_score(RFMT_pca, agglo.labels_)
print(f"Silhouette Score for Agglomerative 5 clusters: {AggloScore:.2f}")

# Cluster Summary
cluster_summary = RFMT.groupby("AggloCluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "Interpurchase_Time": "mean",
    "Cluster": "size"
}).rename(columns={"Cluster": "Size"}).round(2)

print("Cluster Summary:")
print(tabulate(cluster_summary, headers="keys", tablefmt="psql"))

# DBSCAN for Comparison
dbscan = DBSCAN(eps=1.5, min_samples=50)
RFMT["DBSCAN_Cluster"] = dbscan.fit_predict(RFMT_pca)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=RFMT_pca[:, 0], y=RFMT_pca[:, 1], hue=RFMT["DBSCAN_Cluster"], palette="tab10", s=50)
plt.title("Clusters Visualization (PCA Reduced to 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()

# Silhouette Score for DBSCAN
dbscan_labels = RFMT["DBSCAN_Cluster"]
if len(set(dbscan_labels)) > 1:  # Silhouette Score only works if there are at least 2 clusters
    dbscan_silhouette = silhouette_score(RFMT_pca, dbscan_labels)
    print(f"Silhouette Score for DBSCAN: {dbscan_silhouette:.2f}")
else:
    print("DBSCAN produced a single cluster or noise.")

# Cluster Summary
cluster_summary = RFMT.groupby("DBSCAN_Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "Interpurchase_Time": "mean",
    "Cluster": "size"
}).rename(columns={"Cluster": "Size"}).round(2)

print("Cluster Summary:")
print(tabulate(cluster_summary, headers="keys", tablefmt="psql"))


# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
RFMT_tsne = tsne.fit_transform(RFMT_scaled)

# Plot t-SNE with KMeans clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=RFMT_tsne[:, 0], y=RFMT_tsne[:, 1], hue=RFMT["Cluster"], palette="tab10", s=50)
plt.title("t-SNE Visualization of KMeans Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.show()

# Plot t-SNE with DBSCAN clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=RFMT_tsne[:, 0], y=RFMT_tsne[:, 1], hue=RFMT["DBSCAN_Cluster"], palette="tab10", s=50)
plt.title("t-SNE Visualization of DBSCAN Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.show()

# Plot t-SNE with Agglomerative Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=RFMT_tsne[:, 0], y=RFMT_tsne[:, 1], hue=RFMT["AggloCluster"], palette="tab10", s=50)
plt.title("t-SNE Visualization of Agglomerative Clustering")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.show()

'''
from joblib import dump
dump(kmeans, "kmeans_model.joblib")
dump(scaler, "scaler.joblib")
dump(pca, "pca.joblib")
'''