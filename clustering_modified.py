# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA

# Load data and replace '-' with NaN
modified_data = pd.read_excel("/Users/QiaoChuDing_1/Desktop/Birkbeck/2024-2025/4. Project/msc-project-source-code-files-24-25-QiaochuDing/modified_data.xlsx")
modified_data.replace("-", np.nan, inplace=True)
print(modified_data.head())

# Map seasonality to numerical values and define numeric columns
seasonality_map = {'Low': 1, 'Medium': 2, 'High': 3}
modified_data['seasonality'] = modified_data['seasonality'].map(seasonality_map)

numeric_cols = [
    'non_UK_workforce',
    'vacancy_rate',
    'ssv_density',
    'med_annual_wage_differential',
    'visa_grants',
    'jobs_at_risk_of_automation',
    'seasonality'
]

print(modified_data.dtypes)

# Visualise correlation matrix
corr_matrix = modified_data.drop(columns=['industry','sic_code']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Check missing values
print(modified_data.isnull().sum())

# Visualise distribution of numeric columns
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(modified_data[col], kde=True, bins=15)
    plt.title(f'Distribution of {col}')
    plt.show()

# Impute missing values with median
modified_data[numeric_cols] = modified_data[numeric_cols].fillna(modified_data[numeric_cols].median())
print(modified_data.head())

# Plot bar charts for numeric columns by industry
for col in numeric_cols:
    plt.figure(figsize=(20, 8))
    sorted_data = modified_data.sort_values(by=col, ascending=False)
    sns.barplot(x='industry', y=col, data=sorted_data)
    plt.title(f'{col} by Industry')
    plt.xticks(rotation=90)
    plt.show()

# Calculate summary statistics for numeric columns
summary_stats = modified_data[numeric_cols].agg(['mean', 'median', 'std']).reset_index()
print(summary_stats)

# Scale the numeric data
scaler = StandardScaler()
modified_data[numeric_cols] = scaler.fit_transform(modified_data[numeric_cols])
print(modified_data.head())

# Use Elbow and Silhouette methods to find optimal number of clusters
sse = []
silhouette_scores = []
k_range = range(2, 11)

X_scaled = modified_data[numeric_cols]

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method (SSE)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')

plt.tight_layout()
plt.show()

# KMeans clustering with 7 clusters and print industries in each cluster
kmeans = KMeans(n_clusters=7, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X_scaled)
modified_data['cluster_label'] = kmeans.labels_

for cluster_id in sorted(modified_data['cluster_label'].unique()):
    industries_in_cluster = modified_data[modified_data['cluster_label'] == cluster_id]['industry'].tolist()
    print(f"Cluster {cluster_id}:")
    for industry in industries_in_cluster:
        print(f"- {industry}")
    print("\n")

# Print mean of numeric columns for each cluster
cluster_means = modified_data.groupby('cluster_label')[numeric_cols].mean()
print(cluster_means)

# PCA to visualise clusters in 2D space
pca = PCA(n_components=2)
principal_components = pca.fit_transform(modified_data[numeric_cols])
principal_df = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2'])
principal_df['cluster_label'] = modified_data['cluster_label']

plt.figure(figsize=(10, 8))
sns.scatterplot(x='principal component 1', y='principal component 2', hue='cluster_label', data=principal_df, palette='viridis', s=100)
plt.title('PCA of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Davies-Bouldin Index for other cluster options
X = modified_data[numeric_cols]

kmeans_5 = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
clusters_5 = kmeans_5.fit_predict(X)
davies_bouldin_5 = davies_bouldin_score(X, clusters_5)
print(f"Davies-Bouldin Index for 5 clusters: {davies_bouldin_5}")

kmeans_6 = KMeans(n_clusters=6, init='k-means++', random_state=42, n_init=10)
clusters_6 = kmeans_6.fit_predict(X)
davies_bouldin_6 = davies_bouldin_score(X, clusters_6)
print(f"Davies-Bouldin Index for 6 clusters: {davies_bouldin_6}")

kmeans_7 = KMeans(n_clusters=7, init='k-means++', random_state=42, n_init=10)
clusters_7 = kmeans_7.fit_predict(X)
davies_bouldin_7 = davies_bouldin_score(X, clusters_7)
print(f"Davies-Bouldin Index for 7 clusters: {davies_bouldin_7}")

kmeans_8 = KMeans(n_clusters=8, init='k-means++', random_state=42, n_init=10)
clusters_8 = kmeans_8.fit_predict(X)
davies_bouldin_8 = davies_bouldin_score(X, clusters_8)
print(f"Davies-Bouldin Index for 8 clusters: {davies_bouldin_8}")

# Write out cluster labels to Excel
modified_data.to_excel("modified_data_clusters.xlsx", index=False)

# Preparing data for drift analysis

from scipy.spatial.distance import cdist

baseline = pd.read_excel("/Users/QiaoChuDing_1/Desktop/Birkbeck/2024-2025/4. Project/msc-project-source-code-files-24-25-QiaochuDing/modified_data.xlsx")
multipliers = pd.read_excel("/Users/QiaoChuDing_1/Desktop/Birkbeck/2024-2025/4. Project/msc-project-source-code-files-24-25-QiaochuDing/multiplier_matrix.xlsx")

numeric_cols = [
    'non_UK_workforce',
    'vacancy_rate',
    'ssv_density',
    'med_annual_wage_differential',
    'visa_grants',
    'jobs_at_risk_of_automation'
]

model_cols = numeric_cols + ['seasonality']
merge_keys = ['sic_code']

seasonality_map = {'Low': 1, 'Medium': 2, 'High': 3}
baseline['seasonality'] = baseline['seasonality'].map(seasonality_map)

baseline = baseline.replace("-", np.nan)
baseline[numeric_cols] = baseline[numeric_cols].fillna(baseline[numeric_cols].median())

# Establish multipliers

mult_needed = merge_keys + ['scenario'] + numeric_cols
multipliers = multipliers[mult_needed].copy()

# Multiply baseline with multipliers to generate synthetic data

synthetic_list = []

for scen in multipliers['scenario'].dropna().unique():
    m_s = multipliers[multipliers['scenario'] == scen].copy()

    merged = baseline.merge(
        m_s.drop(columns=['scenario']),
        on=merge_keys, how='left', suffixes=('', '_mult')
    )

    for c in numeric_cols:
        mult_col = f"{c}_mult"
        if mult_col not in merged:
            merged[mult_col] = 1.0
        merged[c] = merged[c] * merged[mult_col]

    drop_cols = [f"{c}_mult" for c in numeric_cols]
    merged = merged.drop(columns=drop_cols).copy()
    merged['scenario'] = scen

    synthetic_list.append(merged)

synthetic_all = pd.concat(synthetic_list, ignore_index=True)

