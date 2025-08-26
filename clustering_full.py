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
full_data = pd.read_excel("/Users/QiaoChuDing_1/Desktop/Birkbeck/2024-2025/4. Project/msc-project-source-code-files-24-25-QiaochuDing/full_data.xlsx")
full_data.replace("-", np.nan, inplace=True)
print(full_data.head())

# Map seasonality to numerical values and define numeric columns
seasonality_map = {'Low': 1, 'Medium': 2, 'High': 3}
full_data['seasonality'] = full_data['seasonality'].map(seasonality_map)

numeric_cols = [
    'non_UK_workforce',
    'vacancy_rate',
    'ssv_density',
    'med_annual_wage_differential',
    'visa_grants',
    'jobs_at_risk_of_automation',
    'seasonality'
]

print(full_data.dtypes)

# Visualise correlation matrix
corr_matrix = full_data.drop(columns=['industry','sic_code']).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Check missing values
print(full_data.isnull().sum())

# Visualise distribution of numeric columns
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(full_data[col], kde=True, bins=15)
    plt.title(f'Distribution of {col}')
    plt.show()

# Impute missing values with median
full_data[numeric_cols] = full_data[numeric_cols].fillna(full_data[numeric_cols].median())
full_data.head()

# Plot bar charts for numeric columns by industry
for col in numeric_cols:
    plt.figure(figsize=(20, 8))
    sorted_data = full_data.sort_values(by=col, ascending=False)
    sns.barplot(x='industry', y=col, data=sorted_data)
    plt.title(f'{col} by Industry')
    plt.xticks(rotation=90)
    plt.show()

# Calculate summary statistics for numeric columns
summary_stats = full_data[numeric_cols].agg(['mean', 'median', 'std']).reset_index()
print(summary_stats)

# Scale the numeric data
scaler = StandardScaler()
full_data[numeric_cols] = scaler.fit_transform(full_data[numeric_cols])
print(full_data.head())

# Use Elbow and Silhouette methods to find optimal number of clusters
sse = []
silhouette_scores = []
k_range = range(2, 11)

X_scaled = full_data[numeric_cols]

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

# KMeans clustering with 6 clusters and print industries in each cluster
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X_scaled)
full_data['cluster_label'] = kmeans.labels_

for cluster_id in sorted(full_data['cluster_label'].unique()):
    industries_in_cluster = full_data[full_data['cluster_label'] == cluster_id]['industry'].tolist()
    print(f"Cluster {cluster_id}:")
    for industry in industries_in_cluster:
        print(f"- {industry}")
    print("\n")