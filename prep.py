# Import libraries
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import joblib

# Define constants
DATA_DIR = Path("labour-deps-dashboard/data")
OUT_DIR = Path("labour-deps-dashboard/output")
ID_COLS = ["industry", "sic_code"]
SEASONALITY_MAP = {"Low": 1, "Medium": 2, "High": 3}

NUMERIC_COLS = [
    "non_UK_workforce",
    "vacancy_rate",
    "ssv_density",
    "med_annual_wage_differential",
    "visa_grants",
    "jobs_at_risk_of_automation",
    "seasonality",
]

MODEL_COLS = NUMERIC_COLS
N_CLUSTERS = 7

# Prep data
baseline = pd.read_excel(DATA_DIR / "modified_data.xlsx")
baseline = baseline.replace("-", np.nan)
seasonality_map = {'Low': 1, 'Medium': 2, 'High': 3}
baseline['seasonality'] = baseline['seasonality'].map(seasonality_map)
baseline[NUMERIC_COLS] = baseline[NUMERIC_COLS].fillna(baseline[NUMERIC_COLS].median())
baseline_unscaled = baseline.copy()

# Scale and cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(baseline_unscaled[MODEL_COLS])
kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
baseline_labeled = baseline_unscaled.copy()
baseline_labeled['cluster_label'] = labels

# PCA
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
pca_df = baseline_labeled[ID_COLS + ['cluster_label']].copy()
pca_df['pc1'] = coords[:, 0]
pca_df['pc2'] = coords[:, 1]
explained = pca.explained_variance_ratio_

# Cluster means
summary = (baseline_labeled.groupby('cluster_label')[NUMERIC_COLS].mean().round(2))

# Write outputs
baseline_labeled.to_csv(OUT_DIR / "baseline_with_clusters.csv", index=False)
pca_df.to_csv(OUT_DIR / "pca_coords.csv", index=False)
summary.to_csv(OUT_DIR / "cluster_summary.csv")
joblib.dump(scaler, OUT_DIR / "scaler.pkl")
joblib.dump(kmeans, OUT_DIR / "kmeans.pkl")
pd.DataFrame({
    "explained_pc1": [explained[0]],
    "explained_pc2": [explained[1]],
}).to_csv(OUT_DIR / "pca_explained.csv", index=False)

# Calculate drift
multipliers = pd.read_excel(DATA_DIR / "multiplier_matrix.xlsx")
multipliers['seasonality'] = multipliers['seasonality'].map(seasonality_map)
mult_needed = ['sic_code', 'scenario'] + NUMERIC_COLS
multipliers = multipliers[mult_needed].copy()

synthetic_list = []
for scen in multipliers['scenario'].dropna().unique():
    m_s = multipliers[multipliers['scenario'] == scen].copy()
    merged = baseline_unscaled.merge(
        m_s.drop(columns=['scenario']),
        on=['sic_code'], how='left', suffixes=('', '_mult')
    )
    for c in NUMERIC_COLS:
        mult_col = f"{c}_mult"
        if mult_col not in merged:
            merged[mult_col] = 1.0
        merged[c] = merged[c] * merged[mult_col]
    merged.drop(columns=[f"{c}_mult" for c in NUMERIC_COLS], inplace=True, errors='ignore')
    merged['scenario'] = scen
    synthetic_list.append(merged)

synthetic_all = pd.concat(synthetic_list, ignore_index=True)

# Predict clusters and distances using saved scaler/kmeans
X_syn_scaled = scaler.transform(synthetic_all[MODEL_COLS])
syn_labels = kmeans.predict(X_syn_scaled)
synthetic_all['cluster_label'] = syn_labels

centroids = kmeans.cluster_centers_
syn_dists = cdist(X_syn_scaled, centroids, metric='euclidean')
synthetic_all['dist_to_centroid'] = syn_dists[np.arange(syn_dists.shape[0]), syn_labels]

# Baseline distances (only needed once)
base_dists = cdist(X_scaled, centroids, metric='euclidean')
baseline_labeled['dist_to_centroid'] = base_dists[np.arange(base_dists.shape[0]), labels]

# Merge baseline clusters and distances via sic_code
synthetic_all = synthetic_all.merge(
    baseline_labeled[['sic_code', 'industry', 'cluster_label', 'dist_to_centroid']].rename(
        columns={'cluster_label': 'baseline_cluster_label', 'dist_to_centroid': 'baseline_dist_to_centroid'}
    ),
    on='sic_code', how='left'
)

synthetic_all['drift'] = synthetic_all['dist_to_centroid'] - synthetic_all['baseline_dist_to_centroid']
synthetic_all['cluster_changed'] = (synthetic_all['cluster_label'] != synthetic_all['baseline_cluster_label'])
synthetic_all.to_csv(OUT_DIR / "synthetic_clusters_drift.csv", index=False)