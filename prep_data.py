import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import joblib

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

NUMERIC_COLS_NO_SEAS = [
    'non_UK_workforce',
    'vacancy_rate',
    'ssv_density',
    'med_annual_wage_differential',
    'visa_grants',
    'jobs_at_risk_of_automation',
]
MODEL_COLS = NUMERIC_COLS_NO_SEAS + ['seasonality']
ID_COLS = ['industry', 'sic_code']
SEASONALITY_MAP = {'Low': 1, 'Medium': 2, 'High': 3}

N_CLUSTERS = 7

baseline = pd.read_excel(DATA_DIR / "modified_data.xlsx")
baseline = baseline.replace("-", np.nan)

if baseline['seasonality'].dtype == object:
    baseline['seasonality'] = baseline['seasonality'].map(SEASONALITY_MAP)

baseline_unscaled = baseline.copy()

baseline_unscaled[NUMERIC_COLS_NO_SEAS] = baseline_unscaled[NUMERIC_COLS_NO_SEAS].fillna(
    baseline_unscaled[NUMERIC_COLS_NO_SEAS].median()
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(baseline_unscaled[MODEL_COLS])

kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

baseline_labeled = baseline_unscaled.copy()
baseline_labeled['cluster_label'] = labels

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
pca_df = baseline_labeled[ID_COLS + ['cluster_label']].copy()
pca_df['pc1'] = coords[:, 0]
pca_df['pc2'] = coords[:, 1]
explained = pca.explained_variance_ratio_

summary = (baseline_labeled.groupby('cluster_label')[NUMERIC_COLS_NO_SEAS + ['seasonality']]
           .agg(['mean','median','min','max','count'])
           .round(2))

baseline_labeled.to_csv(OUT_DIR / "baseline_with_clusters.csv", index=False)
pca_df.to_csv(OUT_DIR / "pca_coords.csv", index=False)
summary.to_csv(OUT_DIR / "cluster_summary.csv")
joblib.dump(scaler, OUT_DIR / "scaler.pkl")
joblib.dump(kmeans, OUT_DIR / "kmeans.pkl")
pd.DataFrame({
    "explained_pc1": [explained[0]],
    "explained_pc2": [explained[1]],
}).to_csv(OUT_DIR / "pca_explained.csv", index=False)

multipliers = pd.read_excel(DATA_DIR / "multiplier_matrix.xlsx")
mult_needed = ['sic_code', 'scenario'] + NUMERIC_COLS_NO_SEAS
multipliers = multipliers[mult_needed].copy()

synthetic_list = []
for scen in multipliers['scenario'].dropna().unique():
    m_s = multipliers[multipliers['scenario'] == scen].copy()
    merged = baseline_unscaled.merge(
        m_s.drop(columns=['scenario']),
        on=['sic_code'], how='left', suffixes=('', '_mult')
    )
    for c in NUMERIC_COLS_NO_SEAS:
        mult_col = f"{c}_mult"
        if mult_col not in merged:
            merged[mult_col] = 1.0
        merged[c] = merged[c] * merged[mult_col]
    merged.drop(columns=[f"{c}_mult" for c in NUMERIC_COLS_NO_SEAS], inplace=True, errors='ignore')
    merged['scenario'] = scen
    synthetic_list.append(merged)

synthetic_all = pd.concat(synthetic_list, ignore_index=True)