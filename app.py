import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from pathlib import Path

st.set_page_config(page_title="UK Labour Market Dependencies Dashboard", layout="wide")
OUT_DIR = Path("outputs")

KUMU_URL = "https://embed.kumu.io/3f568e13ef2bb7522bdb8baf3a349991"

@st.cache_data
def load_assets():
    baseline = pd.read_csv(OUT_DIR / "baseline_with_clusters.csv")
    pca_coords = pd.read_csv(OUT_DIR / "pca_coords.csv")
    pca_explained = pd.read_csv(OUT_DIR / "pca_explained.csv")
    drift = pd.read_csv(OUT_DIR / "synthetic_clusters_drift.csv")
    try:
        summary = pd.read_csv(OUT_DIR / "cluster_summary.csv", header=[0,1])
    except Exception:
        keep_cols = [c for c in baseline.columns if c not in ['industry','sic_code','cluster_label']]
        summary = (baseline.groupby('cluster_label')[keep_cols]
                   .agg(['mean','median','min','max','count']).round(2))
    return baseline, pca_coords, pca_explained, drift, summary

baseline, pca_coords, pca_explained, drift, cluster_summary = load_assets()

# ---- Sidebar ----
st.sidebar.header("Controls")
clusters = sorted(baseline['cluster_label'].unique())
selected_cluster = st.sidebar.selectbox("Cluster", options=["All"] + [str(c) for c in clusters], index=0)

id_like = {'industry','sic_code','cluster_label','seasonality'}
numeric_cols = [c for c in baseline.columns if c not in id_like and pd.api.types.is_numeric_dtype(baseline[c])]
default_show = ['non_UK_workforce','vacancy_rate','ssv_density','med_annual_wage_differential']
shown_cols = st.sidebar.multiselect(
    "Indicators to summarise",
    options=numeric_cols,
    default=[c for c in default_show if c in numeric_cols] or numeric_cols[:4]
)

sector_options = ["None"] + sorted(baseline['industry'].astype(str).unique())
highlight_sector = st.sidebar.selectbox("Highlight sector (optional)", options=sector_options, index=0)