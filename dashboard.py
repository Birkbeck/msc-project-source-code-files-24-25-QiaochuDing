# Import libraries
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# Define constants
OUT_DIR = Path("labour-deps-dashboard/output")
KUMU_URL = "https://embed.kumu.io/3b21f59a42a0ea30e5dda4662cf3897a"

# Page configuration
st.set_page_config(page_title="UK Labour Market Dependencies Dashboard", layout="wide")
st.title("UK Labour Market Dependencies Dashboard")
st.caption("Explore the Causal Loop Diagram, industry sector clusters, summary statistics, and PCA visualisation.")

# Kumu
tab_map, tab_clusters, tab_pca = st.tabs(["Systems Map", "Clusters", "PCA"])

with tab_map:
    st.markdown("#### Causal Loop Diagram")
    st.components.v1.iframe(KUMU_URL, height=700)

# Load data
@st.cache_data
def load_assets():
    baseline = pd.read_csv(OUT_DIR / "baseline_with_clusters.csv")
    pca_coords = pd.read_csv(OUT_DIR / "pca_coords.csv")
    pca_explained = pd.read_csv(OUT_DIR / "pca_explained.csv")
    drift = pd.read_csv(OUT_DIR / "synthetic_clusters_drift.csv")
    summary = pd.read_csv(OUT_DIR / "cluster_summary.csv", header=[0,1])
    return baseline, pca_coords, pca_explained, drift, summary

baseline, pca_coords, pca_explained, drift, cluster_summary = load_assets()

# Sidebar settings
st.sidebar.header("Settings")
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

# Other tabs
with tab_clusters:
    st.markdown("#### Cluster overview")
    view_df = baseline if selected_cluster == "All" else baseline[baseline['cluster_label'] == int(selected_cluster)]

    kpi_cols = (shown_cols[:3] if len(shown_cols) >= 3 else shown_cols) or numeric_cols[:3]
    cols = st.columns(len(kpi_cols))
    for col, metric in zip(cols, kpi_cols):
        with col:
            st.metric(label=f"{metric} (mean)", value=f"{view_df[metric].mean():.2f}")

    st.markdown("##### Summary statistics")
    st.dataframe(cluster_summary if isinstance(cluster_summary.columns, pd.MultiIndex)
                 else (view_df[shown_cols]).agg(['mean','median','min','max','count']).round(2))

    st.markdown("##### Industries in view")
    st.dataframe(view_df[['industry','sic_code','cluster_label'] + shown_cols].sort_values(['cluster_label','industry']))

    metric_for_bar = st.selectbox("Bar chart metric", options=shown_cols, index=0)
    bar_data = view_df[['industry', metric_for_bar]].sort_values(metric_for_bar, ascending=False)
    bar = (alt.Chart(bar_data)
           .mark_bar()
           .encode(
               x=alt.X('industry:N', sort='-y'),
               y=alt.Y(f'{metric_for_bar}:Q'),
               tooltip=['industry', metric_for_bar]
           )
           .properties(height=400)
           .interactive())
    st.altair_chart(bar, use_container_width=True)

with tab_pca:
    st.markdown("#### PCA (2D projection of industries)")
    pca_view = pca_coords if selected_cluster == "All" else pca_coords[pca_coords['cluster_label'] == int(selected_cluster)]

    base = (alt.Chart(pca_view)
            .mark_circle(size=80)
            .encode(
                x='pc1:Q', y='pc2:Q',
                color=alt.Color('cluster_label:N', legend=alt.Legend(title="Cluster")),
                tooltip=['industry','cluster_label','pc1','pc2']
            ))

    chart = base
    if highlight_sector != "None":
        highlight = pca_coords[pca_coords['industry'] == highlight_sector]
        if not highlight.empty:
            h = (alt.Chart(highlight)
                 .mark_point(filled=True, shape='triangle-up', size=200)
                 .encode(x='pc1:Q', y='pc2:Q', color=alt.value('red'),
                         tooltip=['industry','cluster_label']))
            chart = base + h

    st.altair_chart(chart.interactive().properties(height=500), use_container_width=True)