# msc-project-source-code-files-24-25-QiaochuDing

This GitHub repository contains the data and code files for an MSc Data Science Project (2025) on **Integrating Systems Thinking and Machine Learning to Manage Migrant Dependencies in the UK Labour Market**.  

An index of contents is listed below.

---

## 1. Data

| File name              | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `full_data.xlsx`        | Dataset of migrant dependency metrics for UK industries, compiled from the project Literature Review. |
| `modified_data.xlsx`    | Modified version of `full_data`, excluding sectors T and U due to data quality issues. |
| `multiplier_matrix.xlsx`| Table of multipliers featuring four scenarios to generate synthetic data.  |

---

## 2. Outputs

| File name                     | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `modified_data_clusters.xlsx`  | Export of the modified dataset with allocated clusters from k-means clustering. |
| `synthetic_clusters_drift.xlsx`| Export of synthetic data instances with drift calculations from baseline centroids and checks for cluster reassignment. |

---

## 3. Helpers

| File name          | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `readme.md`        | Introduction to this repository, information on files, and instructions for accessing outputs. |
| `requirements.txt` | List of Python packages required for this project.                          |
| `labour-deps-dashboard/` | Sub-folder containing Data and Outputs to build the dashboard.         |

---

## 4. Code

| File name                | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `clustering_full.py`      | Exploratory analysis, pre-processing, and clustering performed on the `full_data` file. |
| `clustering_modified.py`  | Exploratory analysis, pre-processing, clustering, and synthetic data generation performed on the `modified_data` file. |
| `prep.py`                 | Prepares data for embedding into the interactive dashboard.                |
| `dashboard.py`            | Configures, executes, and deploys the interactive dashboard.               |

---

## Instructions

### To replicate final clustering outputs
1. Save `modified_data.xlsx` and `multiplier_matrix.xlsx` as inputs.  
2. Run `clustering_modified.py`.  
3. Review output files:  
   - `modified_data_clusters.xlsx`  
   - `synthetic_clusters_drift.xlsx`  
4. *(Optional)* Replicate the above steps with `full_data.xlsx` and `clustering_full.py` to compare outputs using all sectors.

---

### To view the interactive dashboard
1. Go to [the hosted dashboard](https://birkbeck-msc-project-source-code-files-24-25-q-dashboard-xv3lai.streamlit.app/).
2. Alternatively, run:  
   ```bash
   python prep.py
   streamlit run dashboard.py