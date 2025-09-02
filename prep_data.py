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