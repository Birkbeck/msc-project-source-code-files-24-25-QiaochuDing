import pandas as pd
import numpy as np

# Load data
df = pd.read_excel("/Users/QiaoChuDing_1/Desktop/Birkbeck/2024-2025/4. Project/msc-project-source-code-files-24-25-QiaochuDing/compiled_data.xlsx")

print(df.head())

# Replace misisng values
df.replace("-", np.nan, inplace=True)

# Convert numeric columns to float
numeric_cols = [
    'workforce_nonUK (2024)',
    'vacancies_per_100_jobs (2024)',
    'ssv_density (2022)',
    'med_annual_earnings_differential (2023)',
    'visa_grants (2024)',
    'jobs_at_risk_of_automation (2017)'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Impute missing values with mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Encode categorical variables
seasonality_map = {'Low': 1, 'Medium': 2, 'High': 3}
df['seasonality_encoded'] = df['seasonality'].map(seasonality_map)

# Standardise
from sklearn.preprocessing import StandardScaler
features_to_scale = numeric_cols + ['seasonality_encoded']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features_to_scale])
df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale)

print(df_scaled.head())

df_scaled['industry'] = df['industry']
df_scaled['SIC_code (2007)'] = df['SIC_code (2007)']

print(df_scaled)
