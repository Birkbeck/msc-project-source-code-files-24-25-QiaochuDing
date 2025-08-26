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