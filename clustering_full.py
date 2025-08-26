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
