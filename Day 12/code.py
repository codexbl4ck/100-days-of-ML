# Day 12 - Feature Scaling

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# 1. Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

print("Before Scaling:\n", X.head())

# 2. Normalization
minmax_scaler = MinMaxScaler()
X_norm = minmax_scaler.fit_transform(X)
print("\nAfter Min-Max Scaling:\n", pd.DataFrame(X_norm, columns=X.columns).head())

# 3. Standardization
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)
print("\nAfter Standardization:\n", pd.DataFrame(X_std, columns=X.columns).head())
