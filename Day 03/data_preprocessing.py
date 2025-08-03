import numpy as np
import pandas as pd

# 1. Load the dataset
dataset = pd.read_csv('data1.csv')
print(dataset.head())
print(dataset.columns)

X = dataset.iloc[:, :-1].values  # All columns except Salary
y = dataset.iloc[:, -1].values   # Salary column

# 2. Handle missing numerical data (only Age)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:2] = imputer.fit_transform(X[:, 1:2])  # Age is column 1

# 3. Handle missing categorical data (Gender)
cat_imputer = SimpleImputer(strategy='most_frequent')
X[:, 2:3] = cat_imputer.fit_transform(X[:, 2:3])  # Gender is column 2

# 4. One-hot encode Gender and Department (columns 2 and 3)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [2, 3])],  # Gender & Department
    remainder='passthrough'
)
X = ct.fit_transform(X)

# 5. Encode target variable y (Salary) — optional, only if classification (e.g., "Purchased")
# Since Salary is numerical, you don't need to encode it unless it's categorical like Yes/No

# 6. Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 7. Feature scaling on numerical columns only
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train[:, -1:] = sc.fit_transform(X_train[:, -1:])  # Only scale Age (last column)
X_test[:, -1:] = sc.transform(X_test[:, -1:])
