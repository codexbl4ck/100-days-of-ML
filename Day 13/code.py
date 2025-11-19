# ---------------------------
# DAY 13 - LINEAR REGRESSION + EVALUATION (FIXED)
# ---------------------------

# 1️⃣ Import Libraries
import numpy as np
import pandas as pd

# 2️⃣ Load Dataset
dataset = pd.read_csv('data1.csv')
X = dataset.iloc[:, :-1].values   # Features
y = dataset.iloc[:, -1].values    # Salary (target)

# 3️⃣ Handle Missing Numerical Data (Age) - Column index 1
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
X[:, 1:2] = num_imputer.fit_transform(X[:, 1:2])

# 4️⃣ Handle Missing Categorical Data (Country, Gender, Department)
cat_imputer = SimpleImputer(strategy='most_frequent')
X[:, [0, 2, 3]] = cat_imputer.fit_transform(X[:, [0, 2, 3]])

# 5️⃣ One-Hot Encode Categorical Columns (Country, Gender, Department)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0, 2, 3])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# 6️⃣ Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 7️⃣ Feature Scaling (only Age column → now last column)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, -1:] = sc.fit_transform(X_train[:, -1:])
X_test[:, -1:] = sc.transform(X_test[:, -1:])

# 8️⃣ Train Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 9️⃣ Predict Test Data
y_pred = regressor.predict(X_test)

# 🔟 Compare Predictions vs Actual
print("\nPrediction vs Actual:\n")
print(np.concatenate((y_pred.reshape(-1,1), y_test.reshape(-1,1)), axis=1))

# 1️⃣1️⃣ Model Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\nModel Evaluation:")
print("MAE  =", mean_absolute_error(y_test, y_pred))
print("MSE  =", mean_squared_error(y_test, y_pred))
print("RMSE =", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score =", r2_score(y_test, y_pred))
