# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 2. Load dataset
data = pd.read_csv("data.csv")

# 3. Features and target
X = data[['Study Hours', 'Sleep Hours']]
y = data['Result']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 5. Create SVM model (linear kernel)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. Predict new student
print(model.predict([[5, 5]]))  # Expected Pass or Fail
