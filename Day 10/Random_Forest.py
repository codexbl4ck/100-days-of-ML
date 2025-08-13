# 1. Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Load dataset
iris = load_iris()
x = iris.data
y = iris.target

# 3. Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
)

# 4. Create random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# 5. Pridict
y_pred = model.predict(x_test)

# 6. Accuracy
print("Accuracy: ", accuracy_score(y_test, y_pred))

# 7. Predicting new values
# Example: sepal length = 5.0, sepal width = 1.4, petal length = 1.7, petal width 0.5
print("Predicted species: ", iris.target_names[model.predict([[5.0,1.4,1.7,0.5]])[0]])
