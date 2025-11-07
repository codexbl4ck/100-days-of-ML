# Day 11 - Train-Test Split and Cross Validation

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. Load dataset
data = load_iris()
X = data.data
y = data.target

# 2. Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Test accuracy on test set
test_score = model.score(X_test, y_test)
print("Test Accuracy:", test_score)

# 5. Cross-validation accuracy
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))
