# 1. Importing Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 2. Load Dataset
dataset = pd.read_csv('buyers.csv')

# 3. Encode categorical variables
le = LabelEncoder()
for col in dataset.columns:
    dataset[col] = le.fit_transform(dataset[col])
    
# 4. Features and Target
x = dataset[['Age', 'Income', 'Student']]
y = dataset['Buys']

# 5. Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size= 0.3, random_state=42
)

# 6. Model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# 7. Predict
y_pred = model.predict(x_test)

# 8. Accuracy
print("Accuracy: ", accuracy_score(y_test, y_pred))

# 9. Predict new case
print(model.predict([[0,1,1]]))