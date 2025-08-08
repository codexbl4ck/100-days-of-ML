# 1. Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2. Load dataset
dataset = pd.read_csv('fruits.csv')

# 3. Labeling features as (x) and (y)
x = dataset[['Weight (g)','Texture (0 = Smooth 1 = Rough)']]
y = dataset['Fruit']

# 4. Split into training and testing set
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
)

# 5. Creating KNN model with k=3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

# 6. Pridict test data
y_pred = model.predict(x_test)

# 7. Accuracy
print("Accuracy Score: ",accuracy_score(y_test, y_pred))

# 8. Pridicting new value
print(model.predict([[155,0]]))  