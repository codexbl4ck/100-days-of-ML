# Importing Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Reading dataset
dataset = pd.read_csv('data.csv')

# Labeling data as x and y
x = dataset[['Experience','Age','Education_Level']]
y = dataset[['Salary']]

# Training model
model = LinearRegression()
model.fit(x,y)

# Pridicting values
pred_salary = model.predict([[5,30,2]])
print("Pridicted Salary:", pred_salary[0])

#using evaluation techniques
y_pred = model.predict(x)
mse = mean_squared_error(y_pred, y)
print("Mean Square Error: ", mse)

r2 = r2_score(y_pred, y)
print("R² Score:", r2)