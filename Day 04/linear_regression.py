#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Reading csv file
dataset = pd.read_csv('data.csv')

#Labeling data as x and y
x = dataset[["Experience"]].values
y = dataset[["Salary"]].values

#Using train_test_split method
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#Using linear regression model to train
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print("Predicted Value: ", y_pred)

#Training set
plt.scatter(x_train, y_train)
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Experience vs Salary (training set)")
plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.show()

# Test set
plt.scatter(x_test, y_test, color='green')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Experience vs Salary (Test Set)')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.show()