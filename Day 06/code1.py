import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
x = df[['Hours_Studied','Sleep_Hours','Attendance']]
y = df['Passed']

model = LogisticRegression()
model.fit(x,y)

pred = model.predict([[4,8,70]])
print("Will the student pass?", "yes" if pred[0] == 1 else "No")

plt.scatter(df['Hours_Studied'], df['Passed'])
plt.xlabel('Hours Studied')
plt.ylabel('Passed')
plt.title('Hours Studied vs Passing')
plt.show()
