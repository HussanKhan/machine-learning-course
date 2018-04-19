# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
    Main equation:
    y = b + mx
    (dependent) = (minimum point) + (rate of change)(independent)

    for each point we do this (Total sum for each point)
    (y-(mx + b))^2 + ... (yn - (mx+b))^2

    After summing we get total errors
    Now we adjust m and b and run again, until our total error is lowest
    (usally most occuring errorsum)
"""

# Importing the dataset and preproces
dataset = pd.read_csv('Salary_Data.csv')

#See which data is dependent and independent, we try to guess dependent
# print(dataset.head())

#Looks like Salary is dependent on years experiance, remember to extract values for
# 2-dim array. Remember the points are linked by index. Use iloc for accracy, [rows,colums]
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
# print(X)
# print(y)

# Now split into training set and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32)

# We don't need to worry about feature scaling, because library does it for us, and since, there is only 1 feature
from sklearn.linear_model import LinearRegression
# Create an instance to call methods off
lr = LinearRegression()
# Fit the algo with the data
lr.fit(X_train, y_train)

# Now let's test our model, we have 95.5% accuracy
print(lr.score(X_test, y_test))
pred = lr.predict(4)
# Extract value from numpy array
print(pred[0])

# Let's graph the model
# This is creating the graph, usally first instance
plt.scatter(X_train, y_train, color='red')
# Plot line, X is normal, and y are predictions
# This is plotting things on the graph
plt.plot(X_train, lr.predict(X_train), color='blue')
plt.scatter(X_test, y_test, color='green')
# Just add some labels
plt.title('Salary v Experiance')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')

plt.show()
