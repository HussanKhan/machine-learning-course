"""
    Sometimes polynomial regression is better for predicitng.

    y = b0 + b1x1 + b2x2^2 + ... + bnxn^n

    The curve sometimes allows for better fitting of exponential growth.

    We are still predicting y. And we still only chnage weights through b
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# ['Position', 'Level', 'Salary']
# print(dataset.columns)

#Create two different matrixes
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

# We don't have enough data for seperate sets, so we will use all data possible
# We nned to prep the data by adding polynomial preprocessing
# This will transform the matrix, by adding powers of varibles
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# Degree means how many times it will be powered will go
# Increasing degrees makes the fit line fit much better
# Try playing around with it, but be careful of overfitting
pf = PolynomialFeatures(degree=3)
# Now fit and transform the X matrix
X_p = pf.fit_transform(X)

# We had to add the poly features seperatly because we are using the same
# LinearRegression

lr.fit(X_p, y)

# Now let's see the result
plt.scatter(X, y, color='blue')
plt.plot(X, lr.predict(X_p), color='red')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
