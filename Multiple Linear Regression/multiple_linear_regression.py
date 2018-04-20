"""
This is very similar to linear regression, but instead of this y = mx + b
equation looks like this.

y = b + mx + mx ... mx

Each X independent varible has a different weight

This involves multiple dimensions. Imagine having multiple layers of scatter graphss,
and you have to find a line that fits all those points together best.

Each unqiue equation of y = b + mx, would have it's own plane and own unique
best fit line.

And one line would be drawn to account for each plane, but adjusting the weights
or m of each independent varible
"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')

# Get names of all columns
# print(dataset.columns)
# ['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']

# Find dependent variable
# We are gonna predict profit
# Create X and y matrices, remember, they are linked by index
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4:5].values

# Since we hace string (states) data, we need to encode it.
# Here we are creating dummy varibles
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

le_x = LabelEncoder()
# Find column with string, and fit and set equal to. This changes strings into numbers
X[:,3] = le_x.fit_transform(X[:,3])

# Now we split it into columsn of only 1 and 0
ohe = OneHotEncoder(categorical_features= [3])
# And we set it equal to the whole array
X = ohe.fit_transform(X).toarray()

# !!!!!!!! Avoid Dummy Varible Trap
# ALWAYS REMOVE ONE DUMMY VARIABLE
# Library does this for us
# X = [:, 1:]

#Split to training set and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=32)

# Now fit model with training set
# Feature scaling is done automatically

from sklearn.linear_model import LinearRegression
# All we have to do is create an instance and fit it
lr = LinearRegression()
lr.fit(X_train, y_train)

# We have 90% accuracy of preditcing profit
print('Score: ' + str(round(100 * lr.score(X_test, y_test))) + '%')

y_pred = lr.predict(X_test)
index = 0
for i in y_pred:
    print(str(index) + ': ' + '$' + str(round(i[0])) + ' Real: ' +  '$' + str(round(y_test[index][0])))
    index += 1

# Backward elimination, remove varibles that are not very important in preditcing
