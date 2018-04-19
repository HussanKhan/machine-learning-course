# In machine learning we use independant varibles to guess dependant variables

# Libraies we will use all the time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import datasets using pandas
dataset = pd.read_csv('Data.csv')
# Now dataset is a pandas dataframe

# We now need to seperate the dependant variable from the data, but keep index positon
# the independant data is the matrix of features

# First see all the columns
# print(dataset.columns)

# Now create two seperate dataframes, one for x and other for y(dependant)
# Adding the values in the end, makes a matrix
X = dataset[['Country', 'Age', 'Salary']].values
y = dataset['Purchased'].values

# the x matrix and the y matrix are synced by index
# print('X or Independant Variables in matrix' + '\n')
# print(X)
# print('\n'+ 'Y or dependant Variables in matrix' + '\n')
# print(y)
# print('\n' + 'Synced by index')

# !!!!Dealing with missing data!!!!!

# We can remove the dataset or fill the missing spit with the mean
# Helps us fill missing data
# If answer is missing, you must remove it
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
                # What we are looking for, what we fill it with, column or acix

# Now we fit the imputer to the matrix
imputer = imputer.fit(X[:,1:3])
# And now we apply the fit
X[:,1:3] = imputer.transform(X[:,1:3])

# Dealing with categorical data, we need to makes the strings into numbers
# or encode it

# This library helps us convert category data in numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Create an instance
le_x = LabelEncoder()

# Call the fit_transform method and select the column, and we set it equal to it
X[:,0] = le_x.fit_transform(X[:,0])
# Say all rows and first column

# Now we need to prevent the alogirthms of ranking one number or id higher than the other
# We split each category into it's own column, and use 1 or zero to signaify is it's selected
# On and off is the simplest method

# Make and instance of the class,a nd specify colum
oee = OneHotEncoder(categorical_features=[0])
# remember unique toarray
X = oee.fit_transform(X).toarray()

# Now do the same thing for y variable
le_y = LabelEncoder()

# Since it's only two options, this is fine
y = le_y.fit_transform(y)

# Split the data into two batch, the batch for training and a batch for testing

from sklearn.model_selection import train_test_split

# Simply create 4 vars and call instance to split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23, test_size = 0.2)

#!!!!!! FEATURE SCALING

# Distance between two points on a graph
# d = sqrt([x2-x1]^2 + [y2-y1]^2)
# Higher numbers will artifically increase distance

# Now that data is split, we need to scale our features. Our algorithem will
# mistake higher valies as more important like salary vs age, so we need to normalize
# the data so the model treats them fairly

# Use this Libray
from sklearn.preprocessing import StandardScaler

#This scales your data so mean value is zero and std is 1
#Create an instance
scale = StandardScaler()

# fit it to the training set and test set
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


print(X_train)
print(y_train)
