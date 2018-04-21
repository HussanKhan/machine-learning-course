"""
SVR is from SVM
SVM is involved with classification
Basically you create wall or lines (y = mx + b) as close to data points as possible
and than find a best fit line between thoughs lines
-------------- upper bound vector
____________________ good fit vector
-------------- lower bound vector

Allows use to work with exponential data or wavy data, if correct kernal is chosen.
Kernals basically inject data to help create better models.
Kernal take inputs and output their similarites
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

# We need to feature scale, so our kernal can see similarites more clearly

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)



from sklearn.svm import SVR

# Choose the correct kernal for the problem
svr = SVR(kernel='rbf')
svr.fit(X, y)

# See prediction
# First trans form input to fit model, and then inverse_transform or decode to
# see actual value

# Transform X
p_test = sc_x.transform(np.array([[6.5]]))
# Predict transformed X
y_pred = svr.predict(p_test)
# UnTransfrom prediction
y_decode = sc_y.inverse_transform(y_pred)
print(y_decode)

# # Notice how the CEO is considered an outlier
plt.scatter(X, y, color='blue')
plt.plot(X, svr.predict(X), color='red')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
