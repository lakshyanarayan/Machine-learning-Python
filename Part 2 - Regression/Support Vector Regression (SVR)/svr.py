# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling(We have to to do feature scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1,1))

# Fitting the SVR Regression Model to the dataset
# Create your regressor here
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)


temp=np.array([6.5])
# Predicting a new result
#As we have done feature scaling here we have to transform 6.5 as well
#Also we have to transform into array(Many ways)
#Be orecise and know where to use sc_X and sc_y and what are the reasons
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(temp.reshape(1,-1))))

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model(SVR))')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model(SVR))')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()