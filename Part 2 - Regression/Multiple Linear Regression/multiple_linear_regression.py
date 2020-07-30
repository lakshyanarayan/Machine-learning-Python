# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data
#Encoding the independent variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# Encoding the dependent variable (no need here)
#y = LabelEncoder().fit_transform(y)
#Avoiding dummy variable trap
X=X[:,1:]
#You need not do this because library is taking care of that for you
#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#x_opt will only contain independent variable which have high impact on dependent variable(profit)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_ols.summary() 

"""
lesser the p value ,
more that independet varaible will be significant to the D.variable

"""
# remove the 4th column as it has the highest value 
X_opt = X[:, [0, 1, 3, 4, 5]] 
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_ols.summary() 

# remove the 5th column as it has the highest value 
X_opt = X[:, [0, 3, 4, 5]] 
regressor_ols = sm.OLS(endog = y, exog =X_opt).fit() 
regressor_ols.summary() 

# remove the 3rd column as it has the highest value 
X_opt = X[:, [0, 3, 5]] 
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_ols.summary() 

# remove the 2nd column as it has the highest value 
X_opt = X[:, [0, 3]] 
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_ols.summary() 






