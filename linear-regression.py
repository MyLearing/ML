import pandas as pd
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# print(data.head())
# print(data.tail())
# print(data.shape)

sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')

feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
X = data[['TV', 'radio', 'newspaper']]

print(X.head())
print(type(X))
print(X.shape)

y = data['sales']
y = data.sales
print(y.head())
print(type(y))
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
print(linreg.intercept_)
print(linreg.coef_)

list(zip(feature_cols, linreg.coef_))
y_pred = linreg.predict(X_test)
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]
# calculate MAE by hand
print((10 + 0 + 20 + 10)/4.)

# calculate MAE using scikit-learn
from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))

print((10**2 + 0**2 + 20**2 + 10**2)/4.)
print(metrics.mean_squared_error(true, pred))
print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4.))
print(np.sqrt(metrics.mean_squared_error(true, pred)))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# create a Python list of feature names
feature_cols = ['TV', 'Radio']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# select a Series from the DataFrame
y = data.Sales

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(X_test)

# compute the RMSE of our predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))