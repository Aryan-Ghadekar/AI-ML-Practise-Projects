# House Prediction using Linear Regression


import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # We can use MinMaxScaler for Normalization
# from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Normalization
# min_max_scaler = MinMaxScaler().fit(X_test)
# X_norm = min_max_scaler.transform(X)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)


print("Linear Regression Evaluation Metrics:")
print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

