import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_prediction))
mae = mean_absolute_error(y_test, y_prediction)
print(f"RMSE is: {rmse}")
print(f"MAE is: {mae}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_prediction, color = "blue", edgecolors="k", alpha=0.7)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], color="red", lw=2)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual V/s Predicted ")
plt.show()