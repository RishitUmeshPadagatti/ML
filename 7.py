# Linear Regression and Polynomial Regression for Boston Housing Dataset and AutoMPG dataset

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# LINEAR REGRESSION

housing = fetch_california_housing()
x_h = housing.data[:, [0]]
y_h = housing.target

lin = LinearRegression()
lin.fit(x_h, y_h)

print("Linear R2 Score: ", lin.score(x_h, y_h))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_h[:300], y_h[:300], alpha=0.3)
plt.plot(
    x_h[:300],
    lin.predict(x_h[:300]),
    color='red',
    linewidth=2,
    label="Regression Line"
)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression Fit")
plt.legend()



# POLYNOMIAL REGRESSION

auto_df = pd.read_csv("7.auto-mpg.csv", na_values='?')
auto_df = auto_df.dropna()

X_auto = auto_df[['horsepower']].astype(float).values
y_auto = auto_df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(X_auto, y_auto, test_size=0.2, random_state=42)

poly_model = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearRegression()
)
poly_model.fit(X_train, y_train)
y_poly_pred = poly_model.predict(X_test)

print("Polynomial R2 Score: ", poly_model.score(X_test, y_test))

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.5, label="Actual")
sorted_data = sorted(zip(X_test.flatten(), y_poly_pred))
X_sorted = np.array([x for x, y in sorted_data])
y_sorted = np.array([y for x, y in sorted_data])
plt.plot(X_sorted, y_sorted, color="red", linewidth=2, label="Predicted")
plt.title("Polynomial Regression")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()

plt.tight_layout()
plt.show()