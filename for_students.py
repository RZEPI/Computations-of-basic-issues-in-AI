import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 andtheta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

x_matrix = np.c_[np.ones((len(x_train), 1)), x_train]


# TODO: calculate closed-form solution
theta_best = np.linalg.inv((x_matrix.T).dot(x_matrix))
theta_best = theta_best.dot(x_matrix.T).dot(y_train)

print("theta_best: " + str(theta_best))

# TODO: calculate error
mse = 0
for i in range(0, len(x_matrix)):
    mse += (theta_best.dot(x_matrix[i]) - y_train[i])**2
mse /= len(x_train)
print("MSE(theta) " + str(mse))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
mean_x = np.mean(x_train)
standard_deviation_x = np.std(x_train)

mean_y = np.mean(y_train)
standard_deviation_y = np.std(y_train)

x_train = (x_train - mean_x) / standard_deviation_x
x_test = (x_test - mean_x) / standard_deviation_x

y_train = (y_train - mean_y) / standard_deviation_y
y_test = (y_test - mean_y) / standard_deviation_y

# TODO: calculate theta using Batch Gradient Descent
step = 0.01
n_iterations = 10000
x_matrix = np.c_[np.ones((len(x_train), 1)), x_train]
theta = np.array([0, 0])

# batch gradient descent
for i in range(n_iterations):
    gradient = 2/len(x_train) * x_matrix.T.dot(x_matrix.dot(theta) - y_train)
    theta = theta - step * gradient

print("theta: ", theta)
# TODO: calculate error

mse = 0
for i in range(0, len(x_matrix)):
    mse += (theta.dot(x_matrix[i]) - y_train[i])**2
mse /= len(x_train)

print("MSE(theta): " + str(mse))
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()