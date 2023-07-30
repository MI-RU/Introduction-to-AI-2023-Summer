# Linear regression
import numpy as np
import matplotlib.pyplot as plt

# Generate data
# 1, 2
# 2, 3
# 3, 4
data = np.array([[1, 2], [2, 3], [3, 4]])

# Initial condition
# y = x
w = 1
b = 0

# Learning rate
lr = 0.01

# Iteration
for i in range(50):
    # Calculate the error
    # Error: Mean Squared Error
    error = 0
    for d in data:
        x, y = d
        error += (w * x + b - y) ** 2

    # Calculate the gradient
    w_grad = 0
    b_grad = 0
    for d in data:
        x, y = d
        w_grad += (w * x + b - y) * x
        b_grad += (w * x + b - y)

    # Update the parameters
    w -= lr * w_grad
    b -= lr * b_grad

    # Print the error
    print('Epoch: {}, w: {:.3f}, b: {:.3f}, error: {:.3f}'.format(i, w, b, error))

    # Plot the result
    plt.cla()
    plt.scatter(data[:, 0], data[:, 1])
    plt.plot(data[:, 0], data[:, 0] * w + b)
    plt.pause(0.1)

plt.show()
