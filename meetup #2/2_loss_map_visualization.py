# Loss map visualization

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt import BayesianOptimization, UtilityFunction

# Generate data
data = np.array([[1, 2], [2, 3], [3, 4]])

# Loss map visualization
# Define the function
def f(w, b):
    error = 0
    for d in data:
        x, y = d
        error += (w * x + b - y) ** 2
    return error

# Set the range for w and b
pbounds = {'w': (-10, 10), 'b': (-10, 10)}

# Set the optimizer
optimizer = BayesianOptimization(
    f=f,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)
optimizer.set_gp_params(normalize_y=True)
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
#
# Iteration
for i in range(10):
    # Get the next point
    next_point = optimizer.suggest(utility)
    # Calculate the error
    error = f(**next_point)
    # Update the model
    optimizer.register(params=next_point, target=-error)
    # Print the error
    print('Epoch: {}, w: {:.3f}, b: {:.3f}, error: {:.3f}'.format(i, next_point['w'], next_point['b'], error))

# Plot the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.arange(-10, 10, 0.1)
Y = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
ax.plot_surface(X, Y, Z, cmap='rainbow')
plt.show()

# Print the result
print(optimizer.max)
