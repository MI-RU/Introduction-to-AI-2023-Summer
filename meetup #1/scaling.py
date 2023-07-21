# Standard scaler and minmax scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from matplotlib import pyplot as plt

data = np.array([-1, 1, 3, 5, 7, 9])

minmax = MinMaxScaler()
standard = StandardScaler()

scaled_minmax = minmax.fit_transform(data.reshape(-1, 1))
scaled_standard = standard.fit_transform(data.reshape(-1, 1))

print(f"Data: {data}")
print(f"MinMax: {scaled_minmax.reshape(1, -1)[0]}")
print(f"Standard: {scaled_standard.reshape(1, -1)[0]}")

# Plot the data
plt.plot(data, label="Data")
plt.plot(scaled_minmax, label="MinMax")
plt.plot(scaled_standard, label="Standard")
plt.legend()
plt.show()
