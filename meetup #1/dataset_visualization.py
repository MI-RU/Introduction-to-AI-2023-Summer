# Data scaling and visualization exercise

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

iris = load_iris()
data = iris.data
target = iris.target

# Create a dataframe with the iris data and target
df = pd.DataFrame(data, columns=iris.feature_names)
df['target'] = target

# Create scaled data using MinMaxScaler and StandardScaler
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

scaled_minmax = minmax_scaler.fit_transform(data)
scaled_standard = standard_scaler.fit_transform(data)

# Create a dataframe with the scaled data
df_scaled_minmax = pd.DataFrame(scaled_minmax, columns=iris.feature_names)
df_scaled_minmax['target'] = target

df_scaled_standard = pd.DataFrame(scaled_standard, columns=iris.feature_names)
df_scaled_standard['target'] = target

# Plot the data
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.ylim(-2, 8)
plt.xlim(-2, 8)
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target', data=df)
plt.title('Original Data')

plt.subplot(1, 3, 2)
plt.ylim(-2, 8)
plt.xlim(-2, 8)
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target', data=df_scaled_minmax)
plt.title('MinMax Scaled Data')

plt.subplot(1, 3, 3)
plt.ylim(-2, 8)
plt.xlim(-2, 8)
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target', data=df_scaled_standard)
plt.title('Standard Scaled Data')

plt.show()
