# Data visualization using PCA
# Visualize MNIST dataset using PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# PCA
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X)

# Plot MNIST each digit
# 0 ~ 9
for i in range (10):
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 3, 1)
    plt.imshow(X[i].reshape(8, 8), cmap='gray')
    plt.title('Original Image')
    plt.subplot(3, 3, 2)
    plt.imshow(pca.inverse_transform(X_pca[i]).reshape(8, 8), cmap='gray')
    plt.title('PCA Image')
    plt.subplot(3, 3, 3)
    plt.imshow((X[i] - pca.inverse_transform(X_pca[i])).reshape(8, 8), cmap='gray')
    plt.title('Difference Image')
    plt.show()
