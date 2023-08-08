# Data visualization using PCA
# Visualize MNIST dataset using PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# PCA
# n_components: Number of components to keep, if n_components is not set all components are kept
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X)

# Plot MNIST each digit
# 0 ~ 9
for i in range (10):
    plt.figure(figsize=(15, 3))
    # Compare Image
    # Original Image
    var_original = np.var(X[i])
    plt.subplot(1, 3, 1)
    plt.imshow(X[i].reshape(8, 8), cmap='gray')
    plt.title(f'Original Image (Variance: {var_original:.2f}))')
    
    # inverse-PCA Reconstructed Image
    var_pca = np.var(pca.inverse_transform(X_pca[i]))
    plt.subplot(1, 3, 2)
    plt.imshow(pca.inverse_transform(X_pca[i]).reshape(8, 8), cmap='gray')
    plt.title(f'inverse-PCA Reconstructed Image (Variance: {var_pca:.2f})')

    # Difference
    similarity = np.sum((X[i] - pca.inverse_transform(X_pca[i])) ** 2)  # MSE
    plt.subplot(1, 3, 3)
    plt.imshow(X[i].reshape(8, 8) - pca.inverse_transform(X_pca[i]).reshape(8, 8), cmap='gray')
    plt.title(f'Difference (MSE: {similarity:.2f})')
    plt.show()
