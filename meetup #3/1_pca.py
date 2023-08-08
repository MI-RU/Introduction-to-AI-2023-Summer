# Data visualization using PCA
# Visualize MNIST dataset using PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

components = 10

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# PCA Digit 0 to 9
for digit in range(10):
    globals()[f'X_{digit}'] = X[y == digit]
    globals()[f'pca_{digit}'] = PCA(n_components=components, random_state=42)
    globals()[f'X_{digit}_pca'] = globals()[f'pca_{digit}'].fit_transform(globals()[f'X_{digit}'])
    globals()[f'X_{digit}_pca_inv'] = globals()[f'pca_{digit}'].inverse_transform(globals()[f'X_{digit}_pca'])

# Plot MNIST each digit
# 0 ~ 9
for digit in range (10):
    plt.figure(figsize=(15, 3))

    # Original
    plt.subplot(1, 4, 1)
    plt.imshow(globals()[f'X_{digit}'][0].reshape(8, 8), cmap='gray')    
    plt.title('Original')

    # PCA Inverse
    plt.subplot(1, 4, 2)
    plt.imshow(globals()[f'X_{digit}_pca_inv'][0].reshape(8, 8), cmap='gray')
    plt.title('PCA')
    
    # PCA
    plt.subplot(1, 4, 3)
    plt.imshow(globals()[f'X_{digit}_pca'][0].reshape(1, components), cmap='gray')
    plt.title(f'PCA (n_components={components})')
    
    # Difference MSE
    plt.subplot(1, 4, 4)
    plt.imshow((globals()[f'X_{digit}'][0] - globals()[f'X_{digit}_pca_inv'][0]).reshape(8, 8), cmap='gray')
    plt.title(f'Difference (MSE): {np.sum((globals()[f"X_{digit}"][0] - globals()[f"X_{digit}_pca_inv"][0]) ** 2):.2f}')
    plt.show()
