# Data visualization using PCA
# Visualize MNIST dataset using PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target
variance_orignal = np.var(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=False)

# Clustering with SVM
svm_org = SVC(random_state=42)
svm_org.fit(X_train, y_train)
accuracy_org = svm_org.score(X_test, y_test)

# Variance of each dimension
variance = []

# Difference of each dimension (MSE)
similarity = []

# Difference of each dimension (Accuracy)
accuracy = []

# Iterate for dimension 1 ~ 64
for dim in range(1, 65):
    # PCA
    # n_components: Number of components to keep, if n_components is not set all components are kept
    pca = PCA(n_components=dim, random_state=42)
    X_pca = pca.fit_transform(X)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Clustering with SVM
    svm_pca = SVC(random_state=42)
    svm_pca.fit(X_train_pca, y_train)

    # Variance of each dimension
    variance.append(np.var(X_pca))

    # Difference of each dimension (MSE)
    similarity.append(np.sum((X - pca.inverse_transform(X_pca)) ** 2))

    # Difference of each dimension (Accuracy)
    accuracy.append(svm_pca.score(X_test_pca, y_test))

# Plot variance of each dimension
plt.figure(figsize=(20, 10))
plt.subplot(1, 4, 1)
plt.plot(range(1, 65), variance)
plt.title('Variance of each dimension')
plt.xlabel('Dimension')
plt.ylabel('Variance')

# Plot variance difference
plt.subplot(1, 4, 2)
plt.plot(range(1, 65), np.abs(variance_orignal - variance))
plt.title('Variance difference')
plt.xlabel('Dimension')
plt.ylabel('Variance difference')

# Plot MSE
plt.subplot(1, 4, 3)
plt.plot(range(1, 65), similarity)
plt.title('MSE of each dimension')
plt.xlabel('Dimension')
plt.ylabel('MSE')

# Plot accuracy
plt.subplot(1, 4, 4)
plt.plot(range(1, 65), accuracy, label='PCA')
plt.plot(range(1, 65), [accuracy_org] * 64, linestyle='--', color='red', label='Original')
plt.title('Accuracy of each dimension')
plt.xlabel('Dimension')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
