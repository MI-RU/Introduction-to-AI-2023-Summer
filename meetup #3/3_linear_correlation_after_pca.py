# Linear correlation after PCA
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target
num_features = len(diabetes.feature_names)

# Check Linear correlation
# 1. PCA
pca = PCA(n_components=num_features, random_state=42)
X_pca = pd.DataFrame(pca.fit_transform(X))
X_pca.columns = [f'PC{dim}' for dim in range(1, num_features + 1)]

# Linear correlation
corr_org = pd.DataFrame(X).corr()
corr_pca = pd.DataFrame(X_pca).corr()
# Cut off if correlation is less than 1e-10
corr_org[corr_org < np.abs(1e-10)] = 0
corr_pca[corr_pca < np.abs(1e-10)] = 0

# 2. Linear correlation heatmap
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
sns.heatmap(corr_org, annot=True, cmap='YlGnBu')
plt.title('Original Data')
plt.subplot(1, 2, 2)
sns.heatmap(corr_pca, annot=True, cmap='YlGnBu')
plt.title('PCA Data')
plt.show()
