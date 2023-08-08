# Linear correlation after PCA
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Check Linear correlation
# 1. PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pd.DataFrame(pca.fit_transform(X))
X_pca.columns = ['PC1', 'PC2']

# 2. Linear correlation heatmap
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
sns.heatmap(pd.DataFrame(X).corr(), annot=True, cmap='YlGnBu')
plt.title('Original Data')
plt.subplot(1, 2, 2)
sns.heatmap(pd.DataFrame(X_pca).corr(), annot=True, cmap='YlGnBu')
plt.title('PCA Data')
plt.show()
