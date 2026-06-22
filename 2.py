# Develop a program to Compute the correlation matrix to understand the relationships between pairs of features. 
# Visualize the correlation matrix using a heatmap to know which variables have strong positive/negative correlations. 
# Create a pair plot to visualize pairwise relationships between features. Use California Housing dataset.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of California Housing Features')
plt.show()

sns.pairplot(df, diag_kind='kde') # kde = smooth curves
plt.suptitle('Pair Plot of California Housing Features', y=1.02)
plt.show()