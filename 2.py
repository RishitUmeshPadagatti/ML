# Develop a program to Compute the correlation matrix to understand the relationships between pairs of features. 
# Visualize the correlation matrix using a heatmap to know which variables have strong positive/negative correlations. 
# Create a pair plot to visualize pairwise relationships between features. Use California Housing dataset.


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

california_data = fetch_california_housing(as_frame=True)
data = california_data.frame

corelation_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corelation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of California Housing Features')
plt.show()

sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pair Plot of California Housing Features', y=1.02)
plt.show()