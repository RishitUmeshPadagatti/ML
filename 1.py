# Develop a program to create histograms for all numerical features and analyze the distribution of each feature. 
# Generate box plots for all numerical features and identify any outliers. Use California Housing dataset.

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import fetch_california_housing 

data = fetch_california_housing(as_frame=True)
df = data.frame

print("Dataset Shape: ", df.shape)
print("Dataset Columns: ", df.columns)

print("\nGenerating histograms...")
df.hist(bins=20, figsize=(14, 10), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features', fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


print("\nGenerating box plots...")
plt.figure(figsize=(14, 10))

for i, col in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(col)
    plt.tight_layout()
plt.show()
plt.suptitle("Box plots of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


print("\nOutlier Analysis")
for col in df.columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_band = q1 - 1.5*iqr
    upper_band = q3 + 1.5*iqr
    outlier = df[(df[col] < lower_band) | (df[col] > upper_band)]
    print(f"{col}: {len(outlier)} outliers")