import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../MLdatasets/Iris.csv")
X = data[data.columns[1:-1]]

scaled_X = StandardScaler().fit_transform(X)
reduced_X = PCA(n_components=2).fit_transform(scaled_X)

plt.figure(figsize=(7, 7))
sns.scatterplot(
    x=np.ndarray.ravel(reduced_X[:, :1]),
    y=np.ndarray.ravel(reduced_X[:, 1:]),
    hue=data['Species']
)
plt.title("Scatter Plot after reducing the dimension of feature vectors using PCA")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()
