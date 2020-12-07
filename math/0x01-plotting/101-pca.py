#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# your code here

# print(data)
# print(labels)
# print(pca_data)
x = pca_data[:, 0]
y = pca_data[:, 1]
z = pca_data[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c=labels, cmap="plasma")

ax.set_title("PCA of Iris Dataset")
ax.set_xlabel("U1")
ax.set_ylabel("U2")
ax.set_zlabel("U3")

plt.show()
