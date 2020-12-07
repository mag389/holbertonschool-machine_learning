#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.randn(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# my code below

plt.xlabel("x coordinate (m)")
plt.ylabel("y coordinate (m)")
plt.title("Mountain Elevation")
plt.scatter(x, y, c=z)
cbar = plt.colorbar()
cbar.ax.set_ylabel("elevation (m)")

plt.show()
