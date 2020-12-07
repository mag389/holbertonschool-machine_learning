#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
fig, axs = plt.subplots(3, 2)
fig.suptitle("All in One")
# zero
axs[0, 0].plot(np.arange(0, 11), y0, c='red')
axs[0, 0].set_xlim(0, 10)

# one
axs[0, 1].scatter(x1, y1, c='magenta')
axs[0, 1].set_xlabel("Height (in)", fontsize=8)
axs[0, 1].set_ylabel("Weight (lbs)", fontsize=8)
axs[0, 1].set_title("Men's Height vs Weight", fontsize=8)

# two
axs[1, 0].set_xlabel("Time (years)", fontsize=8)
axs[1, 0].set_ylabel("Fraction Remaining", fontsize=8)
axs[1, 0].set_title("Exponential Decay of C-14", fontsize=8)
axs[1, 0].plot(x2, y2)
axs[1, 0].set_xlim(0, 28650)
axs[1, 0].set_yscale("log")

# three
axs[1, 1].set_xlabel("Time (years)", fontsize=8)
axs[1, 1].set_ylabel("Fraction Remaining", fontsize=8)
axs[1, 1].set_title("Exponential Decay of Radioactive Elements", fontsize=8)
axs[1, 1].set_xlim(0, 20000)
axs[1, 1].set_ylim(0, 1)
axs[1, 1].plot(x3, y31, c='red', linestyle='dashed')
axs[1, 1].plot(x3, y32, c='green')
axs[1, 1].legend(["C-14", "Ra-226"], fontsize=8)

# four
ax = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=2)
# alternate using gridspec
# gs = fig.add_gridspec(3, 2)
# ax = fig.add_subplot(gs[2, :])
ax.set_xlabel("Grades", fontsize=8)
ax.set_ylabel("Number of Students", fontsize=8)
ax.set_title("Project A", fontsize=8)
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 10))
ax.set_ylim(0, 30)
ax.hist(student_grades, range=(0, 100), bins=10, edgecolor='black')


plt.tight_layout()
plt.show()
