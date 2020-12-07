#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# my code here

plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")
plt.xlim(0, 100)
plt.xticks(range(0,100,10))
plt.ylim(0, 30)
plt.hist(student_grades, range=(0, 100), bins=10, edgecolor='black')
plt.show()
