#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# my code here
# fruit is [farrah, fred, Felicia] by [apples, bananas, oranges, peaches]
labels = ['Farrah', 'Fred', 'Felicia']
fig, ax = plt.subplots()
width = 0.5

ax.set_ylabel("Quantity of Fruit")
ax.set_ylim(0, 80)
ax.set_yticks(range(0, 81, 10))

bot = [0, 0, 0, 0]
bot[1] = fruit[0]
bot[2] = fruit[0] + fruit[1]
bot[3] = bot[2] + fruit[2]

ax.bar(labels, fruit[0], width, label="apples", color="red")
ax.bar(labels, fruit[1], width, label="bananas", color="yellow", bottom=bot[1])
ax.bar(labels, fruit[2], width, label="oranges", color="orange", bottom=bot[2])
ax.bar(labels,
       fruit[3], width,
       label="peaches", color="#ffe5b4", bottom=bot[3])

ax.legend()
plt.title("Number of Fruit per Person")
plt.show()
