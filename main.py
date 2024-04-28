# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

Z = np.random.randint(2, size=6)
Z_2 = np.random.randint(2, size=6)


print(Z)
fig, ax0 = plt.subplots()

pcm = ax0.pcolormesh([Z, Z])
plt.colorbar(pcm, ax=ax0)

ax0.set_ylim([-20, 20])

ax0.set_title('matplotlib.axes.Axes.pcolormesh() Examples')
plt.show()