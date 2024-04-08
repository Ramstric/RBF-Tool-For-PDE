# Ramon Everardo Hernandez Hernandez | 2 february 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import numpy as np

import pandas as pd

from scipy.interpolate import Rbf

import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.use('Qt5Agg')


# Gaussian Kernel
def kernel(x, b):
    arg = pow((4/b) * x, 2)
    return np.exp(-arg)


# Inverse quad Kernel
def kernel_2(x, b):
    arg = pow((16/b) * x, 2)
    return 1/(1+arg)


def kernel_3(r, e):
    arg = pow((16 / e) * r, 2)
    return np.sqrt(1 + arg)


# Datos originales o muestra
# x_0 = np.linspace(-10, 10, num=20).round(4)
# y_0 = np.sin(0.2*pow(x_0,2)).round(4)
x_0 = np.array([0, 0.054, 0.259, 0.350, 0.482, 0.624, 0.679, 0.770, 1.037, 1.333, 1.505,
            1.688, 1.933, 2.283])

y_0 = np.array([0, 0.633, 3.954, 3.697, 1.755, 0.679, 0.422, 0.375, 2.574, 5.428, 5.428,
            4.141, -0.326, -2.220])

# Variables
size = x_0.size
b = 6                   # El valor de la base debe de elegirse de acuerdo al error de la interpolación

# Inicio del metodo para RBF
mainMatrix = np.empty(shape=(size, size))

for row in range(0, size):
    for col in range(0, size):
        mainMatrix[row, col] = kernel_3(x_0[row] - x_0[col], b)


mainMatrix = np.linalg.inv(mainMatrix)              # Matriz inversa

yMatrix = np.vstack(y_0)                            # Matriz de Y
fMatrix = np.empty(shape=size, dtype=np.single)     # Matriz evaluada


# Funcion interpolada
def RBF(x):
    for col in range(size):
        fMatrix[col] = kernel_3(x - x_0[col], b)

    temp = fMatrix @ mainMatrix
    temp = temp @ yMatrix

    return temp


# Datos interpolados
x_1 = np.linspace(0, 3.683, num=100)
y_RBF = np.empty(x_1.size)

for idx in range(x_1.size):
    y_RBF[idx] = RBF(x_1[idx])


# Interpolacion RBF de SciPy
interp_Func = Rbf(x_0, y_0)
y_interp = interp_Func(x_1)


Table_1 = pd.DataFrame(data=[x_0, y_0], index=['X', 'Y']).T
Table_2 = pd.DataFrame(data=[x_1, y_RBF], index=['X', 'Y']).T

plt.plot(x_1, y_RBF, linewidth=4, color="dodgerblue")
plt.plot(x_1, y_interp, linewidth=2.5, color="sandybrown", ls=(0, (1, 0.1)))
plt.scatter(x_0, y_0, s=5, color="crimson", zorder=2)

ax = plt.gca()
ax.set_facecolor('lavender')

plt.grid(c="white")


plt.show()
