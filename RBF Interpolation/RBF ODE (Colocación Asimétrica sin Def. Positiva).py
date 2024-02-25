# Ramon Everardo Hernandez Hernandez | 2 february 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# mpl.use('Qt5Agg')


# MQuad Kernel
def kernel(r, e):
    arg = pow((16 / e) * r, 2)
    return np.sqrt(1 + arg)


# Derivated MQuad Kernel
def D_kernel(r, e):
    arg_bottom = pow((16/e) * r, 2)
    arg_superior = pow(16/e, 2) * r
    return arg_superior / np.sqrt(1 + arg_bottom)


def f(x):
    arg = 4*np.sin(3*x)
    return arg


# Datos originales o muestra
# x_0 = np.linspace(-10, 10, num=20).round(4)
# y_0 = np.sin(0.2*pow(x_0,2)).round(4)
x_I = np.array([-5, -4.5, -4, -3.75, -3.5, -3, -2.75, -2.5, -1.5, -1.25, -0.5, -0.25, 0, 0.1, 0.5, 0.85, 1.0, 1.25, 1.5, 1.75])
x_B = np.array([-5.74, 2.12])

x_0 = np.block([x_B, x_I])

y_f = np.empty(x_I.size)

for index in range(0, x_I.size):
    y_f[index] = f(x_I[index])

y_g = np.array([-0.023, 2.23])

y_0 = np.block([y_g, y_f])


# Variables
size = x_0.size

b = 24                   # El valor de la base debe de elegirse de acuerdo al error de la interpolación

# Inicio del metodo para RBF
Block_1, Block_2 = np.empty(shape=(x_B.size, size)), np.empty(shape=(x_I.size, size))

for row in range(0, x_B.size):
    for col in range(0, size):
        Block_1[row, col] = kernel(x_0[row] - x_0[col], b)

for row in range(0, x_I.size):
    for col in range(0, size):
        Block_2[row, col] = 2*D_kernel(x_0[row+x_B.size] - x_0[col], b) - kernel(x_0[row+x_B.size] - x_0[col], b)

mainMatrix = np.block([[Block_1], [Block_2]])

mainMatrix = np.linalg.inv(mainMatrix)              # Matriz inversa

yMatrix = np.vstack(y_0)                            # Matriz de Y
fMatrix = np.empty(shape=size, dtype=np.single)     # Matriz evaluada


# Funcion interpolada
def RBF(x):
    for col in range(size):
        fMatrix[col] = kernel(x - x_0[col], b)

    temp = fMatrix @ mainMatrix
    temp = temp @ yMatrix

    return temp


# Datos interpolados
x_1 = np.linspace(-5, 2, num=70)
y_RBF = np.empty(x_1.size)

for idx in range(x_1.size):
    y_RBF[idx] = RBF(x_1[idx])

Table_1 = pd.DataFrame(data=[x_0, y_0], index=['X', 'Y']).T
Table_2 = pd.DataFrame(data=[x_1, y_RBF], index=['X', 'Y']).T

plt.plot(x_1, y_RBF, linewidth=1, color="dodgerblue")
#plt.scatter(x_1, y_RBF, s=5, color="crimson", zorder=2)
#plt.scatter(x_0, y_0, s=5, color="crimson", zorder=2)



ax = plt.gca()
ax.set_facecolor('lavender')

ax.axhline(y=0, color='dimgray', linewidth=1)
ax.axvline(x=0, color='dimgray', linewidth=1)

plt.grid(c="white")


plt.show()
