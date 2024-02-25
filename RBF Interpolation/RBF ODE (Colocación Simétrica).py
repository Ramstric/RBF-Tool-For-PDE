# Ramon Everardo Hernandez Hernandez | 2 february 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# mpl.use('Qt5Agg')


# MQuad Kernel
def kernel(r, e):
    arg = pow(e * r, 2)
    return np.exp(-arg)


# Derivated MQuad Kernel
def D_kernel(r, e):
    arg_superior = pow(e*r, 2)
    return -2 * pow(e, 2) * np.exp(-arg_superior) * r


def DD_kernel(r, e):
    arg_superior = pow(e*r, 2)
    return -2*pow(e, 2)*(-2 * pow(e, 2) * np.exp(-arg_superior) * pow(r, 2) + np.exp(-arg_superior))


def f(x):
    arg = pow(x + 4/3, 2)
    return arg


# Datos originales o muestra
# x_0 = np.linspace(-10, 10, num=20).round(4)
# y_0 = np.sin(0.2*pow(x_0,2)).round(4)
x_I = np.array([-2, -1.9, -1.75, -1.5, -1.4, -1.25, -1, -0.8, -0.75, -0.5, -0.4, -0.25, 0, 0.1,  0.25])
x_B = np.array([-2.12, 1.00])

x_0 = np.block([x_B, x_I])

y_f = np.empty(x_I.size)

for index in range(0, x_I.size):
    y_f[index] = f(x_I[index])

y_g = np.array([0.05, 4.44])

y_0 = np.block([y_g, y_f])


# Variables
size = x_0.size

b = 1.3               # El valor de la base debe de elegirse de acuerdo al error de la interpolación

# Inicio del metodo para RBF
Block_1, Block_2, Block_3, Block_4 = np.zeros(shape=(x_B.size, x_B.size)), np.zeros(shape=(x_B.size, x_I.size)), np.zeros(shape=(x_I.size, x_B.size)), np.zeros(shape=(x_I.size, x_I.size))

for row in range(0, x_B.size):
    for col in range(0, x_B.size):
        Block_1[row, col] = kernel(x_0[row] - x_0[col], b)

for row in range(0, x_B.size):
    for col in range(0, x_I.size):
        Block_2[row, col] = D_kernel(x_0[row] - x_0[col+x_B.size], b)


for row in range(0, x_I.size):
    for col in range(0, x_B.size):
        Block_3[row, col] = D_kernel(x_0[row+x_B.size] - x_0[col], b)

for row in range(0, x_I.size):
    for col in range(0, x_I.size):
        Block_4[row, col] = DD_kernel(x_0[row+x_B.size] - x_0[col+x_B.size], b)

mainMatrix = np.block([[Block_1, Block_2], [Block_3, Block_4]])

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
x_1 = np.linspace(-2, 1, num=30)
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
