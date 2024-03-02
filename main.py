import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#mpl.use('Qt5Agg')
plt.rcParams['figure.figsize'] = (10, 10)


#  MQuad Kernel
def kernel(S, epsilon):
    arg = pow((epsilon) * np.linalg.norm(S), 2)
    return np.sqrt(arg + 1)


# Derivative MQuad Kernel
def D_kernel(S, epsilon):
    arg_bottom = pow((epsilon) * np.linalg.norm(S), 2) + 1
    arg_superior = pow(epsilon, 4)*(pow(S[0], 2) + pow(S[1], 2)) + 2 * pow(epsilon, 2)
    return arg_superior / pow(arg_bottom, 3/2)


"""
def D_kernel(S, epsilon):
    arg_bottom = pow((epsilon) * np.linalg.norm(S), 2) + 1
    arg_superior = pow(epsilon, 2) * (S[0] + S[1])
    return arg_superior / np.sqrt(arg_bottom)
"""

# Must change NODES, RBD Points, Z matrix, Boundary cond both in value and if's

# Interior nodes
x_I = np.arange(-2.75, 3, 0.25)
# Boundary nodes
x_B = np.array([-3, 3])

X = np.block([x_B, x_I])   # Making the Vector X (Boundary + Interior nodes)

# Interior nodes
y_I = np.arange(-2.75, 3, 0.25)
# Boundary nodes
y_B = np.array([-3, 3])

Y = np.block([y_B, y_I])

S = np.zeros(shape=(X.size * Y.size, 2))

X, Y = np.meshgrid(X, Y)
Z = (- 4*pow(Y, 2) - 4*pow(X, 2)) * np.sin(pow(X, 2) + pow(Y, 2)) + 4 * np.cos(pow(X, 2) + pow(Y, 2))
#Z = np.full(shape=(9, 9), fill_value=6, dtype=np.double)
#Z = np.ones(shape=(5, 9))


# Boundary conditions
for row in range(y_I.size + y_B.size):
    for col in range(x_I.size + x_B.size):
        if X[row, col] == -3:
            Z[row, col] = np.sin(9 + pow(Y[row, col], 2))
        elif X[row, col] == 3:
            Z[row, col] = np.sin(9 + pow(Y[row, col], 2))

        elif Y[row, col] == -3:
            Z[row, col] = np.sin(9 + pow(X[row, col], 2))
        elif Y[row, col] == 3:
            Z[row, col] = np.sin(9 + pow(X[row, col], 2))

zMatrix = np.stack(Z.ravel())

positions = np.stack([X.ravel(), Y.ravel()])

for idx in range(Z.size):
    S[idx, 0] = positions[0, idx]
    S[idx, 1] = positions[1, idx]

# Make allocation for RBF.
X_RBF = np.arange(-3, 3, 0.1)
Y_RBF = np.arange(-3, 3, 0.1)

S_RBF = np.zeros(shape=(X_RBF.size * Y_RBF.size, 2))
Z_RBF = np.zeros(shape=X_RBF.size * Y_RBF.size)

X_RBF, Y_RBF = np.meshgrid(X_RBF, Y_RBF)

positions = np.stack([X_RBF.ravel(), Y_RBF.ravel()])
for idx in range(Z_RBF.size):
    S_RBF[idx, 0] = positions[0, idx]
    S_RBF[idx, 1] = positions[1, idx]


# Variables
size = Z.size
b = 2                # El valor de la base debe de elegirse de acuerdo al error de la interpolación

# Inicio del metodo para RBF
mainMatrix = np.zeros(shape=(size, size))

for row in range(size):
    for col in range(size):

        if S[row, 0] == -3 or S[row, 0] == 3:
            mainMatrix[row, col] = kernel(S[row] - S[col], b)

        elif S[row, 1] == -3 or S[row, 1] == 3:
            mainMatrix[row, col] = kernel(S[row] - S[col], b)

        else:
            mainMatrix[row, col] = D_kernel(S[row] - S[col], b)  # Interior points


mainMatrix = np.linalg.inv(mainMatrix)              # Matriz inversa

zMatrix = np.vstack(zMatrix)                            # Matriz de Z

fMatrix = np.empty(shape=size, dtype=np.single)     # Matriz evaluada


# Funcion interpolada
def RBF(s):
    for col in range(size):
        fMatrix[col] = kernel(s - S[col], b)

    temp = fMatrix @ mainMatrix

    temp = temp @ zMatrix

    return temp


# Datos interpolados
#x_1 = np.linspace(0, 3.683, num=100)
for idx in range(S_RBF.shape[0]):
    Z_RBF[idx] = RBF(S_RBF[idx])

Z_RBF = np.array_split(Z_RBF, X_RBF.shape[0])

Z_RBF = np.stack(Z_RBF, axis=0)


# Plot the surface
ax = plt.axes(projection ="3d")

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.25, 1.25, 0.1, 1]))

ax.set_facecolor("slategrey")

surf = ax.plot_surface(X_RBF, Y_RBF, Z_RBF, cmap=cm.Blues_r, linewidth=0, antialiased=False, alpha=1, zorder=1)
#scat = ax.scatter3D(X, Y, Z, s=25, zorder=4, c=Z, cmap=cm.plasma_r)
#scat = ax.scatter3D(X_RBF, Y_RBF, Z_RBF, s=5, c="red")

plt.show()
