import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
#mpl.use('Qt5Agg')
plt.rcParams['figure.figsize'] = (10, 10)


def kernel(s, radius):
    arg = pow((16 / radius) * np.sqrt(pow(s[0], 2) + pow(s[1], 2)), 2)
    return np.sqrt(arg + 1)


# Make data.
X = np.arange(0, 1+1/5, 1/5)
Y = np.arange(0, 1+1/5, 1/5)

S = np.zeros(shape=(X.size * Y.size, 2))

X, Y = np.meshgrid(X, Y)
#R = -pow(2*X + 3*Y, 2)
# R = -9*( pow(X-0.5,2) ) -9*( pow(Y-0.25,2) )
R = np.cosh(X-0.5) * np.cosh(Y-0.5)
Z = R

zMatrix = np.stack(Z.ravel())

positions = np.stack([X.ravel(), Y.ravel()])
for idx in range(Z.size):
    S[idx, 0] = positions[0, idx]
    S[idx, 1] = positions[1, idx]

# Make allocation for RBF.
X_RBF = np.arange(0, 1+1/35, 1/35)
Y_RBF = np.arange(0, 1+1/35, 1/35)

S_RBF = np.zeros(shape=(X_RBF.size * Y_RBF.size, 2))
Z_RBF = np.zeros(shape=X_RBF.size * Y_RBF.size)

X_RBF, Y_RBF = np.meshgrid(X_RBF, Y_RBF)

positions = np.stack([X_RBF.ravel(), Y_RBF.ravel()])
for idx in range(Z_RBF.size):
    S_RBF[idx, 0] = positions[0, idx]
    S_RBF[idx, 1] = positions[1, idx]


# Variables
size = Z.size
b = 10                  # El valor de la base debe de elegirse de acuerdo al error de la interpolación

# Inicio del método para RBF
mainMatrix = np.empty(shape=(size, size))

for row in range(size):
    for col in range(size):
        mainMatrix[row, col] = kernel(S[row] - S[col], b)


mainMatrix = np.linalg.inv(mainMatrix)              # Matriz inversa

zMatrix = np.vstack(zMatrix)                            # Matriz de Z

fMatrix = np.empty(shape=size, dtype=np.single)     # Matriz evaluada


# Función interpolada
def RBF(s):
    for column in range(size):
        fMatrix[column] = kernel(s - S[column], b)

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
ax = plt.axes(projection="3d")

surf = ax.plot_surface(X_RBF, Y_RBF, Z_RBF, cmap=cm.plasma, linewidth=0, antialiased=False, alpha=0.2, zorder=1)
scat = ax.scatter3D(X, Y, Z, s=25, zorder=4, c=Z, cmap=cm.plasma_r)
#scat = ax.scatter3D(X_RBF, Y_RBF, Z_RBF, s=5, c="red")

plt.show()
