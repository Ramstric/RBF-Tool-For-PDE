# Ramon Everardo Hernandez Hernandez | 6 april 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import matplotx

custom_colors = {
    "Cyan": "#56B6C2",
    "Blue": "#61AFEF",
    "Purple": "#C678DD",
    "Green": "#98C379",
    "Rose": "#E06C75",
    "Orange": "#D19A66",
    "Red": "#BE5046",
    "Gold": "#E5C07B",

    "RoyaBlue": "#528BFF",

    "Gray": "#ABB2BF",
    "LightGray": "#CCCCCC",

    "LightBlack": "#282C34",
    "Black": "#1D2025"
}

# mpl.use('Qt5Agg')
plt.style.use(matplotx.styles.onedark)


# Gaussian Kernel
def gaussian_kernel(x: float, radius: float):
    return np.e**(-(4*x/radius)**2)


# Inverse quad Kernel
def invQuad_kernel(x: float, radius: float):
    return 1/(1+(16*x/radius)**2)


def multiQuad_kernel(x: float, radius: float):
    return np.sqrt(1 + (16*x/radius)**2)


# Datos muestreados
x_0 = np.array([0, 0.054, 0.259, 0.350, 0.482, 0.624, 0.679, 0.770, 1.037, 1.333, 1.505, 1.688, 1.933, 2.283])

y_0 = np.array([0, 0.633, 3.954, 3.697, 1.755, 0.679, 0.422, 0.375, 2.574, 5.428, 5.428, 4.141, -0.326, -2.220])

datos_muestreados = pd.DataFrame(data=[x_0, y_0], index=['X', 'Y']).T

if x_0.size != y_0.size:
    raise Exception("Los datos muestreados deben de tener la misma longitud")

# Variables
size = x_0.size
r = 2.5                   # El valor del radio debe de elegirse de acuerdo al error de la interpolación

# Inicio del método para la interpolación con RBF
phi_matrix = np.array([multiQuad_kernel(datos_muestreados.X - rbf_pos, r) for rbf_pos in datos_muestreados.X])

f_matrix = np.vstack(datos_muestreados.Y)

weights_matrix = np.linalg.solve(phi_matrix, f_matrix)


# Funcion interpolada
def interpolate_RBF(x: float, radius: float):
    temp = 0
    for (rbf_pos, w) in zip(datos_muestreados.X, weights_matrix):
        temp += w * multiQuad_kernel(x - rbf_pos, radius)
    return temp


# Datos interpolados
x_1 = np.linspace(0, 3.683, num=100)
y_RBF = [interpolate_RBF(x, r) for x in x_1]


# Interpolacion RBF de SciPy
interp_Func = Rbf(x_0, y_0)
y_interp = interp_Func(x_1)


tabla_muestras = pd.DataFrame(data=[x_0, y_0], index=['X', 'Y']).T
tabla_interpolacion = pd.DataFrame(data=[x_1, y_RBF], index=['X', 'Y']).T

plt.plot(x_1, y_RBF)
plt.scatter(x_0, y_0, s=8, zorder=2, color=custom_colors["LightGray"])
plt.grid()
plt.show()