# Ramon Everardo Hernandez Hernandez | 6 april 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import torch
import matplotlib.pyplot as plt
import matplotx
from math import e
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Gaussian Kernel
def gaussian_kernel(x: float, radius: float):
    return e**(-(4*x/radius)**2)


# Inverse quad Kernel
def invQuad_kernel(x: float, radius: float):
    return 1/(1+(16*x/radius)**2)


def multiQuad_kernel(x: torch.tensor, radius: float):
    return torch.sqrt(1 + (16*x/radius)**2)


# Datos muestreados
x_0 = torch.tensor([0, 0.054, 0.259, 0.350, 0.482, 0.624, 0.679, 0.770, 1.037, 1.333, 1.505, 1.688, 1.933, 2.283],
                   device=device)

f_0 = torch.tensor([0, 0.633, 3.954, 3.697, 1.755, 0.679, 0.422, 0.375, 2.574, 5.428, 5.428, 4.141, -0.326, -2.220],
                   device=device)

if x_0.size(dim=0) != f_0.size(dim=0):
    raise Exception("Los datos muestreados deben de tener la misma longitud")

# Variables
size = x_0.size(dim=0)
r = 2.5                   # El valor del radio debe de elegirse de acuerdo al error de la interpolación

# Inicio del método para la interpolación con RBF
phi_matrix = torch.stack([multiQuad_kernel(x_0 - rbf_pos, r) for rbf_pos in x_0])

weights_matrix = torch.linalg.solve(phi_matrix, f_0)


# Función interpolada
def interpolate_RBF(x: float, radius: float) -> torch.tensor:
    temp = 0
    for (rbf_pos, w) in zip(x_0, weights_matrix):
        temp += w * multiQuad_kernel(x - rbf_pos, radius)
    return temp


# Datos interpolados
x_1 = torch.linspace(0, 3.683, 200, device=device).cpu().detach().numpy()
f_RBF = [interpolate_RBF(x, r).cpu().detach().numpy() for x in x_1]


plt.plot(x_1, f_RBF)
plt.scatter(x_0.cpu().detach().numpy(), f_0.cpu().detach().numpy(), s=8, zorder=2, color=custom_colors["LightGray"])
plt.grid()
plt.show()
