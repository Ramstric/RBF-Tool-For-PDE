# Ramon Everardo Hernandez Hernandez | 6 april 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import torch
from math import e

import matplotlib.pyplot as plt
import matplotx
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

#mpl.use('Qt5Agg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 200
custom_colors = {
    "Blue": "#61AFEF",
    "Orange": "#D49F6E",
    "Green": "#98C379",
    "Rose": "#E06C75",
    "Purple": "#C678DD",
    "Gold": "#E5C07B",
    "Cyan": "#36AABA",

    0: "#61AFEF",
    1: "#D49F6E",
    2: "#98C379",
    3: "#E06C75",
    4: "#C678DD",
    5: "#E5C07B",
    6: "#36AABA",

    "LightCyan": "#56B6C2",


    "AltOrange": "#D19A66",
    "Red": "#BE5046",


    "RoyaBlue": "#528BFF",

    "Gray": "#ABB2BF",
    "LightGray": "#CCCCCC",

    "LightBlack": "#282C34",
    "Black": "#1D2025"
}
plt.style.use(matplotx.styles.onedark)

cmap = LinearSegmentedColormap.from_list('mycmap', [custom_colors[i] for i in range(2)])
cmap_2 = ListedColormap([custom_colors[i] for i in range(5)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian(x: torch.tensor, radius: float):
    return e ** (-(4 * x / radius) ** 2)


def multiQuad_kernel(x: torch.tensor, radius: float):
    return torch.sqrt(1 + (16*x/radius)**2)


# Datos muestreados
x_0 = torch.linspace(-10, 10, 50, device=device)
y_0 = torch.linspace(-10, 10, 50, device=device)

#x_0, y_0 = np.meshgrid(x_0, y_0)
x_0, y_0 = torch.meshgrid(x_0, y_0, indexing='xy')
#z = torch.cosh(x_0 - 0.5) * torch.cosh(y_0 - 0.5)
#z = torch.sin(torch.sqrt(x_0**2 + y_0**2))
#z = 1 - abs(x_0+y_0)-abs(y_0-x_0)
#z = torch.sin(10*(x_0**2 + y_0**2))/10
t = 8
z = torch.sin(torch.sqrt(x_0**2 + y_0**2+t**2))/torch.sqrt(x_0**2 + y_0**2+t**2)

pairs = torch.stack([x_0.ravel(), y_0.ravel()]).T

# Variables
size = x_0.size(dim=0)
r = 1

# Inicio del método para la interpolación con RB
phi_matrix = torch.stack([gaussian(torch.linalg.vector_norm(pairs - pos, dim=1), r) for pos in pairs])
weights_matrix = torch.linalg.solve(phi_matrix, z.ravel())


# Función interpolada
def interpolate(x: torch.tensor, radius: float) -> torch.tensor:
    temp = torch.stack([torch.linalg.vector_norm(torch.sub(x, pos), dim=1) for pos in pairs])
    temp = gaussian(temp, radius)
    return torch.matmul(temp.T, weights_matrix)


# Interpolación
step = 100

x_RBF = torch.linspace(-10, 10, step, device=device)
y_RBF = torch.linspace(-10, 10, step, device=device)
x_RBF, y_RBF = torch.meshgrid(x_RBF, y_RBF, indexing='xy')

pairs_2 = torch.stack([x_RBF.ravel(), y_RBF.ravel()]).T

z_RBF = interpolate(pairs_2, r)
z_RBF = torch.reshape(z_RBF, x_RBF.shape)


# Plotting
ax = plt.axes(projection="3d")
surf = ax.plot_surface(x_RBF.cpu().detach().numpy(), y_RBF.cpu().detach().numpy(), z_RBF.cpu().detach().numpy(),
                       cmap=cm.bone, antialiased=True, alpha=0.8, zorder=1)
scat = ax.scatter3D(x_0.cpu().detach().numpy(), y_0.cpu().detach().numpy(), z.cpu().detach().numpy(),
                    c=z.cpu().detach().numpy(), cmap=cmap_2)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

#ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))

plt.show()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
