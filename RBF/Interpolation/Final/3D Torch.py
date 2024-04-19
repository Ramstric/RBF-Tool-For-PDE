import torch
import RBF_Interpolator

import matplotlib.pyplot as plt
import matplotx
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------[ Plot style ]--------------------------------
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

# --------------------------------[ Datos muestreados ]--------------------------------
x_0 = torch.linspace(-10, 10, 20, device=device)
y_0 = torch.linspace(-10, 10, 20, device=device)
x_0, y_0 = torch.meshgrid(x_0, y_0, indexing='xy')

#z = 1 - abs(x_0+y_0)-abs(y_0-x_0)
t = 7
z = torch.sin(torch.sqrt(x_0**2 + y_0**2+t**2))/torch.sqrt(x_0**2 + y_0**2+t**2)

radius = 1
interpolator = RBF_Interpolator.RBFInterpolator3D("multiQuad", x_0, y_0, f=z, r=radius)

# --------------------------------[ Interpolación ]--------------------------------
step = 30

x_RBF = torch.linspace(-10, 10, step, device=device)
y_RBF = torch.linspace(-10, 10, step, device=device)
x_RBF, y_RBF = torch.meshgrid(x_RBF, y_RBF, indexing='xy')

z_RBF = interpolator.interpolate(x_RBF, y_RBF)

z_RBF = z_RBF+0.001

# --------------------------------[ Plotting ]--------------------------------
fig = plt.figure()
ax = plt.axes(projection="3d")
scat = ax.scatter3D(x_0.cpu().detach().numpy(), y_0.cpu().detach().numpy(), z.cpu().detach().numpy(),
                    c=z.cpu().detach().numpy(), cmap=cmap_2, zorder=-1)
surf = ax.plot_surface(x_RBF.cpu().detach().numpy(), y_RBF.cpu().detach().numpy(), z_RBF.cpu().detach().numpy(),
                       cmap=cm.bone, antialiased=True, alpha=0.6)


ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

plt.show()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

torch.cuda.empty_cache()
