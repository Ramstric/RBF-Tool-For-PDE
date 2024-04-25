import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotx
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import RBF

#mpl.use('Qt5Agg')

torch.set_printoptions(precision=9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams['figure.dpi'] = 120
custom_colors = {"Blue": "#61AFEF", "Orange": "#D49F6E", "Green": "#98C379", "Rose": "#E06C75",
                 "Purple": "#C678DD", "Gold": "#E5C07B", "Cyan": "#36AABA", 0: "#61AFEF", 1: "#D49F6E",
                 2: "#98C379", 3: "#E06C75", 4: "#C678DD", 5: "#E5C07B", 6: "#36AABA", "LightCyan": "#56B6C2",
                 "AltOrange": "#D19A66", "Red": "#BE5046", "RoyaBlue": "#528BFF", "Gray": "#ABB2BF",
                 "LightGray": "#CCCCCC", "LightBlack": "#282C34", "Black": "#1D2025"}
plt.style.use(matplotx.styles.onedark)


# --------------------------------[ Datos muestreados ]--------------------------------
def pairs3D(x: torch.tensor, y: torch.tensor, z: torch.tensor):
    # Slicing out first and last element, to get inner points
    x_inner = x[1:x.size(0) - 1]
    y_inner = y[1:y.size(0) - 1]
    z_inner = z[1:z.size(0) - 1]

    # Creating meshgrid for all points and inner points
    x, y, z = torch.meshgrid(x, y, z, indexing='xy')
    x_inner, y_inner, z_inner = torch.meshgrid(x_inner, y_inner, z_inner, indexing='xy')

    pairs_outer = torch.stack([x.ravel(), y.ravel(), z.ravel()]).T  # Copying all points, to remove inner points from it
    pairs_inner = torch.stack([x_inner.ravel(), y_inner.ravel(), z_inner.ravel()]).T  # Inner points

    # This returns the index of inners points in all points, and slices them out
    for inner_pair in pairs_inner:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)

    # Outer points in vector form
    x_outer = torch.stack([pos[0] for pos in pairs_outer])
    y_outer = torch.stack([pos[1] for pos in pairs_outer])
    z_outer = torch.stack([pos[2] for pos in pairs_outer])
    f_outer = x_outer * y_outer * z_outer  # <--- (Function of Boundary)

    # Inner points in vector form
    x_in = torch.stack([pos[0] for pos in pairs_inner])
    y_in = torch.stack([pos[1] for pos in pairs_inner])
    z_in = torch.stack([pos[2] for pos in pairs_inner])
    f_in = torch.zeros(x_in.size(0), device=device)    # <--- (Function of PDE)

    # All points in vector form
    x_all = torch.cat((x_outer, x_in), 0)
    y_all = torch.cat((y_outer, y_in), 0)
    z_all = torch.cat((z_outer, z_in), 0)
    f_all = torch.cat((f_outer, f_in), 0)

    pairs_all = torch.stack([x_all.ravel(), y_all.ravel(), z_all.ravel()]).T  # To obtain them pairs in order

    return pairs_outer, pairs_inner, pairs_all, f_all


# All points
x = torch.arange(0, 1.2, 0.2, device=device)
y = torch.arange(0, 1.2, 0.2, device=device)
z = torch.arange(0, 1.2, 0.2, device=device)
size = x.size(0)

boundary, inner, pairs, f = pairs3D(x, y, z)

interpolator = RBF.InterpolatorPDE("multiQuad", boundary=boundary, inner=inner, all=pairs, f=f, r=1, ode="f_xx + f_yy")

# --------------------------------[ Interpolación ]--------------------------------
step = 200

x_RBF = torch.linspace(0, 1, step, device=device)
y_RBF = torch.linspace(0, 1, step, device=device)
x_RBF, y_RBF = torch.meshgrid(x_RBF, y_RBF, indexing='xy')

z_RBF = interpolator.interpolate(x_RBF, y_RBF)

#z_RBF = z_RBF+0.001

# --------------------------------[ Plotting ]--------------------------------
fig = plt.figure()
ax = plt.axes(projection="3d")
surf = ax.plot_surface(x_RBF.cpu().detach().numpy(), y_RBF.cpu().detach().numpy(), z_RBF.cpu().detach().numpy(),
                       cmap=cm.bone, antialiased=True, alpha=0.6)


ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

#ax.view_init(azim=20)

plt.show()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

torch.cuda.empty_cache()

