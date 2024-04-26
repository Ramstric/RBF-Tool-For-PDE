import torch
import matplotlib.pyplot as plt
import matplotx
import matplotlib.animation as animation
import RBF
from matplotlib import cm, colors

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
def pairs3D(x_tensor: torch.tensor, y_tensor: torch.tensor):
    # Slicing out first and last element, to get inner points
    x_inner = x_tensor[1:x_tensor.size(0) - 1]
    y_inner = y_tensor[1:y_tensor.size(0) - 1]

    # Creating meshgrid for all points and inner points
    x_temp, y_temp = torch.meshgrid(x_tensor, y_tensor, indexing='xy')
    x_inner, y_inner = torch.meshgrid(x_inner, y_inner, indexing='xy')

    pairs_outer = torch.stack([x_temp.ravel(), y_temp.ravel()]).T  # Copying all points, to remove inner points from it
    pairs_inner = torch.stack([x_inner.ravel(), y_inner.ravel()]).T  # Inner points

    # This returns the index of inners points in all points, and slices them out
    for inner_pair in pairs_inner:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)

    temp = torch.stack([pair for pair in pairs_outer if pair[1].item() == 0])  # These are boundary for t = 0

    # Removing temp from outer points
    for inner_pair in temp:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)

    # These are boundary for x = 0 & x = 1
    temp_2 = torch.stack([pair for pair in pairs_outer if pair[0].item() == 0 or pair[0].item() == 1])

    # Removing temp_2 from outer points
    for inner_pair in temp_2:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)

    pairs_inner = torch.cat((pairs_inner, pairs_outer), 0)  # Added non boundary points back into inner
    pairs_outer = torch.cat((temp, temp_2), 0)  # Replaced with ACTUAL boundary points

    # Outer points in vector form
    x_outer = torch.stack([pos[0] for pos in pairs_outer])
    y_outer = torch.stack([pos[1] for pos in pairs_outer])
    z_outer = x_outer - x_outer ** 2  # <--- (Boundary condition)

    # Inner points in vector form
    x_in = torch.stack([pos[0] for pos in pairs_inner])
    y_in = torch.stack([pos[1] for pos in pairs_inner])
    z_in = torch.zeros(x_in.size(0), device=device)  # <--- (Function of PDE)

    # All points in vector form
    x_all = torch.cat((x_outer, x_in), 0)
    y_all = torch.cat((y_outer, y_in), 0)
    z_all = torch.cat((z_outer, z_in), 0)

    pairs_all = torch.stack([x_all.ravel(), y_all.ravel()]).T  # To obtain them pairs in order

    return pairs_outer, pairs_inner, pairs_all, z_all


# All points
y = torch.arange(0, 1, 0.1, device=device)
x = torch.linspace(0, 1, y.size(0), device=device)

size = x.size(0)

boundary, inner, pairs, z = pairs3D(x, y)

interpolator = RBF.InterpolatorPDE("gaussian", boundary=boundary, inner=inner, all=pairs, f=z, r=5, ode="f_y - f_xx")

# --------------------------------[ Interpolación ]--------------------------------

step = 100

x_RBF = torch.linspace(0, 1, step, device=device)
y_RBF = torch.linspace(0, 1, step, device=device)
x_RBF, y_RBF = torch.meshgrid(x_RBF, y_RBF, indexing='xy')

z_RBF = interpolator.interpolate(x_RBF, y_RBF)

#z_RBF = z_RBF+0.001

# --------------------------------[ Plotting ]--------------------------------
fig = plt.figure()
ax = plt.axes(projection="3d")
surf = ax.plot_surface(x_RBF.cpu().detach().numpy(), y_RBF.cpu().detach().numpy(), z_RBF.cpu().detach().numpy(),
                       cmap=cm.inferno, antialiased=True, alpha=0.6)


ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

#ax.view_init(azim=20)

plt.show()

# --------------------------------[ Plotting 2D ]--------------------------------

"""
step = 200

x_RBF = torch.linspace(0, 1, step, device=device)
y_RBF = torch.full((step,), 0., device=device)
z_RBF = interpolator.interpolate(x_RBF, y_RBF)


fig, ax = plt.subplots(1, 1)
plt.grid()
#plot = plt.plot(x_RBF.cpu().detach().numpy(), z_RBF.cpu().detach().numpy())
plt.scatter(x=x_RBF.cpu().detach().numpy(), y=z_RBF.cpu().detach().numpy(), c=z_RBF.cpu().detach().numpy(),
            cmap="inferno", s=5)

plt.colorbar(label="°C", ax=ax)



x_RBF = torch.linspace(0, 1, step, device=device)
y_RBF = torch.full((step,), 0.8, device=device)
z_RBF = interpolator.interpolate(x_RBF, y_RBF)

plot = plt.plot(x_RBF.cpu().detach().numpy(), z_RBF.cpu().detach().numpy())

x_RBF = torch.linspace(0, 1, step, device=device)
y_RBF = torch.full((step,), 0.9, device=device)
z_RBF = interpolator.interpolate(x_RBF, y_RBF)

plot = plt.plot(x_RBF.cpu().detach().numpy(), z_RBF.cpu().detach().numpy())

plt.ylim(0, 0.3)

plt.xticks([0, 0.5, 1])

fig.set_figwidth(4)

plt.show()
"""

# --------------------------------[ Animating ]--------------------------------

fig, ax = plt.subplots(1, 1)

step = 200
x_RBF = torch.linspace(0, 1, step, device=device)

fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=0.25), cmap=cm.inferno), ax=ax)


def animate(i):
    y_RBF = torch.full((step,), i / 100, device=device)
    z_RBF = interpolator.interpolate(x_RBF, y_RBF)

    ax.clear()

    # Plotting
    ax.grid()

    #ax.plot(x_RBF.cpu().detach().numpy(), z_RBF.cpu().detach().numpy())
    plot = plt.scatter(x=x_RBF.cpu().detach().numpy(), y=z_RBF.cpu().detach().numpy(), c=z_RBF.cpu().detach().numpy(),
                       cmap="inferno", s=2)

    plot.set_clim(0, 0.25)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)


ani = animation.FuncAnimation(fig, animate, 100, interval=500)

writer_video = animation.PillowWriter(fps=60)
ani.save('./animation_drawing.gif', writer=writer_video)

print("Animation saved!")

if device.type == 'cuda':
    print("\n")
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

torch.cuda.empty_cache()
