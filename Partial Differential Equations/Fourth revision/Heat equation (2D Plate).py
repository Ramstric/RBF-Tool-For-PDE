import torch
import matplotlib.pyplot as plt
import matplotx
import matplotlib.animation as animation
import PDE_Interpolation
from matplotlib import cm, colors
import matplotlib as mpl

#mpl.use('Qt5Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


plt.rcParams['figure.dpi'] = 400
custom_colors = {"Blue": "#61AFEF", "Orange": "#D49F6E", "Green": "#98C379", "Rose": "#E06C75",
                 "Purple": "#C678DD", "Gold": "#E5C07B", "Cyan": "#36AABA", 0: "#61AFEF", 1: "#D49F6E",
                 2: "#98C379", 3: "#E06C75", 4: "#C678DD", 5: "#E5C07B", 6: "#36AABA", "LightCyan": "#56B6C2",
                 "AltOrange": "#D19A66", "Red": "#BE5046", "RoyaBlue": "#528BFF", "Gray": "#ABB2BF",
                 "LightGray": "#CCCCCC", "LightBlack": "#282C34", "Black": "#1D2025"}
plt.style.use(matplotx.styles.onedark)


# --------------------------------[ Datos muestreados ]--------------------------------
def pairs3D(x_tensor: torch.tensor, y_tensor: torch.tensor, t_tensor: torch.tensor):
    # Slicing out first and last element, to get inner points
    x_inner = x_tensor[1:x_tensor.size(0) - 1]
    y_inner = y_tensor[1:y_tensor.size(0) - 1]
    t_inner = t_tensor[1:t_tensor.size(0) - 1]

    # Creating meshgrid for all points and inner points
    x_temp, y_temp, t_temp = torch.meshgrid(x_tensor, y_tensor, t_tensor, indexing='xy')
    x_inner, y_inner, t_inner = torch.meshgrid(x_inner, y_inner, t_inner, indexing='xy')

    pairs_outer = torch.stack([x_temp.ravel(), y_temp.ravel(), t_temp.ravel()]).T  # Copying all points, to remove inner points from it
    pairs_inner = torch.stack([x_inner.ravel(), y_inner.ravel(), t_inner.ravel()]).T  # Inner points

    # This returns the index of inners points in all points, and slices them out
    for inner_pair in pairs_inner:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)

    temp = torch.stack([pair for pair in pairs_outer if pair[2].item() == 0])  # These are initial values, for t = 0
    # Removing temp from outer points
    for inner_pair in temp:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)

    # These are boundary for x = 0 & x = 1 for any time non 0
    temp_2_x_left = torch.stack([pair for pair in pairs_outer if pair[0].item() == 0])
    temp_2_x_right = torch.stack([pair for pair in pairs_outer if pair[0].item() == 1])
    # Removing temp_2_x from outer points
    for inner_pair in temp_2_x_left:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)
    for inner_pair in temp_2_x_right:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)

    # These are boundary for y = 0 & y = 1 for any time non 0
    temp_2_y_down = torch.stack([pair for pair in pairs_outer if pair[1].item() == 0])
    temp_2_y_up = torch.stack([pair for pair in pairs_outer if pair[1].item() == 1])
    # Removing temp_2_y from outer points
    for inner_pair in temp_2_y_down:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)
    for inner_pair in temp_2_y_up:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)

    z_temp_2_LeftDown = torch.full((temp_2_x_left.size(0) + temp_2_y_down.size(0),), 0.2, device=device)  # Stationary temp (temperature at boundary for t > 0)
    z_temp_2_RightUp = torch.zeros(temp_2_x_right.size(0) + temp_2_y_up.size(0), device=device)  # Stationary temp (temperature at boundary for t > 0)

    temp_2_x = torch.cat((temp_2_x_left, temp_2_y_down), 0)
    temp_2_y = torch.cat((temp_2_x_right, temp_2_y_up), 0)
    z_temp_2 = torch.cat((z_temp_2_LeftDown, z_temp_2_RightUp), 0)

    pairs_inner = torch.cat((pairs_inner, pairs_outer), 0)  # Added non boundary points back into inner

    # Initial values (t = 0)

    initial_horizontal = torch.stack([pair for pair in temp if pair[1].item() == 0])  # These are boundary for y = 0

    for inner_pair in initial_horizontal:
        index = torch.where(torch.all(torch.eq(inner_pair, temp), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        temp = torch.cat((temp[:index[0].item()], temp[index[0].item() + 1:]), 0)

    initial_vertical = torch.stack([pair for pair in temp if pair[0].item() == 0])  # These are boundary for y = 0

    for inner_pair in initial_vertical:
        index = torch.where(torch.all(torch.eq(inner_pair, temp), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        temp = torch.cat((temp[:index[0].item()], temp[index[0].item() + 1:]), 0)

    # Now temp has all initial values but inside the boundary
    # After setting the z values, must concatenate in the same order as pairs_outer (initial_horizontal, initial_vertical, temp)

    # First, initial conditions on the horizontal boundary
    temp_x = torch.stack([pos[0] for pos in initial_horizontal])
    z_temp = temp_x - temp_x**2  # Initial temp

    z_outer = z_temp  # And store them

    # Second, initial conditions on the vertical boundary
    temp_y = torch.stack([pos[1] for pos in initial_vertical])
    z_temp = temp_y - temp_y**2  # Initial temp

    z_outer = torch.cat((z_outer, z_temp), 0)  # And store them

    # Third, initial conditions inside the boundary
    temp_x = torch.stack([pos[0] for pos in temp])
    z_temp = torch.zeros(temp_x.size(0), device=device)  # Initial temp

    z_outer = torch.cat((z_outer, z_temp), 0)  # And store them

    pairs_outer = torch.cat((initial_horizontal, initial_vertical, temp, temp_2_x, temp_2_y), 0)  # Re attaching al boundary and initial in the same order they are evaluated

    # Outer points in vector form
    x_outer = torch.stack([pos[0] for pos in pairs_outer])
    y_outer = torch.stack([pos[1] for pos in pairs_outer])
    t_outer = torch.stack([pos[2] for pos in pairs_outer])
    z_outer = torch.cat((z_outer, z_temp_2), 0)  # <--- (Boundary condition)

    # Inner points in vector form
    x_in = torch.stack([pos[0] for pos in pairs_inner])
    y_in = torch.stack([pos[1] for pos in pairs_inner])
    t_in = torch.stack([pos[2] for pos in pairs_inner])
    z_in = torch.full((x_in.size(0),), 0., device=device)  # <--- (Function of PDE)

    # All points in vector form
    x_all = torch.cat((x_outer, x_in), 0)
    y_all = torch.cat((y_outer, y_in), 0)
    t_all = torch.cat((t_outer, t_in), 0)
    z_all = torch.cat((z_outer, z_in), 0)

    pairs_all = torch.stack([x_all.ravel(), y_all.ravel(), t_all.ravel()]).T  # To obtain them pairs in order

    return pairs_outer, pairs_inner, pairs_all, z_all, x_in, y_in, z_in, x_outer, y_outer, z_outer


# All points
t = torch.arange(0, 1, 0.1, device=device)
x = torch.linspace(0, 1, t.size(0), device=device)
y = torch.linspace(0, 1, t.size(0), device=device)


boundary, inner, pairs, z, x_i, y_i, z_i, x_o, y_o, z_o = pairs3D(x, y, t)

interpolator = PDE_Interpolation.InterpolatorPDE("gaussian", boundary=boundary, inner=inner, all_pairs=pairs, f=z, r=5, pde="f_t - 0.3f_xx - 0.3f_yy")


#Define custom derivative operator for PDE problem
#def PDE(operation: str, s: torch.tensor, radius: float, x: torch.tensor, y: torch.tensor):
#    return RBF.gaussianDerY(s, y, radius) - 0.2*RBF.gaussianDerXX(s, x, radius)


#interpolator.derivative_operator = PDE
#interpolator.recalculate_weights(r=5, pde="f_y - f_xx")

# --------------------------------[ Interpolación ]--------------------------------

step = 50

#x_RBF = torch.linspace(0, 1, step, device=device)
#y_RBF = torch.linspace(0, 1, step, device=device)
#t_RBF = torch.linspace(0, 1, step, device=device)
#t_RBF = torch.tensor([0.01], device=device)
#x_RBF, y_RBF, t_RBF = torch.meshgrid(x_RBF, y_RBF, t_RBF, indexing='xy')

#z_RBF = interpolator.interpolate(x_RBF, y_RBF, t_RBF)

#x_RBF = torch.squeeze(x_RBF)
#y_RBF = torch.squeeze(y_RBF)
#z_RBF = torch.squeeze(z_RBF)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

#ax.view_init(elev=90, azim=-90)

ax.zaxis.line.set_lw(0.)
ax.set_zticks([])
ax.set_zlim(0, 0.25)

ax.scatter(x_o.cpu().detach().numpy(), y_o.cpu().detach().numpy(), z_o.cpu().detach().numpy(),
                    color=custom_colors["Blue"], antialiased=True, alpha=0.6, vmin=0, vmax=0.25)

ax.scatter(x_i.cpu().detach().numpy(), y_i.cpu().detach().numpy(), z_i.cpu().detach().numpy(),
                    color=custom_colors[3], antialiased=True, alpha=0.6, vmin=0, vmax=0.25)

plt.show()

# --------------------------------[ Plotting ]--------------------------------
"""
fig = plt.figure()
ax = plt.axes(projection="3d")

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

#ax.view_init(elev=90, azim=-90)


def animate(i):
    step = 50

    x_RBF = torch.linspace(0, 1, step, device=device)
    y_RBF = torch.linspace(0, 1, step, device=device)
    t_RBF = torch.tensor([i/100], device=device)
    x_RBF, y_RBF, t_RBF = torch.meshgrid(x_RBF, y_RBF, t_RBF, indexing='xy')

    z_RBF = interpolator.interpolate(x_RBF, y_RBF, t_RBF)

    x_RBF = torch.squeeze(x_RBF)
    y_RBF = torch.squeeze(y_RBF)
    z_RBF = torch.squeeze(z_RBF)

    ax.clear()

    ax.plot_surface(x_RBF.cpu().detach().numpy(), y_RBF.cpu().detach().numpy(), z_RBF.cpu().detach().numpy(),
                    cmap=cm.inferno, antialiased=True, alpha=0.6)

    ax.zaxis.line.set_lw(0.)
    ax.set_zticks([])

    ax.set_zlim(0, 0.25)


ani = animation.FuncAnimation(fig, animate, 100, interval=500)

writer_video = animation.PillowWriter(fps=60)
ani.save('./animation_drawing.gif', writer=writer_video)

print("Animation saved!")
"""
# --------------------------------[ Memory management ]--------------------------------



if device.type == 'cuda':
    print("\n")
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

torch.cuda.empty_cache()
