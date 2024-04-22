import torch
import matplotlib.pyplot as plt
import matplotx

plt.rcParams['figure.dpi'] = 120
custom_colors = {"Blue": "#61AFEF", "Orange": "#D49F6E", "Green": "#98C379", "Rose": "#E06C75",
                 "Purple": "#C678DD", "Gold": "#E5C07B", "Cyan": "#36AABA", 0: "#61AFEF", 1: "#D49F6E",
                 2: "#98C379", 3: "#E06C75", 4: "#C678DD", 5: "#E5C07B", 6: "#36AABA", "LightCyan": "#56B6C2",
                 "AltOrange": "#D19A66", "Red": "#BE5046", "RoyaBlue": "#528BFF", "Gray": "#ABB2BF",
                 "LightGray": "#CCCCCC", "LightBlack": "#282C34", "Black": "#1D2025"}
plt.style.use(matplotx.styles.onedark)

torch.set_printoptions(precision=3)


def pairs3D(x: torch.tensor, y: torch.tensor):
    # Slicing out first and last element, to get inner points
    x_inner = x[1:x.size(0) - 1]
    y_inner = x[1:x.size(0) - 1]

    # Creating meshgrid for all points and inner points
    x, y = torch.meshgrid(x, y, indexing='xy')
    x_inner, y_inner = torch.meshgrid(x_inner, y_inner, indexing='xy')

    pairs_outer = torch.stack([x.ravel(), y.ravel()]).T  # Copying all points, to remove inner points from it
    pairs_inner = torch.stack([x_inner.ravel(), y_inner.ravel()]).T  # Inner points

    # This returns the index of inners points in all points, and slices them out
    for inner_pair in pairs_inner:
        index = torch.where(torch.all(torch.eq(inner_pair, pairs_outer), dim=1))
        # Slicing is [from 0 to i and from i+1 to the end]
        pairs_outer = torch.cat((pairs_outer[:index[0].item()], pairs_outer[index[0].item() + 1:]), 0)

    # Outer points in vector form
    x_outer = torch.stack([pos[0] for pos in pairs_outer])
    y_outer = torch.stack([pos[1] for pos in pairs_outer])
    z_outer = 2*x_outer**2 + 2*y_outer**2  # <--- Change the function for z in outer points

    # Inner points in vector form
    x_in = torch.stack([pos[0] for pos in pairs_inner])
    y_in = torch.stack([pos[1] for pos in pairs_inner])
    z_in = 4*x_in + 4*y_in     # <--- Change the function for z in inner points

    # All points in vector form
    x_all = torch.cat((x_outer, x_in), 0)
    y_all = torch.cat((y_outer, y_in), 0)
    z_all = torch.cat((z_outer, z_in), 0)

    pairs_all = torch.stack([x_all.ravel(), y_all.ravel()]).T  # To obtain them pairs in order

    return x_all, y_all, z_all


# All points
x = torch.arange(-2, 2.25, 0.25)
y = torch.arange(-2, 2.25, 0.25)
size = x.size(0)


x, y, z = pairs3D(x, y)


# --------------------------------[ Plotting ]--------------------------------
fig = plt.figure()
ax = plt.axes(projection="3d")
scat_6 = ax.scatter3D(x, y, z, zorder=-1)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

#ax.view_init(elev=90, azim=-90, roll=0)


plt.show()
