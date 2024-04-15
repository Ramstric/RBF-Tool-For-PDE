import torch
import RBF_Interpolator
import numpy as np

import matplotlib.pyplot as plt
import matplotx
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import os
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------[ Plot style ]--------------------------------
#mpl.use('Qt5Agg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 150
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
x_0 = torch.linspace(-10, 10, 10, device=device)
y_0 = torch.linspace(-10, 10, 10, device=device)
x_0, y_0 = torch.meshgrid(x_0, y_0, indexing='xy')
#z = torch.sin(torch.sqrt(x_0**2 + y_0**2))/torch.sqrt(x_0**2 + y_0**2)
z = 1 - abs(x_0+y_0)-abs(y_0-x_0)

radius = 1
interpolator = RBF_Interpolator.RBFInterpolator3D("gaussian", x_0, y_0, f=z, r=radius)

# --------------------------------[ Interpolación ]--------------------------------
step = 1000

x_RBF = torch.linspace(-10, 10, step, device=device)
y_RBF = torch.linspace(-10, 10, step, device=device)
x_RBF, y_RBF = torch.meshgrid(x_RBF, y_RBF, indexing='xy')


fig = plt.figure()
ax = plt.axes(projection="3d")

frames = 120
steps_1 = torch.linspace(0, 10, 60)
steps = torch.cat((steps_1, torch.flip(steps_1, [0])))
delay = 0


def animate(i):
    if i < frames:
        interpolator.radius = steps[i]
        interpolator.recalculate_weights()

        z_RBF = interpolator.interpolate(x_RBF, y_RBF)
        z_RBF = z_RBF + 0.001

        # Plotting
        ax.clear()

        ax.scatter3D(x_0.cpu().detach().numpy(), y_0.cpu().detach().numpy(), z.cpu().detach().numpy(),
                            c=z.cpu().detach().numpy(), cmap=cmap_2, zorder=-1)
        ax.plot_surface(x_RBF.cpu().detach().numpy(), y_RBF.cpu().detach().numpy(), z_RBF.cpu().detach().numpy(),
                               cmap=cm.bone, antialiased=True, alpha=0.8)

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        ax.set_zlim(-20, 0)

        torch.cuda.empty_cache()
    else:
        pass


ani = animation.FuncAnimation(fig, animate, frames+delay, interval=500)

writer_video = animation.PillowWriter(fps=60)
ani.save('animation_drawing3D.gif', writer=writer_video)

print("Animation saved!")

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'animation_drawing3D.gif')
os.startfile(filename)