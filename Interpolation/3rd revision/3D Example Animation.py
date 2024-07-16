import numpy as np
import RBF_Interpolator_3D
import matplotlib.pyplot as plt
import matplotx
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import os

#mpl.use('Qt5Agg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 100
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

cmap_2 = ListedColormap([custom_colors[i] for i in range(5)])

# Datos muestreados
x_0 = np.linspace(-4, 4, 8)
y_0 = np.linspace(-4, 4, 8)
x_0, y_0 = np.meshgrid(x_0, y_0)
#z = np.cosh(x_0 - 0.5) * np.cosh(y_0 - 0.5)
z = np.sin(np.sqrt(x_0**2 + y_0**2))


interpolator = RBF_Interpolator_3D.RBFInterpolator3D("gaussian", x_0, y_0, f=z, r=10)

# Interpolación
x_RBF = np.linspace(-4, 4, 25)
y_RBF = np.linspace(-4, 4, 25)
x_RBF, y_RBF = np.meshgrid(x_RBF, y_RBF)

pairs_2 = np.asarray([x_RBF.ravel(), y_RBF.ravel()]).T

fig = plt.figure()
ax = plt.axes(projection='3d')

frames = 120
steps = np.linspace(4, 9.9, frames)
delay = 180


def animate(i):
    if i < frames:
        interpolator.radius = steps[i]
        interpolator.recalculate_weights()

        z_RBF = [interpolator.interpolate(pair) for pair in pairs_2]
        z_RBF = np.array_split(z_RBF, x_RBF.shape[0])
        z_RBF = np.stack(z_RBF, axis=0)

        # Plotting
        ax.clear()

        ax.plot_surface(x_RBF, y_RBF, z_RBF, cmap=cm.bone, antialiased=True, alpha=0.5, zorder=1)
        ax.scatter3D(x_0, y_0, z, c=z, cmap=cmap_2)

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    else:
        pass


ani = animation.FuncAnimation(fig, animate, frames+delay, interval=500)

writer_video = animation.PillowWriter(fps=60)
ani.save('animation_drawing3D.gif', writer=writer_video)

print("Animation saved!")

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'animation_drawing3D.gif')
os.startfile(filename)
