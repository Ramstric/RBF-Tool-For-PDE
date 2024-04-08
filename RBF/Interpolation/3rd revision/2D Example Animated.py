# Ramon Everardo Hernandez Hernandez | 6 april 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import numpy as np
import matplotlib.pyplot as plt
import matplotx
import RBF_Interpolator
import matplotlib.animation as animation
from labellines import labelLines


plt.rcParams['figure.dpi'] = 120
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
# mpl.use('Qt5Agg')
plt.style.use(matplotx.styles.onedark)


# Datos muestreados
x_0 = np.array([0, 0.054, 0.259, 0.350, 0.482, 0.624, 0.679, 0.770, 1.037, 1.333, 1.505, 1.688, 1.933, 2.283])
y_0 = np.array([0, 0.633, 3.954, 3.697, 1.755, 0.679, 0.422, 0.375, 2.574, 5.428, 5.428, 4.141, -0.326, -2.220])

interpolator = RBF_Interpolator.RBFInterpolator("gaussian", x_0, y_0, 1)

# Interpolación
x_1 = np.linspace(0, 2.5, num=100)

fig, ax = plt.subplots(1, 1)


def animate(i):
    interpolator.radius = (i+1)/101
    interpolator.recalculate_weights()

    y_RBF = [interpolator.interpolate(x) for x in x_1]

    ax.clear()

    # Plotting
    ax.grid()
    ax.scatter(x_0, y_0, s=8, zorder=2, color=custom_colors["LightGray"])

    ax.plot(x_1, y_RBF, label=f"r={i/100}", color=(i/100, 0.35, 0.6))
    labelLines(ax.get_lines(), align=False, fontsize=14)

    ax.set_xlim(0, 2.5)
    ax.set_ylim(-3, 6)


ani = animation.FuncAnimation(fig, animate, 100, interval=500)

writer_video = animation.PillowWriter(fps=60)
ani.save('animation_drawing.gif', writer=writer_video)
