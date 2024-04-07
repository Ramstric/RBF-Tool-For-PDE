import numpy as np
import matplotlib.pyplot as plt
import matplotx
import RBF_Interpolator_3D
from matplotlib import cm
#mpl.use('Qt5Agg')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 200
plt.style.use(matplotx.styles.onedark)

# Datos muestreados
x_0 = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
y_0 = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])

x_0, y_0 = np.meshgrid(x_0, y_0)
z = np.cosh(x_0 - 0.5) * np.cosh(y_0 - 0.5)

interpolator = RBF_Interpolator_3D.RBFInterpolator3D("multiQuad", x_0, y_0, f=z, r=10)

# Interpolación
x_RBF = np.arange(0, 1 + 1/35, 1/35)
y_RBF = np.arange(0, 1 + 1/35, 1/35)

x_RBF, y_RBF = np.meshgrid(x_RBF, y_RBF)

pairs_2 = np.asarray([x_RBF.ravel(), y_RBF.ravel()]).T

z_RBF = [interpolator.interpolate(pair) for pair in pairs_2]
z_RBF = np.array_split(z_RBF, x_RBF.shape[0])
z_RBF = np.stack(z_RBF, axis=0)

# Plotting
ax = plt.axes(projection="3d")
surf = ax.plot_surface(x_RBF, y_RBF, z_RBF, cmap=cm.plasma, linewidth=0, antialiased=False, alpha=0.2, zorder=1)
scat = ax.scatter3D(x_0, y_0, z, s=25, zorder=4, c=z, cmap=cm.plasma_r)

plt.show()
