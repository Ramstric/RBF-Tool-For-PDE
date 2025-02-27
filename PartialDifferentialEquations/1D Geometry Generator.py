import numpy as np
import torch

from templates.custom_plotly import custom
import plotly.graph_objects as go


def ode(i, j):
    # Initial conditions
    # When u(x, 0) = 20sin(0.16x)
    if i == 0:
        return 0.

    # Dirichlet boundary conditions
    # When u(0, t) = 0 and u(1, t) = 0
    elif j == 0:
        return 0.
    elif j == 20 or j == 0:
        return np.sin(0.126 * i) # Dirichlet boundary conditions index

    else:
        # Inner points index
        return 0.5


time = np.arange(0, 1.05, 0.05)
x = np.arange(0, 1.05, 0.05)
x, time = np.meshgrid(x, time, indexing="xy")

u = np.fromfunction(np.vectorize(ode), (x.shape[0], x.shape[1]))

indices = np.argwhere(u == 0.5) # Take note of the inner points index!

inner_x = x[indices[:, 0], indices[:, 1]]
inner_time = time[indices[:, 0], indices[:, 1]]
inner_u = np.zeros(inner_x.shape)

indices = np.argwhere(u == 50) # Take note of the Dirichlet boundary conditions index!

dirichlet_x = x[indices[:, 0], indices[:, 1]]
dirichlet_time = time[indices[:, 0], indices[:, 1]]
dirichlet_u = np.zeros(dirichlet_x.shape)

indices = np.argwhere((u != 50) & (u != 0.5)) # Exclude the Dirichlet boundary conditions and inner points

initial_x = x[indices[:, 0], indices[:, 1]]
initial_time = time[indices[:, 0], indices[:, 1]]
initial_u = u[indices[:, 0], indices[:, 1]]

boundary_x = np.concatenate((initial_x, dirichlet_x))
boundary_time = np.concatenate((initial_time, dirichlet_time))
boundary_u = np.concatenate((initial_u, dirichlet_u))

fig = go.Figure(data=[go.Scatter3d(
    z=inner_u,
    x=inner_x,
    y=inner_time,
    name="Inner points"
)])

fig.add_scatter3d(
    z=initial_u,
    x=initial_x,
    y=initial_time,
    name="Initial points"
)

fig.add_scatter3d(
    z=dirichlet_u,
    x=dirichlet_x,
    y=dirichlet_time,
    name="Dirichlet points"
)

fig.add_scatter3d(
    z=boundary_u,
    x=boundary_x,
    y=boundary_time,
    name="Boundary points"
)

fig.update_layout(template=custom, font_size=12)

fig.show()

# Save np arrays
np.save("./temp_geometry/boundary_x.npy", boundary_x)
np.save("./temp_geometry/boundary_time.npy", boundary_time)
np.save("./temp_geometry/boundary_u.npy", boundary_u)

np.save("./temp_geometry/inner_x.npy", inner_x)
np.save("./temp_geometry/inner_time.npy", inner_time)
np.save("./temp_geometry/inner_u.npy", inner_u)

