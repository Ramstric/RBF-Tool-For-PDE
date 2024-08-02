import numpy as np
import torch

from Templates.custom_plotly import custom
import plotly.graph_objects as go


def ode(i, j):
    if j == 0 or j == 20:
        return 50.

    elif i == 0:
        return 0.

    else:
        return 0.5


time = np.arange(0, 2.1, 0.1)
x = np.arange(0, 2.1, 0.1)
x, time = np.meshgrid(x, time, indexing="xy")

u = np.fromfunction(np.vectorize(ode), (x.shape[0], x.shape[1]))

indices = np.argwhere(u == 0.5)

inner_x = x[indices[:, 0], indices[:, 1]]
inner_time = time[indices[:, 0], indices[:, 1]]
inner_u = np.zeros(inner_x.shape)

indices = np.argwhere(u == 50)

dirichlet_x = x[indices[:, 0], indices[:, 1]]
dirichlet_time = time[indices[:, 0], indices[:, 1]]
dirichlet_u = u[indices[:, 0], indices[:, 1]]

indices = np.argwhere(u == 0)

initial_x = x[indices[:, 0], indices[:, 1]]
initial_time = time[indices[:, 0], indices[:, 1]]
initial_u = u[indices[:, 0], indices[:, 1]]

boundary_x = np.concatenate((initial_x, dirichlet_x))
boundary_time = np.concatenate((initial_time, dirichlet_time))
boundary_u = np.concatenate((initial_u, dirichlet_u))

print(torch.from_numpy(boundary_x))
print(torch.from_numpy(boundary_time))
print(torch.from_numpy(boundary_u))

print(torch.from_numpy(inner_x))
print(torch.from_numpy(inner_time))
print(torch.from_numpy(inner_u))

fig = go.Figure(data=[go.Scatter3d(
    z=boundary_u,
    x=boundary_x,
    y=boundary_time
)])

fig.add_scatter3d(
    z=inner_u,
    x=inner_x,
    y=inner_time
)

fig.update_layout(template=custom, font_size=12)

fig.show()
