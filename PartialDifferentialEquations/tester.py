import torch

from RBF.Interpolator import Interpolator

from templates.atom_dark_colors import colors
from templates.custom_plotly import custom
import plotly.graph_objects as go

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Analytical solution
def analytical_solution(_x, _y):
    # List of 3D functions
    return 5 + 3*_x + 0.15*_x**2 + 3*_y + 0.15*_y**2 # 3D Paraboloid
    #return torch.sin(torch.sqrt(_x**2 + _y**2)) # 3D Vortex
    #return torch.sin(_x) * torch.sin(_y) # 3D Saddle

# Polynomial matrix
def polynomial_matrix(num_points, variables=None):
    # For this code variables are [x, y] in that order
    _x = variables[0]
    _y = variables[1]
    return torch.cat((torch.ones((num_points, 1)), _x, _y, _x**2, _x*_y, _y**2), dim=1) # torch.cat( [1, x, y, x^2, xy, y^2], dim=1)

# Measured data
x = torch.linspace(-5, 5, 20)
y = torch.linspace(-5, 5, 20)
x, y = torch.meshgrid(x, y, indexing='xy')
z = analytical_solution(x, y)

k = 2 # RBF parameter
interpolator = Interpolator(x, y, f=z, rbf_name="polyharmonic_spline", rbf_parameter=k, polynomial_matrix=polynomial_matrix)

# Domain discretization for interpolation
x_RBF = torch.linspace(-5, 5, 40)
y_RBF = torch.linspace(-5, 5, 40)
x_RBF, y_RBF = torch.meshgrid(x_RBF, y_RBF, indexing='xy')
z_analytical = analytical_solution(x_RBF, y_RBF) # Analytical solution
z_RBF = interpolator.interpolate(x_RBF, y_RBF) # Interpolated data

# RMSE
rmse = torch.sqrt(torch.mean((z_RBF - analytical_solution(x_RBF, y_RBF))**2))
print(f"RMSE: {rmse}")

# Plotting
fig = go.Figure()

# Analytical data surface
fig.add_trace(go.Surface(z=z_analytical, x=x_RBF, y=y_RBF,
                         colorscale=[colors["Red"], colors["Red"]], opacity=0.5,
                         name="Analytical", showlegend=True, showscale=False))

# Interpolated data surface
fig.add_trace(go.Surface(z=z_RBF, x=x_RBF, y=y_RBF,
                         colorscale="turbid", opacity=0.75,
                         name="Interpolated", showlegend=True, showscale=False))

# Measured data markers
fig.add_trace(go.Scatter3d(x=x.flatten(), y=y.flatten(), z=z.flatten(),
                           mode="markers", marker=dict(color=colors["Red"], size=2, symbol="circle"),
                           name="Data markers", showlegend=True))

fig.update_layout(template=custom, font_size=10, scene=dict(aspectmode="cube"),
                  scene_xaxis_title="X", scene_yaxis_title="Y", scene_zaxis_title="Z")

fig.show()
