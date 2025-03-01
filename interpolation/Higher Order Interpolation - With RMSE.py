import torch

from RBF.Interpolator import Interpolator

from templates.atom_dark_colors import colors
from templates.custom_plotly import custom
import plotly.graph_objects as go
import sys

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Analytical solution
def analytical_solution(_x, _y, _z):
    return 5 + 3*_x + 0.15*_x**2 + 3*_y + 0.15*_y**2 + 3*_z + 0.15*_z**2

# Polynomial matrix
def polynomial_matrix(num_points, variables=None):
    # For this code variables are [x, y, z] in that order
    _x = variables[0]
    _y = variables[1]
    _z = variables[2]

    #      torch.cat( [1, x, y, z, x^2, y^2, z^2, xy, xz, yz], dim=1)
    return torch.cat((torch.ones((num_points, 1)), _x, _y, _z, _x**2, _y**2, _z**2, _x*_y, _x*_z, _y*_z), dim=1)


# Measured data
x = torch.linspace(-5, 5, 10)
y = torch.linspace(-5, 5, 10)
z = torch.linspace(-5, 5, 10)
x, y, z = torch.meshgrid(x, y, z, indexing='xy')

# Analytical solution
f = analytical_solution(x, y, z)

k = 3 # RBF parameter
interpolator = Interpolator(x, y, z, f=f, rbf_name="polyharmonic_spline", rbf_parameter=k, polynomial_matrix=polynomial_matrix)

# Domain discretization for interpolation
x_RBF = torch.linspace(-5, 5, 50)
y_RBF = torch.linspace(-5, 5, 50)
z_RBF = torch.linspace(-5, 5, 50)
x_RBF, y_RBF, z_RBF = torch.meshgrid(x_RBF, y_RBF, z_RBF, indexing='xy')
f_analytical = analytical_solution(x_RBF, y_RBF, z_RBF) # Analytical solution
f_RBF = interpolator.interpolate(x_RBF, y_RBF, z_RBF) # Interpolated data


# RMSE
rmse = torch.sqrt(torch.mean((f_RBF - analytical_solution(x_RBF, y_RBF, z_RBF))**2))
print(f"RMSE: {rmse}")
