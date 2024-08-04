import torch

from RBF.Interpolator import Interpolator

from templates.atom_dark_colors import colors
from templates.custom_plotly import custom
import plotly.graph_objects as go

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Smoothing parameter
sigma = 1

# Measured data
x = torch.linspace(-5, 5, 10)
y = torch.linspace(-5, 5, 10)
x, y = torch.meshgrid(x, y, indexing='xy')

z = torch.sin(torch.sqrt(x**2 + y**2))

interpolator = Interpolator(x, y, f=z, radius=sigma, rbf_name="gaussian")

# Interpolated data
x_RBF = torch.linspace(-5, 5, 40)
y_RBF = torch.linspace(-5, 5, 40)
x_RBF, y_RBF = torch.meshgrid(x_RBF, y_RBF, indexing='xy')

z_RBF = interpolator.interpolate(x_RBF, y_RBF)

# Plotting
fig = go.Figure(data=[go.Surface(z=z_RBF, x=x_RBF, y=y_RBF, colorscale="dense_r")])

fig.update_layout(template=custom, font_size=12)

fig.show()
