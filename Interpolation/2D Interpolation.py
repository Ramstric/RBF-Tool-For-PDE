import torch

from RBF.Interpolator import Interpolator

from Templates.atom_dark_colors import colors
from Templates.custom_plotly import custom
import plotly.graph_objects as go

# Smoothing parameter
sigma = 1

# Measured data
time = torch.tensor([1., 2., 3., 4., 5., 6.])
temperatura = torch.tensor([20., 18., 23., 22., 21., 19.])

interpolator = Interpolator(time, f=temperatura, radius=sigma, rbf_name="gaussian")

# Interpolated data
time_interpolation = torch.linspace(1, 6, 48)
temperatura_interpolation = interpolator.interpolate(time_interpolation)

fig = go.Figure()

# Interpolated data plot
fig.add_trace(
    go.Scatter(x=time_interpolation, y=temperatura_interpolation, mode="lines", line=dict(color=colors["LightGray"]),
               showlegend=False))

# Measured data markers
fig.add_trace(
    go.Scatter(x=time, y=temperatura, mode="markers", marker=dict(color=colors["Red"], size=10, symbol="x"),
               showlegend=False))

fig.update_layout(template=custom)
fig.update_layout(xaxis_range=[0, 9], yaxis_range=[0, 24],
                  xaxis_title="Time", yaxis_title="Temperature",
                  xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C",
                  xaxis_dtick=1)

fig.show()
