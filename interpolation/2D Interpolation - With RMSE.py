import torch

from RBF.Interpolator import Interpolator

from templates.atom_dark_colors import colors
from templates.custom_plotly import custom
import plotly.graph_objects as go

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Analytical solution
# T(t) = 20 + 3t - 0.5t^2
def analytical_solution(t):
    return 20 + 3*t - 0.5*t**2

# Measured data
time = torch.linspace(1, 6, 6)
temperatura = analytical_solution(time)

k = 1 # K parameter of Poly-harmonic Spline
interpolator = Interpolator(time, f=temperatura, radius=k, rbf_name="polyharmonic_spline")

# Time steps for interpolation
time_interpolation = torch.linspace(1, 6, 48)

# Analytical data
analytical_data = analytical_solution(time_interpolation)

# Interpolated data
temperatura_interpolation = interpolator.interpolate(time_interpolation)

# RMSE
rmse = torch.sqrt(torch.mean((temperatura_interpolation - analytical_solution(time_interpolation))**2))

print(f"RMSE: {rmse}")

# Plotting
fig = go.Figure()

# Analytical data plot
fig.add_trace(
    go.Scatter(x=time_interpolation, y=analytical_data, mode="lines", line=dict(color=colors["Blue"]),
               showlegend=True, name="Analytical"))


# Interpolated data plot
fig.add_trace(
    go.Scatter(x=time_interpolation, y=temperatura_interpolation, mode="lines", line=dict(color=colors["LightGray"]),
               showlegend=True, name="Interpolated"))

# Measured data markers
fig.add_trace(
    go.Scatter(x=time, y=temperatura, mode="markers", marker=dict(color=colors["Red"], size=10, symbol="x"),
               showlegend=False))

fig.update_layout(template=custom)
fig.update_layout(xaxis_range=[0, 7], yaxis_range=[10, 25],
                  xaxis_title="Time", yaxis_title="Temperature",
                  xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C",
                  xaxis_dtick=1)

fig.show()
