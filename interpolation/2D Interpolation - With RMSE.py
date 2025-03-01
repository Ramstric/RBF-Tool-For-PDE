import torch

from RBF.Interpolator import Interpolator

from templates.atom_dark_colors import colors
from templates.custom_plotly import custom
import plotly.graph_objects as go

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Analytical solution
def analytical_solution(t):
    # f =  20 + 3 t - 0.5 t^2
    return 5 + 3*t + 0.15*t**2

def polynomial_matrix(num_points, variables=None):
    # For this code variables are [x] in that order
    x = variables[0]
    return torch.cat((torch.ones((num_points, 1)), x, x**2), dim=1) # torch.cat( [1, x, x^2], dim=1)

# Measured data
time = torch.linspace(1, 10, 10)
temperatura = analytical_solution(time)

k = 1 # K parameter of Poly-harmonic Spline
interpolator = Interpolator(time, f=temperatura, rbf_name="polyharmonic_spline", rbf_parameter=k, polynomial_matrix=polynomial_matrix)

# Time steps for interpolation
time_interpolation = torch.linspace(time[0], time[-1], 15)
analytical_data = analytical_solution(time_interpolation) # Analytical data
temperatura_interpolation = interpolator.interpolate(time_interpolation) # Interpolated data

# RMSE
rmse = torch.sqrt(torch.mean((temperatura_interpolation - analytical_solution(time_interpolation))**2))

# Plotting
fig = go.Figure()

# Measured data markers
fig.add_trace(
    go.Scatter(x=time, y=temperatura, mode="markers", marker=dict(color=colors["Red"], size=6, symbol="x"),
               name="Data markers", showlegend=True, legendgroup="data", legendgrouptitle=dict(text="Data")))

# Data point to add RMSE value to the legend in scientific notation
fig.add_trace(
    go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color=colors["Blue"], size=0.1, symbol="circle"),
               name=f"RMSE: {rmse:.2e}", showlegend=True, legendgroup="data"))

# Analytical data plot
fig.add_trace(
    go.Scatter(x=time_interpolation, y=analytical_data, mode="lines", line=dict(color=colors["Red"]),
               showlegend=True, name="Analytical", legendgroup="plots", legendgrouptitle=dict(text="Plots")))

# Interpolated data plot
fig.add_trace(
    go.Scatter(x=time_interpolation, y=temperatura_interpolation, mode="lines", line=dict(color=colors["LightGray"]),
               showlegend=True, name="Interpolated", legendgroup="plots"))



fig.update_layout(template=custom)
fig.update_layout(xaxis_range=[time[0]-1, time[-1]+1], yaxis_range=[temperatura.min() - 5, temperatura.max()+5],
                  xaxis_title="Time", yaxis_title="Temperature",
                  xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C",
                  xaxis_dtick=1)

fig.show()
