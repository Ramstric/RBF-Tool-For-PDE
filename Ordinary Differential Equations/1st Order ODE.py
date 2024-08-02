import torch

from Templates.atom_dark_colors import colors
from Templates.custom_plotly import custom
import plotly.graph_objects as go
from RBF import DifferentialEquationSolver as DESolver
from RBF import RadialBasisFunctions as rbf

torch.set_default_dtype(torch.float64)


def ode(t):
    return t - 7.5 + 10 / (t + 0.2)


def solution(t):
    return 10 * torch.log(t + 0.2) + 0.5 * t ** 2 - 7.5 * t + 25


# Smoothing parameter
sigma = 2

# Measured data

# Initial conditions
initial_time = torch.tensor([0])
initial_temperature = torch.tensor([8.905])

post_time = torch.arange(0.1, 6.1, 0.1)
post_temperature = ode(post_time)

time = torch.cat((initial_time, post_time))
temperature = torch.cat((initial_temperature, post_temperature))

derivative_operator = rbf.multiquadric_derivative

fig = go.Figure()

DEInterpolator = DESolver.DifferentialInterpolator(boundary=[initial_time], inner=[post_time], f=temperature,
                                                   radius=sigma, rbf_name="multiquadric",
                                                   derivative_operator=derivative_operator)

# Interpolated data
time_interpol = torch.linspace(0, 6, 60)
temperature_interpol = DEInterpolator.interpolate(time_interpol)

temperature_exact = solution(time_interpol)

# Original data plot
fig.add_trace(
    go.Scatter(x=time_interpol, y=temperature_exact, mode="lines",
               line=dict(color=colors["LightGray"], width=0.75),
               showlegend=True, name="Exact solution"))

# Interpolated data plot
fig.add_trace(
    go.Scatter(x=time_interpol, y=temperature_interpol, mode="lines",
               line=dict(color=colors["Red"], width=0.75),
               showlegend=True, name="Interpolated solution"))

fig.update_layout(template=custom)
fig.update_layout(yaxis_range=[0, 24], xaxis_range=[0, 6],
                  xaxis_title="Time", yaxis_title="Temperature", xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C", )

fig.show()
