import torch

from templates.atom_dark_colors import colors
from templates.custom_plotly import custom
import plotly.graph_objects as go
from RBF import DifferentialEquationSolver as DESolver
from RBF import RadialBasisFunctions as rbf

torch.set_default_dtype(torch.float64)


def derivative_operator(r, radius):
    #return rbf.multiquadric_derivative(r, radius)
    #return rbf.multiquadric_derivative(r, radius) - rbf.multiquadric(r, radius)
    return rbf.multiquadric_second_derivative(r, radius) + rbf.multiquadric_derivative(r, radius)


def ode(t):
    #return torch.exp(t-4) + 3*(t-1)**2 + torch.sin(t-3)                                       # -> y' = ...
    #return 3 * (t - 1) ** 2 - (t - 1) ** 3 + torch.sin(t - 3) + torch.cos(t - 3)              # -> y' - y = ...
    return 2*torch.exp(t - 4) + 6*(t - 1) + 3*(t - 1)**2 + torch.sin(t - 3) + torch.cos(t - 3)  # -> y'' + y' = ...


def solution(t):
    return torch.exp(t - 4) + (t - 1) ** 3 - torch.cos(t - 3)


# Smoothing parameter
sigma = 2

# Measured data

# Dirichlet boundary conditions
initial_time = torch.tensor([0., 2.75])
initial_temperature = torch.tensor([0.0083, 4.6769])

post_time = torch.arange(0.1, 2.75, 0.1)
post_temperature = ode(post_time)

time = torch.cat((initial_time, post_time))
temperature = torch.cat((initial_temperature, post_temperature))

fig = go.Figure()

DEInterpolator = DESolver.DifferentialInterpolator(boundary=[initial_time], inner=[post_time], f=temperature,
                                                   radius=sigma, rbf_name="multiquadric",
                                                   derivative_operator=derivative_operator)

# Interpolated data
time_interpol = torch.linspace(0, 2.5, 60)
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
fig.update_layout(yaxis_range=[0, 4], xaxis_range=[0, 2.75],
                  xaxis_title="Time", yaxis_title="Temperature", xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C", )

fig.show()
