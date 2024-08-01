import numpy as np
import plotly.express as px
import pandas as pd
import plotly.io as pio
import torch
import plotly.graph_objects as go
from math import log10, floor

custom_colors = {"Blue": "#61AFEF", "Orange": "#D49F6E", "Green": "#98C379", "Rose": "#E06C75",
                 "Purple": "#C678DD", "Gold": "#E5C07B", "Cyan": "#36AABA", 0: "#61AFEF", 1: "#D49F6E",
                 2: "#98C379", 3: "#E06C75", 4: "#C678DD", 5: "#E5C07B", 6: "#36AABA", "LightCyan": "#56B6C2",
                 "AltOrange": "#D19A66", "Red": "#BE5046", "RoyaBlue": "#528BFF", "Gray": "#ABB2BF",
                 "LightGray": "#CCCCCC", "LightBlack": "#282C34", "Black": "#1D2025"}

custom = go.layout.Template()

custom.layout.plot_bgcolor = '#191919'
custom.layout.paper_bgcolor = '#191919'
custom.layout.font.color = '#d4d4d4'

custom.layout.title.font.size = 20
custom.layout.title.font.family = 'Segoe UI'
custom.layout.title.x = 0.5

custom.layout.xaxis.gridcolor = '#434343'
custom.layout.yaxis.gridcolor = '#434343'

custom.layout.margin.pad = 10
custom.layout.margin.l = 90
custom.layout.margin.t = 60
custom.layout.xaxis.title.standoff = 25
custom.layout.yaxis.title.standoff = 25

custom.layout.width = 800
custom.layout.height = 600


def find_exp(number) -> int:
    base10 = log10(abs(number))
    return abs(floor(base10))


def gaussian(x, x_j, r):
    return np.e ** (-abs(x - x_j) ** 2 / ((2 * r) ** 2))


def gaussian_derivative(x, x_j, r):
    return - (x - x_j) * gaussian(x, x_j, r) / (2 * (r ** 2))


def multiquadric(x, x_j, c):
    return np.sqrt(abs(x - x_j) ** 2 + c ** 2)


def multiquadric_derivative(x, x_j, c):
    return (x - x_j) / multiquadric(x, x_j, c)


def multiquadric_MAPS(x, x_j, c):
    return ((1 / 9) * (4 * c ** 2 + abs(x - x_j) ** 2) * multiquadric(x, x_j, c)) - (
            (c ** 3 / 3) * np.log(c + multiquadric(x, x_j, c)))


def multiquadric_MAPS_derivative(x, x_j, c):
    return ((abs(x - x_j) / 3) * (2 * c ** 2 + abs(x - x_j) ** 2) / multiquadric(x, x_j, c)) - (
            (abs(x - x_j) * c ** 3) / (3 * (c + multiquadric(x, x_j, c)) * multiquadric(x, x_j, c)))


def inverse_multiquadric(x, x_j, r):
    return 1 / multiquadric(x, x_j, r)


def ODE(t):
    return t - 7.5 + 10 / (t + 0.2)


def solution(t):
    return 10 * np.log(t + 0.2) + 0.5 * t ** 2 - 7.5 * t + 25


# Smoothing parameter
sigma = 2

# RBF to use
RBF = multiquadric
dRBF = multiquadric_derivative

# Measured data
initial_time = np.array([0])
initial_temperature = np.array([8.905])

post_time = np.arange(0.1, 6.1, 0.1)
post_temperature = ODE(post_time)

time = np.concatenate((initial_time, post_time))
temperature = np.concatenate((initial_temperature, post_temperature))

# Matrices for the RBF interpolation
#boundary = np.array([RBF(pos, time, sigma) for pos in initial_time])
#interior = np.array([dRBF(pos, time, sigma) for pos in post_time])

boundary = np.array([RBF(initial_time, pos, sigma) for pos in time]).T
interior = np.array([dRBF(post_time, pos, sigma) for pos in time]).T


matrix = np.concatenate((boundary, interior))

# Weights interpolation
omega = np.linalg.solve(matrix, temperature)

#print(omega)

omega_torch = torch.linalg.solve(torch.from_numpy(matrix), torch.from_numpy(temperature))

print(omega)
print(omega_torch)

#print(omega_torch.numpy())

#fig = px.line(df, x="x", y=["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline"], color_discrete_sequence=[custom_colors["Blue"], custom_colors["Orange"], custom_colors["Green"], custom_colors["Purple"]])

fig = go.Figure()

pair = list(zip(time, omega))


# Helper function to do the weighted sum of the RBFs
def interpolate(x, r):
    y = 0
    for (pos, w) in pair:
        y += w * RBF(x, pos, r)
    return y


# Interpolated data
time_interpol = np.linspace(0, 9, 400)
temperature_interpol = [interpolate(x, sigma) for x in time_interpol]
temperature_exact = solution(time_interpol)

error = np.square(np.subtract(temperature_exact, temperature_interpol)).mean()
error_string = "{:.2e}".format(error).split("e")
error_string = f"{error_string[0]} \\times 10^{{{error_string[1]}}}"
# Title of the plot
fig.update_layout(
    title_text=f"$\\Large{{ \\text{{Para }} {len(time)} \\text{{ puntos y suavizado }}\\sigma = {sigma} }}$",
    title_x=0.5)

# Original data plot
fig.add_trace(
    go.Scatter(x=time_interpol, y=temperature_exact, mode="lines",
               line=dict(color=custom_colors["LightGray"], width=0.75),
               showlegend=True, name="Exact solution"))

fig.add_annotation(x=6, y=15, text="$T(t)=10 \\ln{(t+0.2)}+ 0.5t^2 - 7.5t + 25$", showarrow=False,
                   font=dict(size=16, color=custom_colors["LightGray"]), bgcolor='#191919')

# Interpolated data plot
fig.add_trace(
    go.Scatter(x=time_interpol, y=temperature_interpol, mode="lines", line=dict(color=custom_colors["Red"], width=0.75),
               showlegend=True, name="Interpolated solution"))

fig.add_annotation(x=3, y=22,
                   text=f"$\\hat{{T}}(t)= w_{{1}} \\cdot \\phi_{{t,t_1}} + w_{{2}} \\cdot \\phi_{{t,t_2}} + \\dots + w_{{{len(time)}}} \\cdot \\phi_{{t,t_{{{len(time)}}}}}$",
                   showarrow=False,
                   font=dict(size=16, color='#eb6c60'), bgcolor='#191919')

# Domain discretization markers
fig.add_trace(
    go.Scatter(x=time, y=np.zeros(len(time)), mode="markers",
               marker=dict(line_color=custom_colors["Red"], size=10, symbol="line-ns", line_width=1),
               showlegend=False))

# RMSE annotation
fig.add_annotation(x=7.5, y=3, text=f"$\\text{{RMSE}} = {error_string}$", showarrow=False,
                   font=dict(size=16, color=custom_colors["LightGray"]), bgcolor='#222222', bordercolor="#434343",
                   borderwidth=0.5, borderpad=20)

fig.update_layout(template=custom, yaxis_range=[0, 24], xaxis_range=[0, 9])
fig.update_layout(font=dict(size=16), title=dict(font=dict(size=28)), xaxis_title="Time", yaxis_title="Temperature",
                  xaxis_dtick=1, xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C",
                  legend=dict(x=0.97, xanchor="right", y=0.975, traceorder="normal", font_size=12, bgcolor='#222222', bordercolor='#434343', borderwidth=0.5))

#fig.write_html("./ode-2.html", include_plotlyjs=False)
#fig.write_image("ode-2.png", scale=2)
fig.show()
