import numpy as np
import plotly.express as px
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

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


def gaussian(x, x_j, r):
    return np.e ** (-abs(x - x_j) ** 2 / (2 * r) ** 2)


def multiquadric(x, x_j, r):
    return np.sqrt((r ** 2) + (abs(x - x_j) ** 2))


def inverse_multiquadric(x, x_j, r):
    return 1 / multiquadric(x, x_j, r)


def thin_plate_spline(x, x_j, r):
    return (abs(x - x_j) ** 2) * np.log(abs(x - x_j))


# Smoothing parameter
sigma = 1.0

# Measured data
time = np.array([1, 2, 3, 4, 5, 6])
temperatura = np.array([20, 18, 23, 22, 21, 19])

# RBF to use
RBF = multiquadric

# Weights interpolation
omega = np.linalg.solve(np.array([RBF(time, pos, sigma) for pos in time]).T, temperatura)

#fig = px.line(df, x="x", y=["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline"], color_discrete_sequence=[custom_colors["Blue"], custom_colors["Orange"], custom_colors["Green"], custom_colors["Purple"]])

fig = go.Figure()

pair = list(zip(time, omega))

#RBFs plots
for (t, w) in pair:
    fig.add_trace(
        go.Scatter(x=np.linspace(0, 7, 300), y=w * RBF(np.linspace(0, 7, 300), t, sigma), mode="lines",
                   line=dict(color=custom_colors[t-1]), showlegend=False))

fig.update_traces(opacity=0.125)


# Helper function to do the weighted sum of the RBFs
def interpolate(x, r):
    y = 0
    for (pos, w) in pair:
        y += w * RBF(x, pos, r)
    return y


# Interpolated data
time_interpol = np.linspace(1, 6, 400)
temperatura_interpol = [interpolate(x, sigma) for x in time_interpol]

# Title of the plot
fig.update_layout(title_text=f"$\Large{{\\text{{Suavizado }}\\sigma = {sigma}}}$", title_x=0.5)

# Simple interpolation plot
#fig.add_trace(go.Scatter(x=time, y=temperatura, mode="lines", line=dict(color=custom_colors["Cyan"]), showlegend=False))

# Interpolated data plot
fig.add_trace(
    go.Scatter(x=time_interpol, y=temperatura_interpol, mode="lines", line=dict(color=custom_colors["LightGray"]),
               showlegend=False))

# Measured data markers
fig.add_trace(
    go.Scatter(x=time, y=temperatura, mode="markers", marker=dict(color=custom_colors["Red"], size=10, symbol="x"),
               showlegend=False))

fig.update_layout(template=custom, yaxis_range=[0, 24], xaxis_range=[0, 7])
fig.update_layout(font=dict(size=16), title=dict(font=dict(size=28)), xaxis_title="Time", yaxis_title="Temperature",
                  xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C")

fig.write_image("interpolation-6.png", scale=2)
