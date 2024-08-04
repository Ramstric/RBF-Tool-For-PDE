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


sigma = 0.3  # Parámetro de suavizado

time = np.array([1, 2, 3, 4, 5, 6])
temperatura = np.array([20, 18, 23, 22, 21, 19])

df = pd.DataFrame({"t": time, "T": temperatura})

#fig = px.line(df, x="x", y=["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline"], color_discrete_sequence=[custom_colors["Blue"], custom_colors["Orange"], custom_colors["Green"], custom_colors["Purple"]])

fig = go.Figure()

fig.add_trace(go.Scatter(x=np.linspace(0, 7, 300), y=temperatura[0]*gaussian(np.linspace(0, 7, 300), 1, sigma), mode="lines", line=dict(color=custom_colors["Blue"]), showlegend=False))      # Fig 0 - Line plot

fig.add_trace(go.Scatter(x=np.linspace(0, 7, 300), y=temperatura[1]*gaussian(np.linspace(0, 7, 300), 2, sigma), mode="lines", line=dict(color=custom_colors["Orange"]), showlegend=False))    # Fig 1 - Line plot

fig.add_trace(go.Scatter(x=np.linspace(0, 7, 300), y=temperatura[2]*gaussian(np.linspace(0, 7, 300), 3, sigma), mode="lines", line=dict(color=custom_colors["Green"]), showlegend=False))     # Fig 2 - Line plot

fig.add_trace(go.Scatter(x=np.linspace(0, 7, 300), y=temperatura[3]*gaussian(np.linspace(0, 7, 300), 4, sigma), mode="lines", line=dict(color=custom_colors["Rose"]), showlegend=False))      # Fig 3 - Line plot

fig.add_trace(go.Scatter(x=np.linspace(0, 7, 300), y=temperatura[4]*gaussian(np.linspace(0, 7, 300), 5, sigma), mode="lines", line=dict(color=custom_colors["Purple"]), showlegend=False))    # Fig 4 - Line plot

fig.add_trace(go.Scatter(x=np.linspace(0, 7, 300), y=temperatura[5]*gaussian(np.linspace(0, 7, 300), 6, sigma), mode="lines", line=dict(color=custom_colors["Gold"]), showlegend=False))      # Fig 5 - Line plot

fig.update_traces(opacity=0.2)

def interpolate(x, r):
    y = 0
    for (pos, t) in zip(time, temperatura):

        y += t*gaussian(x, pos, r)

    return y


time_interpol = np.linspace(0, 7, 400)
temperatura_interpol = [interpolate(x, sigma) for x in time_interpol]

fig.add_trace(go.Scatter(x=time_interpol, y=temperatura_interpol, mode="lines", line=dict(color=custom_colors["LightGray"], width=2), showlegend=False))  # Fig 7 - Line plot

fig.add_trace(go.Scatter(x=time, y=temperatura, mode="markers", marker=dict(color=custom_colors["Red"], size=10, symbol="x"), showlegend=False))  # Fig 7 - Line plot

fig.update_layout(template=custom)
fig.update_layout(font=dict(size=16), title=dict(font=dict(size=28)), xaxis_title="Time", yaxis_title="Temperature")
fig.update_layout(yaxis_range=[0, 27], xaxis_range=[0, 7])

fig.update_layout(xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C")

#fig.data = (fig.data[1], fig.data[2], fig.data[3], fig.data[4], fig.data[5], fig.data[6], fig.data[0])

fig.write_image("interpolation-3.png", scale=2)
