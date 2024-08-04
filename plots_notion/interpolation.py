import numpy as np
import plotly.express as px
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "plotly_dark"


def gaussian(x, x_j, r):
    return np.e ** (-abs(x - x_j) ** 2 / (2 * r) ** 2)


def multiquadric(x, x_j, r):
    return np.sqrt((r ** 2) + (abs(x - x_j) ** 2))


def inverse_multiquadric(x, x_j, r):
    return 1 / multiquadric(x, x_j, r)


def thin_plate_spline(x, x_j, r):
    return (abs(x - x_j) ** 2) * np.log(abs(x - x_j))


sigma = 0.5  # Parámetro de suavizado

time = np.array([1, 2, 3, 4, 5, 6])
temperatura = np.array([20, 18, 23, 22, 21, 19])

df = pd.DataFrame({"t": time, "T": temperatura})

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
custom.layout.xaxis.title.standoff = 25
custom.layout.yaxis.title.standoff = 25

custom.layout.width = 800
custom.layout.height = 600

#fig = px.line(df, x="x", y=["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline"], color_discrete_sequence=[custom_colors["Blue"], custom_colors["Orange"], custom_colors["Green"], custom_colors["Purple"]])

fig = px.scatter(df, x="t", y="T")
fig.update_traces(marker=dict(color=custom_colors["Red"], size=10))

x_interpolation = np.linspace(1, 6, 100)
interpolation = np.interp(x_interpolation, time, temperatura)

fig.add_trace(go.Scatter(x=x_interpolation,
                         y=interpolation,
                         mode="lines",
                         line=dict(
                             color="rgba(54, 170, 186, 0.4)",
                             width=2),
                         showlegend=False))

fig.add_trace(go.Scatter(x=[x_interpolation[26]],
                         y=[interpolation[26]],
                         mode="markers",
                         marker=dict(
                             color=custom_colors["Green"],
                             size=10, symbol="cross"),
                         showlegend=False))

fig.add_annotation(x=x_interpolation[26]+0.05,
                   y=interpolation[26]-0.3,
                   text=f"Interpolated point\n({x_interpolation[26]:.2f} s, {interpolation[26]:.2f} °C)",
                   showarrow=True,
                   arrowcolor=custom_colors["Green"],
                   arrowhead=2,
                   ay=80,
                   ax=50,
                   opacity=1,
                   bgcolor='#191919')


fig.add_annotation(x=time[4]-0.01,
                   y=temperatura[4]-0.4,
                   text="Tabulated point",
                   showarrow=True,
                   arrowcolor=custom_colors["Red"],
                   arrowhead=2,
                   ay=70,
                   ax=-10,
                   opacity=1,
                   bgcolor='#191919')


fig.update_layout(template=custom)
fig.update_layout(font=dict(size=16), title=dict(font=dict(size=28)), xaxis_title="Time", yaxis_title="Temperature")
fig.update_layout(yaxis_range=[0, 24], xaxis_range=[0, 7])

fig.update_layout(xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C")

fig.data = (fig.data[2], fig.data[1], fig.data[0])
fig.show()
fig.write_image("interpolation.png", scale=2)
