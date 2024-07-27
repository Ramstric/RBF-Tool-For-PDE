import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

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

custom.layout.xaxis.gridcolor = '#191919'
custom.layout.yaxis.gridcolor = '#191919'

custom.layout.margin.pad = 10
custom.layout.margin.l = 90
custom.layout.margin.t = 60
custom.layout.xaxis.title.standoff = 25
custom.layout.yaxis.title.standoff = 25

custom.layout.width = 800
custom.layout.height = 600


def f(x, c):
    return x**2 + c


def df(x):
    return 2*x


x, y = np.meshgrid(np.arange(-4, 4.5, .5), np.arange(-4, 4.5, .5))
dy = df(x)
dx = np.ones(dy.shape)

dyu = dy / np.sqrt(dx**2 + dy**2)
dxu = dx / np.sqrt(dx**2 + dy**2)

fig = ff.create_quiver(x, y, dxu, dyu, scale=.15, arrow_scale=.35, line=dict(width=1, color="rgba(97, 175, 239, 0.5)"))

x = np.arange(-4, 4.5, .1)

for i in range(0, 6):
    fig.add_trace(go.Scatter(x=x, y=f(x, i - 3), mode="lines", line=dict(color=custom_colors[i+1], width=1)))

fig.update_layout(template=custom, yaxis_range=[-4, 4], xaxis_range=[-4, 4], showlegend=False)
fig.update_layout(font=dict(size=16), title=dict(font=dict(size=28)), xaxis_title="", yaxis_title="",
                  xaxis_ticksuffix="", yaxis_ticksuffix="")

fig.write_image("DiffEq-1.png", scale=2)
