import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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


def T(t):
    return 10 * np.log(t + 0.2) + 0.5 * t ** 2 - 7.5 * t + 25


def dT(t):
    return t - 7.5 + 10 / (t + 0.2)


time = np.arange(0, 8, 0.025)
temperature = [T(t) for t in time]
derivative = [dT(t) for t in time]

fig = go.Figure()

# Temp plot
fig.add_trace(go.Scatter(x=time,
                         y=temperature,
                         mode='lines+markers',
                         marker_color=temperature,
                         marker_size=4,
                         marker_colorscale='inferno',
                         line=dict(color="rgba(51, 22, 128, 0.25)")))

measure_time = 0.2
measure_temperature = T(measure_time)


def slope(x, x1, y1):
    return dT(x1)*(x - x1) + y1


x_slope = np.linspace(measure_time-0.1, measure_time+0.1, 2)
y_slope = [slope(x, measure_time, measure_temperature) for x in x_slope]


n_colors = 100
colors = px.colors.sample_colorscale("inferno", [n/(n_colors - 1) for n in range(n_colors)])

percentage_color = int((measure_temperature - 8.906) * (100 / 11.278))

# Derivative plot
fig.add_trace(go.Scatter(x=[measure_time],
                         y=[measure_temperature],
                         mode='markers',
                         marker_symbol='x-thin',
                         marker_line_color=colors[percentage_color],
                         marker_line_width=2,
                         marker_size=12,
                         showlegend=False))

# Derivative arrow plot
fig.add_trace(go.Scatter(x=x_slope,
                         y=y_slope,
                         #mode='lines',
                         #line=dict(color=colors[percentage_color], width=3),
                         marker=dict(color=colors[percentage_color], size=15, symbol="arrow-bar-up", angleref="previous"),
                         showlegend=False))

fig.add_annotation(x=measure_time, y=measure_temperature, text=f"$T'(t)={measure_temperature:.2f}$", showarrow=False,
                   font=dict(size=16, color=colors[percentage_color]), bgcolor='#191919')

fig.add_annotation(x=6, y=15, text="$T(t)=10 \\ln{(t+0.2)}+ 0.5t^2 - 7.5t + 25$", showarrow=False,
                   font=dict(size=16, color="#ffa32d"), bgcolor='#191919')

fig.update_layout(template=custom, yaxis_range=[0, 25], xaxis_range=[0, 8], showlegend=False)
fig.update_layout(font=dict(size=16), title=dict(font=dict(size=28)), xaxis_title="Time", yaxis_title="Temperature",
                  xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C")

fig.write_image("derivative-2.png", scale=2)

fig.show()
