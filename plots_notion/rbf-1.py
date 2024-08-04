import numpy as np
import plotly.express as px
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "plotly_dark"


def gaussian(x, x_j, r):
    return np.e ** (-abs(x - x_j) ** 2 / (2 * r) ** 2)


def multiquadric(x, x_j, r):
    return np.sqrt((r**2) + (abs(x - x_j)**2))


def inverse_multiquadric(x, x_j, r):
    return 1 / multiquadric(x, x_j, r)


def thin_plate_spline(x, x_j, r):
    return (abs(x - x_j) ** 2) * np.log(abs(x - x_j))


sigma = 0.5  # Parámetro de suavizado

x_values = np.linspace(-2, 2, 100)
y_gaussian_values = gaussian(x_values, 0, sigma)
y_multiquadric_values = multiquadric(x_values, 0, sigma)
y_inverse_multiquadric_values = inverse_multiquadric(x_values, 0, sigma)
y_thin_plate_spline_values = thin_plate_spline(x_values, 0, sigma)

data = {
    "x": x_values,
    "gaussian": y_gaussian_values,
    "multiquadric": y_multiquadric_values,
    "inverse_multiquadric": y_inverse_multiquadric_values,
    "thin_plate_spline": y_thin_plate_spline_values
}

df = pd.DataFrame(data)

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
custom.layout.xaxis.title.standoff = 25
custom.layout.yaxis.title.standoff = 25

custom.layout.width = 800
custom.layout.height = 600

#fig = px.line(df, x="x", y=["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline"], color_discrete_sequence=[custom_colors["Blue"], custom_colors["Orange"], custom_colors["Green"], custom_colors["Purple"]])
fig = px.line(df, x="x", y="thin_plate_spline", color_discrete_sequence=[custom_colors["Purple"]])
fig.add_vline(x=0, line_dash="dash", line_color=custom_colors["Red"])
fig.update_layout(template=custom)
fig.update_layout(font=dict(size=16),  title=dict(font=dict(size=28)), xaxis_title="$\\Large{x}$", yaxis_title="$\\Large{\\phi(x)}$", legend_title_text="RBFs", legend_y=0.5, legend_title_side="top center")

fig.write_image("rbf-1.png", scale=2)
