import numpy as np

from RBF import WeightsInterpolator

from templates.custom_plotly import custom
import plotly.graph_objects as go

weights = np.load("PartialDifferentialEquations/weights_matrix.npy")
center_points = np.load("PartialDifferentialEquations/center_points.npy")

Interpolator = WeightsInterpolator.WeightsInterpolator(center_points, weights, 0.2, "multiquadric")

# Interpolated data
x_interpol = np.linspace(0, 2, 50)

for i in np.arange(0, 1, 0.05):
    t_interpol = np.full(x_interpol.shape, i)

    u_interpol = Interpolator.interpolate(x_interpol, t_interpol)

    # Plotting
    fig = go.Figure(data=[go.Scatter(
        x=x_interpol,
        y=u_interpol
    )])

    fig.update_layout(template=custom, font_size=12, xaxis_range=[0, 2], yaxis_range=[-4, 4])

    fig.show()
