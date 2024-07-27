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


def gaussian_derivative(x, x_j, r):
    return - (x - x_j) * gaussian(x, x_j, r) / (2* r ** 2)


def multiquadric(x, x_j, r):
    return np.sqrt((r ** 2) + (abs(x - x_j) ** 2))


def inverse_multiquadric(x, x_j, r):
    return 1 / multiquadric(x, x_j, r)


def ODE(t):
    return t - 7.5 + 10/(t + 0.2)

def solution(t):
    return 10 * np.log(t + 0.2) + 0.5 * t ** 2 - 7.5 * t + 25


# Smoothing parameter
sigma = 1

# Measured data
time = np.arange(0, 6.05, 0.05)
temperatura = solution(time)

# RBF to use
RBF = multiquadric

# Weights interpolation
omega = np.linalg.solve(np.array([RBF(pos, time, sigma) for pos in time]), temperatura)

#fig = px.line(df, x="x", y=["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline"], color_discrete_sequence=[custom_colors["Blue"], custom_colors["Orange"], custom_colors["Green"], custom_colors["Purple"]])

fig = go.Figure()

pair = list(zip(time, omega))

print(pair)

#RBFs plots
for (t, w) in pair:
    fig.add_trace(
        go.Scatter(x=np.linspace(0, 7, 300), y=w * RBF(np.linspace(0, 7, 300), t, sigma), mode="lines",
                   line=dict(color=custom_colors[int(t)]), showlegend=False))

fig.update_traces(opacity=0.125)


# Helper function to do the weighted sum of the RBFs
def interpolate(x, r):
    y = 0
    for (pos, w) in pair:
        y += w * RBF(x, pos, r)
    return y


other_weights = (174913978.38933226, -196551217.38605022, -5520328460.573477, 31539430238.554302, -84606484172.82289, 129834401605.84781, -96719817504.80086, -41453282096.09073, 200055837618.47495, -244147455897.5314, 135091319038.54874, 21228971402.39146, -98844971373.55957, 94034403250.5805, -105006336569.7814, 184344390869.72696, -261602191070.7306, 233285365144.95724, -100512287100.9186, -20556785640.004375, 28602011916.259094, 50153816108.38209, -100404733343.11081, 57868129458.56967, 21209406431.720993, -45201870087.98557, 14438197027.540659, -20108708849.61706, 110859005432.5102, -206486075172.35684, 186185054792.62744, -45089456284.390274, -88039577088.40678, 99608391261.67818, -15541117670.485285, -48501594636.55432, 27767645099.97808, 40715802136.00819, -89926064811.46655, 95873453125.95506, -73184939977.01201, 41744998195.584854, -21989119551.459816, 35109931911.94975, -77112175643.98213, 105493621774.81186, -79384436597.28604, 8128768076.201613, 62350759489.621826, -105103636856.7156, 126096594977.82608, -124889235284.86627, 80999639437.53688, -214898103.88770533, -57236457808.20949, 34557228376.32521, 44013801528.29448, -92635046000.5692, 67913691180.32649, -17964805181.96626, 7340328954.0983, -24283264697.380325, 2515191079.7133117, 74879484567.08766, -139420669922.67212, 121612568636.41554, -39691852464.88343, -24841841161.345383, 26100859488.948917, 1751062268.4993474, 2191980830.240219, -54449055090.671745, 118671302360.02782, -149200718347.39532, 128334787894.38466, -71151695481.43625, 10624018555.741674, 22912332288.763874, -20624337514.563637, -756277265.7004007, 17113301550.557238, -19788605786.299076, 17281772024.887436, -17354985567.567024, 16696083020.692787, -11147070616.26764, 7177020503.342124, -13899547098.329918, 26643949909.71467, -29173969041.677254, 15216514784.222763, 1441567160.349085, -5931498289.301671, 4846610961.107138, -17570169077.440125, 45593672301.48985, -64366593476.17997, 52958198572.35698, -21780116446.958115, -362265479.9601826, 990161754.3587967, 7381468618.813928, -11084377821.59663, 12079403451.256863, -16122794168.975945, 18170527443.94847, -10614667305.6922, -2094326929.0407627, 9344613957.028688, -10897343454.363035, 15213226293.120535, -21998001242.07677, 18881630723.766247, 564579331.651978, -24817656508.26352, 36446991321.40953, -30737870883.035614, 17138702857.773594, -6317306599.193797, 1412970463.5646994, -146547143.075125)
pair_2 = list(zip(time, other_weights))

print(pair_2)

def other_interpolate(x, r):
    y = 0
    for (pos, w) in pair_2:
        y += w * RBF(x, pos, r)
    return y


# Interpolated data
time_interpol = np.linspace(0, 6, 400)
temperatura_interpol = [interpolate(x, sigma) for x in time_interpol]
temperatura_interpol_2 = [other_interpolate(x, sigma) for x in time_interpol]

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
    go.Scatter(x=time_interpol, y=temperatura_interpol_2, mode="lines", marker=dict(color=custom_colors["Red"]),
               showlegend=False))

fig.update_layout(template=custom, yaxis_range=[0, 24], xaxis_range=[0, 7])
fig.update_layout(font=dict(size=16), title=dict(font=dict(size=28)), xaxis_title="Time", yaxis_title="Temperature",
                  xaxis_ticksuffix=" s", yaxis_ticksuffix=" °C")

fig.write_image("ode-1.png", scale=2)
