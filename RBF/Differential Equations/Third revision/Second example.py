import torch
import matplotlib.pyplot as plt
import matplotx
import RBF
from math import e

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams['figure.dpi'] = 120
custom_colors = {"Blue": "#61AFEF", "Orange": "#D49F6E", "Green": "#98C379", "Rose": "#E06C75",
                 "Purple": "#C678DD", "Gold": "#E5C07B", "Cyan": "#36AABA", 0: "#61AFEF", 1: "#D49F6E",
                 2: "#98C379", 3: "#E06C75", 4: "#C678DD", 5: "#E5C07B", 6: "#36AABA", "LightCyan": "#56B6C2",
                 "AltOrange": "#D19A66", "Red": "#BE5046", "RoyaBlue": "#528BFF", "Gray": "#ABB2BF",
                 "LightGray": "#CCCCCC", "LightBlack": "#282C34", "Black": "#1D2025"}
plt.style.use(matplotx.styles.onedark)


# Function of PDE
def f(x):
    return (0.75 * e**(3-0.5*x) * x**2 * torch.cos(e**(3-0.5*x)) - 0.25 * e**(6-x) * x**2 * torch.sin(e**(3-0.5*x)) -
            2 * e**(3-0.5*x) * x * torch.cos(e**(3-0.5*x)) + 2 * torch.sin(e**(3-0.5*x)) -
            2 * x * torch.sin(e**(3-0.5*x)))


# --------------------------------[ Datos muestreados ]--------------------------------
x_B = torch.tensor([0, 24, 11.1, 2.1], device=device)  # Boundary nodes
x_I = torch.linspace(0.01, 23.99, 100, device=device)  # Interior nodes

x_0 = torch.cat((x_B, x_I), dim=0)                  # Making the Vector X (Boundary + Interior nodes)

y_B = torch.tensor([0, 0.07108404697, 9.6106694813376, 2.9914835455256], device=device)    # Boundary conditions
y_I = f(x_I)                                               # Evaluation PDE function on the Interior nodes

y_0 = torch.cat((y_B, y_I), dim=0)                  # Making the Vector Y (Boundary conditions + Function values)

# --------------------------------[ Interpolación ]--------------------------------
x_1 = torch.linspace(0, 24, 100, device=device)

interpolator = RBF.InterpolatorPDE("multiQuad", x_0, f=y_0, r=1.6, boundary=x_B, ode="y'' - y'")
y_RBF = interpolator.interpolate(x_1)

print(interpolator.interpolate(torch.tensor([15.], device=device)))

# --------------------------------[ Plotting ]--------------------------------
fig, ax = plt.subplots(1, 1)
plt.grid()
torch.set_printoptions(precision=12)
plot = plt.plot(x_1.cpu().detach().numpy(), y_RBF.cpu().detach().numpy())
#plt.scatter(x_0.cpu().detach().numpy(), y_0.cpu().detach().numpy(), s=8, zorder=2, color=custom_colors["LightGray"])

plt.show()
