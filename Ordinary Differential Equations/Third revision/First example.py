import torch
import matplotlib.pyplot as plt
import matplotx
import RBF

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
    return 0.25*torch.sin(x) + 0.5*torch.cos(x) + 0.25*x*torch.cos(x)


# --------------------------------[ Datos muestreados ]--------------------------------
x_B = torch.tensor([0, 2, 14.2074], device=device)      # Boundary nodes
x_I = torch.linspace(0.5, 14, 50, device=device)  # Interior nodes

x_0 = torch.cat((x_B, x_I), dim=0)                 # Making the Vector X (Boundary + Interior nodes)

y_B = torch.tensor([0, 0.4546487134128, 3.543], device=device)    # Boundary conditions
y_I = f(x_I)                          # Evaluation PDE function on the Interior nodes

y_0 = torch.cat((y_B, y_I), dim=0)                 # Making the Vector Y (Boundary conditions + Function values)

interpolator = RBF.InterpolatorPDE("gaussian", x_0, f=y_0, r=0.9, boundary=x_B, ode="y + y' + y''")

# --------------------------------[ Interpolación ]--------------------------------
x_1 = torch.linspace(0, 14.3, 80, device=device)

y_RBF = interpolator.interpolate(x_1)

# --------------------------------[ Plotting ]--------------------------------
fig, ax = plt.subplots(1, 1)
plt.grid()

plot = plt.plot(x_1.cpu().detach().numpy(), y_RBF.cpu().detach().numpy())
plt.scatter(x_0.cpu().detach().numpy(), y_0.cpu().detach().numpy(), s=8, zorder=2, color=custom_colors["LightGray"])

plt.show()
