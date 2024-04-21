# Ramon Everardo Hernandez Hernandez | 19 april 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

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
    return 0


# --------------------------------[ Datos muestreados ]--------------------------------
x_B = torch.tensor([-2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2,
                    -2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2,
                    -2, -2, -2, -2, -2, -2, -2, -2,
                    2, 2, 2, 2, 2, 2, 2, 2], device=device)      # Boundary nodes
x_I = torch.arange(-1.6, 2, 0.4, device=device)      # Interior nodes

x_0 = torch.cat((x_B, x_I), dim=0)                 # Making the Vector X (Boundary + Interior nodes)

y_B = torch.tensor([-2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6,
                    -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6], device=device)  # Boundary nodes
y_I = torch.arange(-1.6, 2, 0.4, device=device)      # Interior nodes

y_0 = torch.cat((x_B, x_I), dim=0)                 # Making the Vector X (Boundary + Interior nodes)


#z_B = torch.tensor([2, 0.0125342863113], device=device)    # Boundary conditions
#z_I = torch.zeros(x_I.size(), device=device)                    # Evaluation PDE function on the Interior nodes

z_0 = torch.cat((z_B, z_I), dim=0)                 # Making the Vector Y (Boundary conditions + Function values)

interpolator = RBF.InterpolatorPDE("multiQuad", x_0, f=y_0, r=1.49771, boundary=x_B)

# --------------------------------[ Interpolación ]--------------------------------
x_1 = torch.linspace(0, 90, 150, device=device)

y_RBF = interpolator.interpolate(x_1)

#print(interpolator.interpolate(torch.tensor([3.99522861], device=device)))

# --------------------------------[ Plotting ]--------------------------------
fig, ax = plt.subplots(1, 1)
plt.grid()

plot = plt.plot(x_1.cpu().detach().numpy(), y_RBF.cpu().detach().numpy())
#plt.scatter(x_0.cpu().detach().numpy(), y_0.cpu().detach().numpy(), s=8, zorder=2, color=custom_colors["LightGray"])

#plt.xlim(0, 2.5)
#plt.ylim(-3, 6)
plt.show()
