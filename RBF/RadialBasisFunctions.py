import torch


# Gaussian RBF
def gaussian(r: torch.Tensor, radius: float):
    return torch.exp(- r**2 / (2 * radius**2))


def gaussian_derivative(r, radius):
    return - r * gaussian(r, radius) / (radius**2)


# Partial Derivatives

def gaussian_t(s: torch.Tensor, t: torch.Tensor, radius: float):
    return (-2 * t * radius ** 2) * gaussian(s, radius)


def gaussian_xx(s: torch.Tensor, x: torch.Tensor, radius: float):
    return (-2 * radius ** 2) * ((-2 * (x * radius) ** 2) * gaussian(s, radius) + gaussian(s, radius))


def gaussian_yy(s: torch.Tensor, y: torch.Tensor, radius: float):
    return (-2 * radius ** 2) * ((-2 * (y * radius) ** 2) * gaussian(s, radius) + gaussian(s, radius))


# Multiquadric RBF
def multiquadric(r: torch.Tensor, radius: float):
    return torch.sqrt(r**2 + radius**2)


def multiquadric_derivative(r: torch.Tensor, radius: float):
    return r / multiquadric(r, radius)


def multiquadric_second_derivative(r: torch.Tensor, radius: float):
    return radius**2 / (multiquadric(r, radius) * (r**2 + radius**2))


# Partial Derivatives
def multiquadric_x(s: torch.Tensor, x: torch.Tensor, radius: float):
    return (x * radius ** 2) / multiquadric(s, radius)


def multiquadric_xx(s: torch.Tensor, y: torch.Tensor, radius: float):
    return (radius ** 4 * y ** 2 + 1) / multiquadric(s, radius) ** 3


def multiquadric_y(s: torch.Tensor, y: torch.Tensor, radius: float):
    return (y * radius ** 2) / multiquadric(s, radius)


def multiquadric_yy(s: torch.Tensor, x: torch.Tensor, radius: float):
    return (radius ** 4 * x ** 2 + 1) / multiquadric(s, radius) ** 3
