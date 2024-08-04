import torch


# Gaussian RBF
def gaussian(r: torch.Tensor, radius: float):
    return torch.exp(- r**2 / (2 * radius**2))


def gaussian_derivative(r, radius):
    return - r * gaussian(r, radius) / (radius**2)


# Partial Derivatives
def gaussian_t(r: torch.Tensor, t: torch.Tensor, radius: float):
    return - t * gaussian(r, radius) / (radius**2)


def gaussian_xx(r: torch.Tensor, x: torch.Tensor, radius: float):
    return (x**2 - radius**2) / (radius**4 * torch.exp(r**2 / (2 * radius**2)))


# Multiquadric RBF
def multiquadric(r: torch.Tensor, radius: float):
    return torch.sqrt(r**2 + radius**2)


def multiquadric_derivative(r: torch.Tensor, radius: float):
    return r / multiquadric(r, radius)


def multiquadric_second_derivative(r: torch.Tensor, radius: float):
    return radius**2 / (multiquadric(r, radius) * (r**2 + radius**2))


# Partial Derivatives
def multiquadric_t(r: torch.Tensor, t: torch.Tensor, radius: float):
    return t / multiquadric(r, radius)


def multiquadric_tt(r: torch.Tensor, x: torch.Tensor, radius: float):
    return (radius**2 + x**2) / (multiquadric(r, radius) * (r**2 + radius**2))


def multiquadric_xx(r: torch.Tensor, t: torch.Tensor, radius: float):
    return (radius**2 + t**2) / (multiquadric(r, radius) * (r**2 + radius**2))

