import torch
from math import e


def gaussian(x: torch.tensor, radius: float):
    return e ** (-(radius * x) ** 2)


def gaussian_x(s: torch.tensor, x: torch.tensor, radius: float):
    return (-2 * x * radius ** 2) * gaussian(s, radius)


def gaussian_xx(s: torch.tensor, x: torch.tensor, radius: float):
    return (-2 * radius ** 2) * ((-2 * (x * radius) ** 2) * gaussian(s, radius) + gaussian(s, radius))


def gaussian_y(s: torch.tensor, y: torch.tensor, radius: float):
    return (-2 * y * radius ** 2) * gaussian(s, radius)


def gaussian_yy(s: torch.tensor, y: torch.tensor, radius: float):
    return (-2 * radius ** 2) * ((-2 * (y * radius) ** 2) * gaussian(s, radius) + gaussian(s, radius))


def multiQuad(x: torch.tensor, radius: float):
    return torch.sqrt(1 + (radius * x) ** 2)


def multiQuad_x(s: torch.tensor, x: torch.tensor, radius: float):
    return (x * radius ** 2) / multiQuad(s, radius)


def multiQuad_xx(s: torch.tensor, y: torch.tensor, radius: float):
    return (radius ** 4 * y ** 2 + 1) / multiQuad(s, radius) ** 3


def multiQuad_y(s: torch.tensor, y: torch.tensor, radius: float):
    return (y * radius ** 2) / multiQuad(s, radius)


def multiQuad_yy(s: torch.tensor, x: torch.tensor, radius: float):
    return (radius ** 4 * x ** 2 + 1) / multiQuad(s, radius) ** 3
