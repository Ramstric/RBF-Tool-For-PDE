import numpy as np


# Gaussian RBF
def gaussian(r, radius: float):
    return np.exp(- r**2 / (2 * radius**2))


def gaussian_derivative(r, radius):
    return - r * gaussian(r, radius) / (radius**2)


# Partial Derivatives
def gaussian_t(r, t, radius: float):
    return - t * gaussian(r, radius) / radius**2


def gaussian_xx(r, x, radius: float):
    return (x**2 - radius**2) / (radius**4 * gaussian(r, radius))


# Multiquadric RBF
def multiquadric(r, radius: float):
    return np.sqrt(r**2 + radius**2)


def multiquadric_derivative(r, radius: float):
    return r / multiquadric(r, radius)


def multiquadric_second_derivative(r, radius: float):
    return radius**2 / (multiquadric(r, radius) * (r**2 + radius**2))


# Partial Derivatives
def multiquadric_t(r, t, radius: float):
    return t / multiquadric(r, radius)


def multiquadric_xx(r, t, radius: float):
    return (radius**2 + t**2) / (multiquadric(r, radius) * (r**2 + radius**2))

