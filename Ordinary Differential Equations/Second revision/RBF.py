# Ramon Everardo Hernandez Hernandez | 7 april 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import numpy as np
import torch
from math import e

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Interpolator(object):
    def __init__(self, kernel_str: str, *x: torch.tensor, f: torch.tensor, r: float):
        self.kernel_str = kernel_str
        self.method = getattr(self, kernel_str)

        self.v = x
        self.f_vector = f.ravel()

        self.radius = r

        self.pairs = torch.stack([arg.ravel() for arg in x]).T

        self.phi_matrix = torch.stack(
            [self.method(torch.linalg.vector_norm(self.pairs - pos, dim=1), self.radius) for pos in self.pairs])
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, self.f_vector)

    @staticmethod
    def gaussian(x: torch.tensor, radius: float):
        return e ** (-(4 * x / radius) ** 2)

    @staticmethod
    def invQuad(x: torch.tensor, radius: float):
        return 1 / (1 + (16 * x / radius) ** 2)

    @staticmethod
    def multiQuad(x: torch.tensor, radius: float):
        return torch.sqrt(1 + (16 * x / radius) ** 2)

    def recalculate_weights(self):
        self.method = getattr(self, self.kernel_str)

        self.pairs = torch.stack([self.v[0].ravel(), self.v[1].ravel()]).T

        self.phi_matrix = torch.stack(
            [self.method(torch.linalg.vector_norm(self.pairs - pos, dim=1), self.radius) for pos in self.pairs])
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, self.f_vector)

    def interpolate(self, *x: torch.tensor):
        self.method = getattr(self, self.kernel_str)

        data_pairs = torch.stack([arg.ravel() for arg in x]).T

        # Distance between the data points and the interpolation points
        temp = torch.stack([torch.linalg.vector_norm(torch.sub(data_pairs, pos), dim=1) for pos in self.pairs])

        # Applying kernel function
        temp = self.method(temp, self.radius)

        # Adding weights
        temp = torch.matmul(temp.T, self.weights_matrix)

        return torch.reshape(temp, x[0].shape)


class InterpolatorPDE(object):
    def __init__(self, kernel_str: str, *x: torch.tensor, f: torch.tensor, r: float, boundary: torch.tensor):
        self.kernel_str = kernel_str
        self.method = getattr(self, kernel_str)

        self.v = x
        self.f_vector = f.ravel()

        self.radius = r

        self.pairs = torch.stack([arg.ravel() for arg in x]).T

        A = torch.stack([self.method(torch.linalg.vector_norm(self.pairs - pos, dim=1), self.radius) for pos in self.pairs if pos in boundary])
        B = torch.stack([self.derivative_operator(pos - self.pairs, self.radius) for pos in self.pairs if pos not in boundary])
        B = torch.squeeze(B)  # Why does it have to be negative?

        self.phi_matrix = torch.cat((A, B))

        #self.phi_matrix = torch.stack(
        #    [self.method(torch.linalg.vector_norm(self.pairs - pos, dim=1), self.radius) for pos in self.pairs])
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, self.f_vector)

    @staticmethod
    def gaussian(x: torch.tensor, radius: float):
        return e ** (-(4 * x / radius) ** 2)

    @staticmethod
    def invQuad(x: torch.tensor, radius: float):
        return 1 / (1 + (16 * x / radius) ** 2)

    @staticmethod
    def multiQuad(x: torch.tensor, radius: float):
        return torch.sqrt(1 + (x / radius) ** 2)

    @staticmethod
    def invQuadDerivative(x: torch.tensor, radius: float):
        return -32 * x / (1 + (16 * x / radius) ** 2) ** 2

    @staticmethod
    def multiQuadDerivative(x: torch.tensor, radius: float):
        return ((1 / radius) ** 2 * x) / torch.sqrt(1 + (x / radius) ** 2)

    @staticmethod
    def multiQuadSecondDerivative(x: torch.tensor, radius: float):
        return (1 / radius) ** 2 * (1 - (x / radius) ** 2) / (1 + (x / radius) ** 2) ** (3/2)

    def derivative_operator(self, x: torch.tensor, radius: float):
        methodDerivative = getattr(self, self.kernel_str+"Derivative")
        return 2 * methodDerivative(x, radius) - self.method(x, radius)

    def recalculate_weights(self):
        self.method = getattr(self, self.kernel_str)

        self.pairs = torch.stack([arg.ravel() for arg in self.v]).T

        self.phi_matrix = torch.stack(
            [self.method(torch.linalg.vector_norm(self.pairs - pos, dim=1), self.radius) for pos in self.pairs])
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, self.f_vector)

    def interpolate(self, *x: torch.tensor):
        self.method = getattr(self, self.kernel_str)

        data_pairs = torch.stack([arg.ravel() for arg in x]).T

        # Distance between the data points and the interpolation points
        temp = torch.stack([torch.linalg.vector_norm(torch.sub(data_pairs, pos), dim=1) for pos in self.pairs])

        # Applying kernel function
        temp = self.method(temp, self.radius)

        # Adding weights
        temp = torch.matmul(temp.T, self.weights_matrix)

        return torch.reshape(temp, x[0].shape)
