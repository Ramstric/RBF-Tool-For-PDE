# Ramon Everardo Hernandez Hernandez | 7 april 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import numpy as np
import torch
from math import e

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RBFInterpolator(object):
    def __init__(self, kernel_str, x: torch.tensor, f: torch.tensor, r: float):
        self.kernel_str = kernel_str
        self.method = getattr(self, kernel_str)

        self.x = x
        self.f = f

        #if x.size(dim=0) != f.size(dim=0):
        #    raise Exception("Los datos muestreados deben de tener la misma longitud")

        self.radius = r

        self.phi_matrix = torch.stack([self.method(self.x - rbf_pos, r) for rbf_pos in self.x])
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, self.f)

    @staticmethod
    def gaussian(x: torch.tensor, radius: float):
        return e**(-(4 * x / radius) ** 2)

    @staticmethod
    def invQuad(x: torch.tensor, radius: float):
        return 1 / (1 + (16 * x / radius) ** 2)

    @staticmethod
    def multiQuad(x: torch.tensor, radius: float):
        return torch.sqrt(1 + (16 * x / radius) ** 2)

    def recalculate_weights(self):
        self.method = getattr(self, self.kernel_str)
        self.phi_matrix = torch.stack([self.method(self.x - rbf_pos, self.radius) for rbf_pos in self.x])
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, self.f)

    def interpolate(self, x: float):
        temp = 0
        for (rbf_pos, w) in zip(self.x, self.weights_matrix):
            temp += w * self.method(x - rbf_pos, self.radius)
        return temp


class RBFInterpolator3D(object):
    def __init__(self, kernel_str: str, *x: torch.tensor, f: torch.tensor, r: float):
        self.kernel_str = kernel_str
        self.method = getattr(self, kernel_str)

        self.v = x
        self.f_vector = f.ravel()

        self.radius = r

        self.pairs = torch.stack([self.v[0].ravel(), self.v[1].ravel()]).T

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

        data_pairs = torch.stack([x[0].ravel(), x[1].ravel()]).T

        # Distance between the data points and the interpolation points
        temp = torch.stack([torch.linalg.vector_norm(torch.sub(data_pairs, pos), dim=1) for pos in self.pairs])

        # Applying kernel function
        temp = self.method(temp, self.radius)

        # Adding weights
        temp = torch.matmul(temp.T, self.weights_matrix)

        return torch.reshape(temp, x[0].shape)
