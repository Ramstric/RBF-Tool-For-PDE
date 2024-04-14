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

        if x.size(dim=0) != f.size(dim=0):
            raise Exception("Los datos muestreados deben de tener la misma longitud")

        self.radius = r

        self.phi_matrix = torch.stack([self.method(self.x - rbf_pos, r) for rbf_pos in self.x])
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, self.f)

    def gaussian(self, x: torch.tensor, radius: float):
        return e**(-(4 * x / radius) ** 2)

    def invQuad(self, x: torch.tensor, radius: float):
        return 1 / (1 + (16 * x / radius) ** 2)

    def multiQuad(self, x: torch.tensor, radius: float):
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

#To continue!!!
class RBFInterpolator3D_Alt(object):
    def __init__(self, kernel_str: str, *x: np.array, f: np.array, r: float):
        self.kernel_str = kernel_str
        self.method = getattr(self, kernel_str)

        self.v = x
        self.f_vector = f.ravel()

        self.radius = r

        self.pairs = np.asarray([self.v[0].ravel(), self.v[1].ravel()]).T

        self.phi_matrix = np.array([self.method(np.linalg.norm(self.pairs - pos, axis=1), 10) for pos in self.pairs])
        self.weights_matrix = np.linalg.solve(self.phi_matrix, self.f_vector)

    def gaussian(self, x: float, radius: float):
        return np.e ** (-(4 * x / radius) ** 2)

    def invQuad(self, x: float, radius: float):
        return 1 / (1 + (16 * x / radius) ** 2)

    def multiQuad(self, x: float, radius: float):
        return np.sqrt(1 + (16 * x / radius) ** 2)

    def set_new_data(self, *x: np.array, f: np.array):
        self.v = x
        self.f_vector = f.ravel()
        self.recalculate_weights()

    def recalculate_weights(self):
        self.method = getattr(self, self.kernel_str)

        self.pairs = np.asarray([self.v[0].ravel(), self.v[1].ravel()]).T

        self.phi_matrix = np.array(
            [self.method(np.linalg.norm(self.pairs - rbf_pos, axis=1), 10) for rbf_pos in self.pairs])
        self.weights_matrix = np.linalg.solve(self.phi_matrix, self.f_vector)

    def interpolate(self, x: np.array):
        temp = 0
        for (rbf_pos, w) in zip(self.pairs, self.weights_matrix):
            temp += w * self.method(np.linalg.norm(x - rbf_pos), self.radius)
        return temp


class RBFInterpolator3D(object):
    def __init__(self, kernel_str: str, *x: np.array, f: np.array, r: float):
        self.kernel_str = kernel_str
        self.method = getattr(self, kernel_str)

        self.v = x
        self.f_vector = f.ravel()

        self.radius = r

        self.pairs = np.asarray([self.v[0].ravel(), self.v[1].ravel()]).T

        self.phi_matrix = np.array([self.method(np.linalg.norm(self.pairs - pos, axis=1), 10) for pos in self.pairs])
        self.weights_matrix = np.linalg.solve(self.phi_matrix, self.f_vector)

    def gaussian(self, x: float, radius: float):
        return np.e ** (-(4 * x / radius) ** 2)

    def invQuad(self, x: float, radius: float):
        return 1 / (1 + (16 * x / radius) ** 2)

    def multiQuad(self, x: float, radius: float):
        return np.sqrt(1 + (16 * x / radius) ** 2)

    def set_new_data(self, *x: np.array, f: np.array):
        self.v = x
        self.f_vector = f.ravel()
        self.recalculate_weights()

    def recalculate_weights(self):
        self.method = getattr(self, self.kernel_str)

        self.pairs = np.asarray([self.v[0].ravel(), self.v[1].ravel()]).T

        self.phi_matrix = np.array(
            [self.method(np.linalg.norm(self.pairs - rbf_pos, axis=1), 10) for rbf_pos in self.pairs])
        self.weights_matrix = np.linalg.solve(self.phi_matrix, self.f_vector)

    def interpolate(self, x: np.array):
        temp = 0
        for (rbf_pos, w) in zip(self.pairs, self.weights_matrix):
            temp += w * self.method(np.linalg.norm(x - rbf_pos), self.radius)
        return temp
