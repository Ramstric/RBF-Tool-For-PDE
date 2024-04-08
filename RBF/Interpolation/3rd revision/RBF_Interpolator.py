# Ramon Everardo Hernandez Hernandez | 6 april 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import numpy as np


class RBFInterpolator(object):
    def __init__(self, kernel_str, x: np.array, f: np.array, r: float):
        self.kernel_str = kernel_str
        self.method = getattr(self, kernel_str)

        self.x = x
        self.f = f

        if x.size != f.size:
            raise Exception("Los datos muestreados deben de tener la misma longitud")

        self.radius = r

        self.phi_matrix = np.array([self.method(self.x - rbf_pos, self.radius) for rbf_pos in self.x])
        self.f_matrix = np.vstack(self.f)
        self.weights_matrix = np.linalg.solve(self.phi_matrix, self.f_matrix)

    def gaussian(self, x: float, radius: float):
        return np.e ** (-(4 * x / radius) ** 2)

    def invQuad(self, x: float, radius: float):
        return 1 / (1 + (16 * x / radius) ** 2)

    def multiQuad(self, x: float, radius: float):
        return np.sqrt(1 + (16 * x / radius) ** 2)

    def recalculate_weights(self):
        self.method = getattr(self, self.kernel_str)
        self.phi_matrix = np.array([self.method(self.x - rbf_pos, self.radius) for rbf_pos in self.x])
        self.f_matrix = np.vstack(self.f)
        self.weights_matrix = np.linalg.solve(self.phi_matrix, self.f_matrix)

    def interpolate(self, x: float):
        temp = 0
        for (rbf_pos, w) in zip(self.x, self.weights_matrix):
            temp += w * self.method(x - rbf_pos, self.radius)
        return temp
