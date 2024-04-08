# Ramon Everardo Hernandez Hernandez | 6 april 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

import numpy as np


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
