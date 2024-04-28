# Ramon Everardo Hernandez Hernandez | 7 april 2024
#        Radial Basis Function Interpolation
#             for Non-Linear Data

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
    def __init__(self, kernel_str: str, boundary: torch.tensor, inner: torch.tensor, all: torch.tensor, f: torch.tensor, r: float, pde: str):
        self.kernel_str = kernel_str
        self.method = getattr(self, kernel_str)
        self.ode = pde

        self.radius = r

        self.pairs = all
        self.boundary = boundary
        self.inner = inner

        self.f_vector = f

        boundary_matrix = torch.stack([self.method(torch.linalg.vector_norm(pos - self.pairs, dim=1), self.radius)
                                       for pos in self.boundary])

        internal_matrix = torch.stack([self.derivative_operator(pde, torch.linalg.vector_norm(pos - self.pairs, dim=1),
                                       self.radius, x=pos[0] - torch.stack([pos[0] for pos in self.pairs]),
                                       y=pos[1] - torch.stack([pos[1] for pos in self.pairs])) for pos in self.inner])

        internal_matrix = torch.squeeze(internal_matrix)

        self.phi_matrix = torch.cat((boundary_matrix, internal_matrix))

        self.weights_matrix = torch.linalg.solve(self.phi_matrix, self.f_vector)

    def derivative_operator(self, operation: str, s: torch.tensor, radius: float, x: torch.tensor, y: torch.tensor):

        chars = operation.split()
        operators = chars[1::2]

        # Math operators dictionary
        op = {'+': lambda a, b: a + b, '-': lambda a, b: a - b}

        # Values for function and derivatives
        method = {"f": self.method(s, radius), "f_x": self.gaussianDerX(s, x, radius),
                  "f_y": self.gaussianDerY(s, y, radius), "f_xx": self.gaussianDerXX(s, x, radius),
                  "f_yy": self.gaussianDerYY(s, y, radius)}

        temp = method[chars[0]]  # Initial value before parsing
        temp_op = ''

        for char in chars[1:]:
            if char in operators:
                temp_op = char  # Set math operator
                continue
            else:
                temp = op[temp_op](temp, method[char])  # Evaluate operation between saved value in temp and new value

        return temp

    def recalculate_weights(self, r: float, pde: str):
        self.method = getattr(self, self.kernel_str)

        self.radius = r

        boundary_matrix = torch.stack([self.method(torch.linalg.vector_norm(pos - self.pairs, dim=1), self.radius)
                                       for pos in self.boundary])

        internal_matrix = torch.stack([self.derivative_operator(pde, torch.linalg.vector_norm(pos - self.pairs, dim=1),
                                       self.radius, x=pos[0] - torch.stack([pos[0] for pos in self.pairs]),
                                       y=pos[1] - torch.stack([pos[1] for pos in self.pairs])) for pos in self.inner])

        internal_matrix = torch.squeeze(internal_matrix)

        self.phi_matrix = torch.cat((boundary_matrix, internal_matrix))

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

    @staticmethod
    def gaussian(x: torch.tensor, radius: float):
        return e ** (-(radius * x) ** 2)

    def gaussianDerX(self, s: torch.tensor, x: torch.tensor, radius: float):
        return (-2 * x * radius ** 2) * self.gaussian(s, radius)

    def gaussianDerXX(self, s: torch.tensor, x: torch.tensor, radius: float):
        return (-2 * radius ** 2) * ((-2 * (x * radius) ** 2) * self.gaussian(s, radius) + self.gaussian(s, radius))

    def gaussianDerY(self, s: torch.tensor, y: torch.tensor, radius: float):
        return (-2 * y * radius ** 2) * self.gaussian(s, radius)

    def gaussianDerYY(self, s: torch.tensor, y: torch.tensor, radius: float):
        return (-2 * radius ** 2) * ((-2 * (y * radius) ** 2) * self.gaussian(s, radius) + self.gaussian(s, radius))

    @staticmethod
    def multiQuad(x: torch.tensor, radius: float):
        return torch.sqrt(1 + (radius * x) ** 2)

    def multiQuadDerX(self, s: torch.tensor, x: torch.tensor, radius: float):
        return (x * radius ** 2) / self.multiQuad(s, radius)

    def multiQuadDerXX(self, s: torch.tensor, y: torch.tensor, radius: float):
        return (radius ** 4 * y ** 2 + 1) / self.multiQuad(s, radius) ** 3

    def multiQuadDerY(self, s: torch.tensor, y: torch.tensor, radius: float):
        return (y * radius ** 2) / self.multiQuad(s, radius)

    def multiQuadDerYY(self, s: torch.tensor, x: torch.tensor, radius: float):
        return (radius ** 4 * x ** 2 + 1) / self.multiQuad(s, radius) ** 3