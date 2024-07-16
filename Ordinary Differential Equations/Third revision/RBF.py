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
    def __init__(self, kernel_str: str, *x: torch.tensor, f: torch.tensor, r: float, boundary: torch.tensor, inner: torch.tensor, ode: str):
        self.kernel_str = kernel_str
        self.method = getattr(self, kernel_str)
        self.ode = ode

        self.radius = r

        self.x = x
        self.pairs = torch.stack([arg.ravel() for arg in x]).T
        self.boundary = boundary
        self.inner = inner

        self.f_vector = f.ravel()

        boundary_matrix = torch.stack([self.method(torch.linalg.vector_norm(pos - self.pairs, dim=1), self.radius)
                                       for pos in self.boundary])
        internal_matrix = torch.stack([self.derivative_operator(ode, pos - self.pairs, self.radius)
                                       for pos in self.inner])
        internal_matrix = torch.squeeze(internal_matrix)  # Why does it have to be negative?

        self.phi_matrix = torch.cat((boundary_matrix, internal_matrix))

        print(self.phi_matrix.shape)
        print(self.f_vector.shape)
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, self.f_vector)

    @staticmethod
    def gaussian(x: torch.tensor, radius: float):
        return e ** (-(radius * x)**2)

    def gaussianDerivative(self, x: torch.tensor, radius: float):
        return (-2*x*radius**2) * self.gaussian(x, radius)

    def gaussianSecondDerivative(self, x: torch.tensor, radius: float):
        return (-2*radius**2)*(-2*(radius*x)**2 * self.gaussian(x, radius) + self.gaussian(x, radius))

    @staticmethod
    def multiQuad(x: torch.tensor, radius: float):
        return torch.sqrt(1 + (radius * x)**2)

    @staticmethod
    def multiQuadDerivative(x: torch.tensor, radius: float):
        return (x * radius**2)/torch.sqrt(1 + (radius * x)**2)

    @staticmethod
    def multiQuadSecondDerivative(x: torch.tensor, radius: float):
        return radius**2/((1 + (radius * x)**2)*torch.sqrt(1 + (radius * x)**2))

    def derivative_operator(self, operation: str, x: torch.tensor, radius: float):
        chars = operation.split()
        operators = chars[1::2]

        methodDerivative = getattr(self, self.kernel_str + "Derivative")
        methodSecondDerivative = getattr(self, self.kernel_str + "SecondDerivative")

        # Operators
        op = {'+': lambda a, b: a + b, '-': lambda a, b: a - b}
        val = {"y": self.method(x, radius), "y'": methodDerivative(x, radius), "y''": methodSecondDerivative(x, radius)}

        temp = val[chars[0]]
        temp_op = ''
        for char in chars[1:]:
            if char in operators:
                temp_op = char
                continue
            else:
                temp = op[temp_op](temp, val[char])

        return temp

    def recalculate_weights(self, r: float):
        self.method = getattr(self, self.kernel_str)

        self.radius = r

        self.pairs = torch.stack([arg.ravel() for arg in self.x]).T

        boundary_matrix = torch.stack([self.method(torch.linalg.vector_norm(pos - self.pairs, dim=1), self.radius)
                                       for pos in self.pairs if pos in self.boundary])
        internal_matrix = torch.stack([self.derivative_operator(self.ode, pos - self.pairs, self.radius)
                                       for pos in self.pairs if pos not in self.boundary])
        internal_matrix = torch.squeeze(internal_matrix)  # Why does it have to be negative?

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
