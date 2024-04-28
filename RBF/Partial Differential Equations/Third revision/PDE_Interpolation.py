import torch
from math import e
import RadialBasisFunctions as RadialBasisFunction
import re

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
    def __init__(self, kernel_str: str, boundary: torch.tensor, inner: torch.tensor, all_pairs: torch.tensor, f: torch.tensor, r: float, pde: str):
        self.kernel_str = kernel_str
        self.method = getattr(RadialBasisFunction, kernel_str)
        self.ode = pde

        self.radius = r

        self.pairs = all_pairs
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

        # Math operators dictionary
        op = {'+': lambda a, b: a + b, '-': lambda a, b: a - b}

        # Values for function and derivatives
        method = {"f": self.method(s, radius),
                  "f_x": getattr(RadialBasisFunction, self.kernel_str+"_x")(s, x, radius),
                  "f_y": getattr(RadialBasisFunction, self.kernel_str+"_y")(s, y, radius),
                  "f_xx": getattr(RadialBasisFunction, self.kernel_str+"_xx")(s, x, radius),
                  "f_yy": getattr(RadialBasisFunction, self.kernel_str+"_yy")(s, y, radius)}

        variables = re.findall(r"\b(\d?\.?\d+)*(f[_x|y]*?)\b", operation)
        operators = re.findall(r"[+\-*/]", operation)

        # Coefficients into floats
        for variable in variables:
            variables[variables.index(variable)] = (1, variable[1]) if variable[0] == '' else (float(variable[0]), variable[1])

        temp = variables[0][0] * method[variables[0][1]]  # Initial value before parsing: coefficient * function

        for operator, variable in zip(operators, variables[1:]):
            temp = op[operator](temp, variable[0] * method[variable[1]])  # Evaluate operation between saved value in temp and new value

        return temp

    def recalculate_weights(self, r: float, pde: str):
        self.method = getattr(RadialBasisFunction, self.kernel_str)

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
        self.method = getattr(RadialBasisFunction, self.kernel_str)

        data_pairs = torch.stack([arg.ravel() for arg in x]).T

        # Distance between the data points and the interpolation points
        temp = torch.stack([torch.linalg.vector_norm(torch.sub(data_pairs, pos), dim=1) for pos in self.pairs])

        # Applying kernel function
        temp = self.method(temp, self.radius)

        # Adding weights
        temp = torch.matmul(temp.T, self.weights_matrix)

        return torch.reshape(temp, x[0].shape)
