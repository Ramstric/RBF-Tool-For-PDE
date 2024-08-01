import torch
from math import e
import RadialBasisFunctions as RadialBasisFunction
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                                       y=pos[1] - torch.stack([pos[1] for pos in self.pairs]),
                                       t=pos[2] - torch.stack([pos[2] for pos in self.pairs])) for pos in self.inner])

        internal_matrix = torch.squeeze(internal_matrix)

        self.phi_matrix = torch.cat((boundary_matrix, internal_matrix))

        self.weights_matrix = torch.linalg.solve(self.phi_matrix, self.f_vector)

    def derivative_operator(self, operation: str, s: torch.tensor, radius: float, x: torch.tensor, y: torch.tensor, t: torch.tensor):

        return RadialBasisFunction.gaussian_t(s, t, radius) - 2*(RadialBasisFunction.gaussian_xx(s, x, radius) + RadialBasisFunction.gaussian_yy(s, y, radius))

        # Math operators dictionary
        op = {'+': lambda a, b: a + b, '-': lambda a, b: a - b}

        # Values for function and derivatives
        method = {"f": self.method(s, radius),
                  "f_t": getattr(RadialBasisFunction, self.rbf_name + "_t")(s, t, radius),
                  "f_xx": getattr(RadialBasisFunction, self.rbf_name + "_xx")(s, x, radius),
                  "f_yy": getattr(RadialBasisFunction, self.rbf_name + "_yy")(s, y, radius)}

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
