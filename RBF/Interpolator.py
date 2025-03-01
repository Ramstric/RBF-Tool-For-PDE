import sys

import torch
from RBF import RadialBasisFunctions


class Interpolator(object):
    def __init__(self, *r: torch.Tensor, f: torch.Tensor, rbf_name: str, rbf_parameter: float, polynomial_matrix=None):
        self.rbf = getattr(RadialBasisFunctions, rbf_name)
        self.radius = rbf_parameter
        self.function_polynom_matrix = polynomial_matrix

        # Data points, from meshgrid to pairs of coordinates [x, y, z, ...]
        # data_vectors = torch.stack( [ x, y, z, ... ] ).T
        # [x_0 y_0 z_0],
        # [x_1 y_1 z_1],
        # [x_2 y_2 z_2],
        # [... ... ...]
        self.data_vectors = torch.stack([var.ravel() for var in r]).T
        self.num_points = self.data_vectors.shape[0]

        # Distance r = ||x - x_i|| between the data points and the rbf centers
        # distances =
        # [ x_0-x_0, x_0-x_1, x_0-x_2, ...],
        # [ x_1-x_0, x_1-x_1, x_1-x_2, ...],
        # [ x_2-x_0, x_2-x_1, x_2-x_2, ...],
        # [     ...,     ...,     ..., ...]
        self.distances = torch.stack([x - self.data_vectors for x in self.data_vectors])
        self.distances = torch.linalg.norm(self.distances, dim=2)

        # Applying radial basis function
        # phi_matrix =
        # [ rbf(x_0-x_0), rbf(x_0-x_1), rbf(x_0-x_2), ...],
        # [ rbf(x_1-x_0), rbf(x_1-x_1), rbf(x_1-x_2), ...],
        # [ rbf(x_2-x_0), rbf(x_2-x_1), rbf(x_2-x_2), ...],
        # [          ...,          ...,          ..., ...]
        self.phi_matrix = self.rbf(self.distances, self.radius)

        if rbf_name == "polyharmonic_spline":
            # Create polynomial matrix of the form [1, x, y, z, x^2, y^2, z^2, ...]
            # polynomial_matrix =
            # [   1, x_0, y_0, z_0, x_0^2, y_0^2, z_0^2, ...],
            # [   1, x_1, y_1, z_1, x_1^2, y_1^2, z_1^2, ...],
            # [   1, x_2, y_2, z_2, x_2^2, y_2^2, z_2^2, ...],
            # [ ..., ..., ..., ...,   ...,   ...,   ..., ...]
            #self.polynomial_matrix = torch.cat((torch.ones((self.num_points, 1)), self.data_vectors, self.data_vectors**2), dim=1)
            self.polynomial_matrix = self.function_polynom_matrix(self.num_points, variables=[var.unsqueeze(1) for var in self.data_vectors.T])

            # Augment the phi matrix with the polynomial matrix
            # augmented_phi =
            # [     phi_matrix      ,   polynomial_matrix   ],
            # [ polynomial_matrix^T ,           0           ]
            phi_rows = self.num_points
            poly_cols = self.polynomial_matrix.shape[1]

            augmented_phi = torch.zeros((phi_rows + poly_cols, phi_rows + poly_cols))
            augmented_phi[:phi_rows, :phi_rows] = self.phi_matrix
            augmented_phi[:phi_rows, phi_rows:] = self.polynomial_matrix
            augmented_phi[phi_rows:, :phi_rows] = self.polynomial_matrix.T

            # Augment the known_values vector
            # augmented_known_values =
            # [ f ],
            # [ 0 ]
            augmented_known_values = torch.cat((f.ravel(), torch.zeros(poly_cols)))

            # Weights interpolation
            # weights_matrix = [w_0, w_1, w_2, ...]
            self.weights_matrix = torch.linalg.solve(augmented_phi, augmented_known_values)
        else:
            # Weights interpolation
            # weights_matrix = [w_0, w_1, w_2, ...]
            self.weights_matrix = torch.linalg.solve(self.phi_matrix, f.ravel())

    def interpolate(self, *r: torch.Tensor):
        interpol_vectors = torch.stack([arg.ravel() for arg in r]).T
        num_interpol_points = interpol_vectors.shape[0]

        # Distance between the interpolation points and the rbf centers
        interpol_distance = torch.stack([x - self.data_vectors for x in interpol_vectors])
        interpol_distance = torch.linalg.norm(interpol_distance, dim=2)

        # Applying radial basis function
        temp = self.rbf(interpol_distance, self.radius)

        if hasattr(self, "polynomial_matrix"):
            # Separate RBF and polynomial weights
            rbf_weights = self.weights_matrix[:self.num_points]
            polynomial_weights = self.weights_matrix[self.num_points:]

            # Apply RBF weights
            temp = torch.matmul(temp, rbf_weights)

            # Add polynomial terms
            polynomial_values = self.function_polynom_matrix(num_interpol_points, variables=[arg.unsqueeze(1) for arg in interpol_vectors.T])
            polynomial_terms = torch.matmul(polynomial_values, polynomial_weights)

            # Add RBF and polynomial terms
            temp = temp + polynomial_terms
        else:
            # Adding weights
            temp = torch.matmul(temp, self.weights_matrix)

        return torch.reshape(temp, r[0].shape)
