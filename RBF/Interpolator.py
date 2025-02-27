import torch
from RBF import RadialBasisFunctions


class Interpolator(object):
    def __init__(self, *r: torch.Tensor, f: torch.Tensor, radius: float, rbf_name: str):
        self.rbf = getattr(RadialBasisFunctions, rbf_name)
        self.num_points = r[0].shape[0]
        self.radius = radius

        # Data points, from meshgrid to pairs of coordinates [x, y, z, ...]
        self.data_vectors = torch.stack([var.ravel() for var in r]).T

        # Distance r = ||x - x_i|| between the data points and the rbf centers
        self.distances = torch.stack([x - self.data_vectors for x in self.data_vectors])
        self.distances = torch.linalg.norm(self.distances, dim=2)

        # Applying radial basis function
        self.phi_matrix = self.rbf(self.distances, self.radius)

        # Create polynomial matrix of the form [1, x, y, z, x^2, y^2, z^2, ...]
        #polynomial_matrix = torch.cat((torch.ones((self.num_points, 1)), self.data_vectors, self.data_vectors**2), dim=1)
        polynomial_matrix = torch.cat((torch.ones((self.num_points, 1)), self.data_vectors, self.data_vectors**2), dim=1)

        # Augment the phi matrix
        phi_rows = self.num_points
        poly_cols = polynomial_matrix.shape[1]

        augmented_phi = torch.zeros((phi_rows + poly_cols, phi_rows + poly_cols))
        augmented_phi[:phi_rows, :phi_rows] = self.phi_matrix
        augmented_phi[:phi_rows, phi_rows:] = polynomial_matrix
        augmented_phi[phi_rows:, :phi_rows] = polynomial_matrix.T

        # Augment the known_values vector
        augmented_known_values = torch.cat((f.ravel(), torch.zeros(poly_cols)))

        # Weights interpolation
        self.weights_matrix = torch.linalg.solve(augmented_phi, augmented_known_values)

        # Weights interpolation
        #self.weights_matrix = torch.linalg.solve(self.phi_matrix, f.ravel())
        self.weights_matrix = torch.linalg.solve(augmented_phi, augmented_known_values)

    def interpolate(self, *r: torch.Tensor):
        interpol_vectors = torch.stack([arg.ravel() for arg in r]).T
        num_interpol_points = interpol_vectors.shape[0]

        # Distance between the interpolation points and the rbf centers
        interpol_distance = torch.stack([x - self.data_vectors for x in interpol_vectors])
        interpol_distance = torch.linalg.norm(interpol_distance, dim=2)

        # Applying radial basis function
        temp = self.rbf(interpol_distance, self.radius)

        # Adding weights
        #temp = torch.matmul(temp, self.weights_matrix)

        #return torch.reshape(temp, r[0].shape)

        # Separate RBF and polynomial weights
        rbf_weights = self.weights_matrix[:self.num_points]
        polynomial_weights = self.weights_matrix[self.num_points:]

        # Apply RBF weights
        temp = torch.matmul(temp, rbf_weights)

        # Add polynomial terms
        polynomial_values = torch.cat((torch.ones((num_interpol_points, 1)), interpol_vectors, interpol_vectors**2), dim=1)
        polynomial_terms = torch.matmul(polynomial_values, polynomial_weights)

        # Add RBF and polynomial terms
        temp = temp + polynomial_terms

        return torch.reshape(temp, r[0].shape)
