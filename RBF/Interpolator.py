import torch
from RBF import RadialBasisFunctions


class Interpolator(object):
    def __init__(self, *r: torch.Tensor, f: torch.Tensor, radius: float, rbf_name: str):
        self.rbf = getattr(RadialBasisFunctions, rbf_name)

        self.radius = radius

        # Data points, from meshgrid to pairs of coordinates [x, y, z, ...]
        self.data_vectors = torch.stack([var.ravel() for var in r]).T

        # Distance r = ||x - x_i|| between the data points and the rbf centers
        self.distances = torch.stack([x - self.data_vectors for x in self.data_vectors])
        self.distances = torch.linalg.norm(self.distances, dim=2)

        # Applying radial basis function
        self.phi_matrix = self.rbf(self.distances, self.radius)

        # Weights interpolation
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, f.ravel())

    def interpolate(self, *r: torch.Tensor):
        interpol_vectors = torch.stack([arg.ravel() for arg in r]).T

        # Distance between the interpolation points and the rbf centers
        interpol_distance = torch.stack([x - self.data_vectors for x in interpol_vectors])
        interpol_distance = torch.linalg.norm(interpol_distance, dim=2)

        # Applying radial basis function
        temp = self.rbf(interpol_distance, self.radius)

        # Adding weights
        temp = torch.matmul(temp, self.weights_matrix)

        return torch.reshape(temp, r[0].shape)
