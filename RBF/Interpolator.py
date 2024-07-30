import torch
from RBF import RadialBasisFunctions as rbf


class Interpolator(object):
    def __init__(self, *r: torch.Tensor, f: torch.Tensor, radius: float, rbf_name: str):
        self.method = getattr(rbf, rbf_name)

        self.radius = radius

        self.data_vectors = torch.stack([arg.ravel() for arg in r]).T

        # Distance r = ||x - x_i|| between the data points and the interpolation points
        self.distances = torch.stack([self.data_vectors - center for center in self.data_vectors])
        self.distances = torch.linalg.norm(self.distances, dim=2)

        # Applying kernel function
        self.phi_matrix = self.method(self.distances, self.radius)

        # Weights interpolation
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, f.ravel())

    def interpolate(self, *r: torch.Tensor):
        interpol_vectors = torch.stack([arg.ravel() for arg in r]).T

        # Distance between the data points and the interpolation points
        #interpol_distance = torch.stack([torch.linalg.vector_norm(torch.sub(interpol_vectors, pos), dim=1) for pos in self.data_vectors])
        interpol_distance = torch.stack([interpol_vectors - center for center in self.data_vectors])
        interpol_distance = torch.linalg.norm(interpol_distance, dim=2)

        # Applying kernel function
        temp = self.method(interpol_distance, self.radius)

        # Adding weights
        temp = torch.matmul(temp.T, self.weights_matrix)

        return torch.reshape(temp, r[0].shape)
