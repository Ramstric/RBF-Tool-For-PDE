import torch
from RBF import RadialBasisFunctions


class DifferentialInterpolator(object):
    def __init__(self, *r: torch.Tensor, boundary: torch.Tensor, inner: torch.Tensor, f: torch.Tensor, radius: float, rbf_name: str, derivative_operator):
        self.rbf = getattr(RadialBasisFunctions, rbf_name)

        self.radius = radius

        # Data points, from meshgrid to pairs of coordinates [x, y, z, ...]
        self.center_points = torch.stack([var.ravel() for var in r]).T

        self.boundary = boundary
        self.inner = inner

        # Distance r = ||x - x_i|| between the data points and the rbf centers
        self.boundary_distances = torch.stack([x - self.center_points for x in boundary])
        self.boundary_distances = torch.linalg.norm(self.boundary_distances, dim=2)

        self.inner_distances = torch.stack([x - self.center_points for x in inner])
        #self.inner_distances = torch.linalg.norm(self.inner_distances, dim=2)
        self.inner_distances = torch.squeeze(self.inner_distances)

        boundary_matrix = self.rbf(self.boundary_distances, self.radius)
        internal_matrix = derivative_operator(self.inner_distances, self.radius)

        self.phi_matrix = torch.cat((boundary_matrix, internal_matrix))

        # Weights interpolation
        self.weights_matrix = torch.linalg.solve(self.phi_matrix, f.ravel())

    def interpolate(self, *r: torch.Tensor):
        interpol_vectors = torch.stack([arg.ravel() for arg in r]).T

        # Distance between the interpolation points and the rbf centers
        interpol_distance = torch.stack([x - self.center_points for x in interpol_vectors])
        interpol_distance = torch.linalg.norm(interpol_distance, dim=2)

        # Applying radial basis function
        temp = self.rbf(interpol_distance, self.radius)

        # Adding weights
        temp = torch.matmul(temp, self.weights_matrix)

        return torch.reshape(temp, r[0].shape)
