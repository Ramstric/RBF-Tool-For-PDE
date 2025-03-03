import torch
from RBF import RadialBasisFunctions


class DifferentialInterpolator(object):
    def __init__(self, boundary: list[torch.Tensor], inner: list[torch.Tensor], f: torch.Tensor, rbf_name: str, rbf_parameter: float, polynomial_matrix=None, derivative_operator=None):
        self.rbf = getattr(RadialBasisFunctions, rbf_name)

        self.radius = radius

        self.boundary = torch.stack([var.ravel() for var in boundary]).T
        self.inner = torch.stack([var.ravel() for var in inner]).T

        # Data points, from meshgrid to pairs of coordinates [x, y, z, ...]
        self.center_points = torch.cat((self.boundary, self.inner))

        # Distance r = ||(x, y, z, ...) - (x_i, y_i, z_i, ...)||
        # between the boundary/inner points and the rbf centers (all points)
        self.boundary_vectors = torch.stack([x - self.center_points for x in self.boundary])
        self.boundary_distances = torch.linalg.norm(self.boundary_vectors, dim=2)

        boundary_matrix = self.rbf(self.boundary_distances, self.radius)

        self.inner_vectors = torch.stack([x - self.center_points for x in self.inner])

        if len(boundary) and len(inner) == 1:
            self.inner_distances = torch.squeeze(self.inner_vectors)
            internal_matrix = derivative_operator(self.inner_distances, self.radius)
        else:
            self.inner_distances = torch.linalg.norm(self.inner_vectors, dim=2)
            internal_matrix = derivative_operator(self.inner_distances, self.radius, variables=[var.T for var in self.inner_vectors.T])

        self.phi_matrix = torch.cat((boundary_matrix, internal_matrix))
        print(torch.linalg.cond(self.phi_matrix))
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
