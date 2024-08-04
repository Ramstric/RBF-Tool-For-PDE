import numpy as np
from RBF import RadialBasisFunctions_Numpy as RadialBasisFunctions


class WeightsInterpolator(object):
    def __init__(self, center_points, weights, radius: float, rbf_name: str):
        self.rbf = getattr(RadialBasisFunctions, rbf_name)

        self.radius = radius
        self.weights_matrix = weights

        # Data points, from meshgrid to pairs of coordinates [x, y, z, ...]
        self.center_points = center_points

    def interpolate(self, *r):
        interpol_vectors = np.stack([arg.ravel() for arg in r]).T

        # Distance between the interpolation points and the rbf centers
        interpol_distance = np.stack([x - self.center_points for x in interpol_vectors])
        interpol_distance = np.linalg.norm(interpol_distance, axis=2)

        # Applying radial basis function
        temp = self.rbf(interpol_distance, self.radius)

        # Adding weights
        temp = np.matmul(temp, self.weights_matrix)

        return np.reshape(temp, r[0].shape)
