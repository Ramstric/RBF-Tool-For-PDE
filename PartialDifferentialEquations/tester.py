import numpy as np

# Example points (x, y)
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 1.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.2, 0.5],
    [0.8, 0.5],
    # ... more points
])

from scipy.spatial.distance import cdist

def find_stencil(target_point, all_points, stencil_size):
    """Finds the nearest neighbors for a target point."""
    distances = cdist([target_point], all_points)[0]  # Calculate distances
    nearest_indices = np.argsort(distances)[:stencil_size] #Get the indices of the closest points
    return all_points[nearest_indices]

# Example usage:
target_point = np.array([0.5, 0.5])
stencil_size = 5
stencil = find_stencil(target_point, points, stencil_size)

print("Target point:", target_point)
print("Stencil points:\n", stencil)

import matplotlib.pyplot as plt

def visualize_stencil(target_point, stencil, all_points):
    """Visualizes the stencil."""
    plt.figure(figsize=(6, 6))
    plt.scatter(all_points[:, 0], all_points[:, 1], label="All Points")
    plt.scatter(target_point[0], target_point[1], color="red", label="Target Point")
    plt.scatter(stencil[:, 0], stencil[:, 1], color="green", label="Stencil Points")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Stencil Visualization")
    plt.grid(True)
    plt.show()

visualize_stencil(target_point, stencil, points)