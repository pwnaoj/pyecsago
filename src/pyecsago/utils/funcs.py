from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def generate_synthetic_data(num_clusters: int = 10, points_per_cluster: int = 50, dimensions: int = 2, spread: float = 0.05, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic data with multiple clusters.

    Args:
        num_clusters: Number of clusters to generate.
        points_per_cluster: Number of points per cluster.
        dimensions: Dimensionality of the space (default 2D).
        spread: Spread of points around the cluster center.
        seed: Random seed for reproducibility (optional).

    Returns:
        Tuple of (data, real_centers) as numpy arrays.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate cluster centers randomly
    real_centers = np.random.rand(num_clusters, dimensions)

    # Generate points around each center
    data = []
    for center in real_centers:
        cluster_points = np.random.normal(loc=center, scale=spread, size=(points_per_cluster, dimensions))
        data.append(cluster_points)

    # Combine all generated points
    data = np.vstack(data)

    return data, real_centers

def visualize_results_simple(data: np.ndarray, real_centers: np.ndarray | None = None, refined_centers: np.ndarray | None = None, title: str = "Clustering Visualization") -> None:
    """Visualizes data, real centers (if available), and refined centers."""

    # Draw data points
    plt.scatter(data[:, 0], data[:, 1], c='lightblue', label='Data')

    # Draw real centers if available
    if real_centers is not None:
        plt.scatter(real_centers[:, 0], real_centers[:, 1], c='green', marker='x', label='Real Centers', s=100)

    # Draw refined centers if available
    if refined_centers is not None:
        plt.scatter(refined_centers[:, 0], refined_centers[:, 1], c='red', marker='o', label='Refined Centers', s=100)

    # Add legend and title
    plt.legend()
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Show plot
    plt.show()

def visualize_results(data: np.ndarray, real_centers: np.ndarray | None = None, refined_centers: np.ndarray | None = None, refined_sigmas: np.ndarray | None = None, title: str = "Clustering Visualization") -> None:
    """Visualizes data, real centers, and refined centers including radii."""
    # Draw data points
    plt.scatter(data[:, 0], data[:, 1], c='lightblue', label='Data')

    # Draw real centers if available
    if real_centers is not None:
        plt.scatter(real_centers[:, 0], real_centers[:, 1], c='green', marker='x', label='Real Centers', s=100)

    # Draw refined centers if available
    if refined_centers is not None:
        plt.scatter(refined_centers[:, 0], refined_centers[:, 1], c='red', marker='o', label='Refined Centers', s=100)
        # Draw circles representing the radius of each refined center
        if refined_sigmas is not None:
            for center, sigma in zip(refined_centers, refined_sigmas):
                circle = plt.Circle(center, sigma, color='red', fill=False, linestyle='--', linewidth=1.5)
                plt.gca().add_patch(circle)

    # Add legend and title
    plt.legend()
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Show plot
    plt.show()
