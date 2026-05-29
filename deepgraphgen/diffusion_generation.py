"""
Diffusion noise scheduling and graph noise utilities.

Implements the continuous diffusion process from GraphGDP:
https://arxiv.org/pdf/2212.01842.pdf
"""

import numpy as np


def generate_beta_value(beta_min, beta_max, t_array):
    """
    Generate beta values for the diffusion schedule.

    β(t) = β_min + t * (β_max - β_min)  for t ∈ [0, 1]
    """
    return beta_min + t_array * (beta_max - beta_min)


def compute_mean_value_noise(t_array, whole_beta_values, index_t):
    """
    Compute the mean and variance for the forward diffusion process.

    p_0t(A_t | A_0) = N(A_t; A_0 * e^{-∫₀ᵗ ½β(s)ds}, I * (1 - e^{-∫₀ᵗ β(s)ds}))
    """
    integral_beta = np.trapz(whole_beta_values[:index_t], t_array[:index_t])
    mean = np.exp(-0.5 * integral_beta)
    variance = 1 - np.exp(-integral_beta)
    return mean, variance


def compute_mean_value_whole_noise(t_array, whole_beta_values):
    """Compute mean and variance for all time steps."""
    mean_values = []
    variance_values = []
    for index_t in range(len(t_array)):
        mean, variance = compute_mean_value_noise(t_array, whole_beta_values, index_t)
        mean_values.append(mean)
        variance_values.append(variance)
    return mean_values, variance_values


def transform_to_symmetric(matrix):
    """
    Symmetrize a matrix by averaging upper and lower triangles.
    """
    return np.triu(matrix, k=0) + np.triu(matrix, k=1).T


def add_noise_to_graph(graph, mean_beta, variance):
    """
    Add Gaussian noise to a graph's adjacency matrix.

    Args:
        graph: Adjacency matrix (numpy array)
        mean_beta: Mean scaling factor
        variance: Noise variance

    Returns:
        (noisy_matrix, gradient): Noisy adjacency matrix and score gradient
    """
    mean_beta = graph * mean_beta
    noise_matrix = np.random.normal(mean_beta, np.sqrt(variance), size=mean_beta.shape)
    noise_matrix = transform_to_symmetric(noise_matrix)

    if variance == 0:
        gradient = np.zeros(graph.shape)
    else:
        gradient = -(noise_matrix - mean_beta) / variance

    return noise_matrix, gradient
