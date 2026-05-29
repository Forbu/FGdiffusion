"""Tests for diffusion noise generation."""

import numpy as np
import networkx as nx

from deepgraphgen.diffusion_generation import (
    generate_beta_value,
    compute_mean_value_whole_noise,
    add_noise_to_graph,
    transform_to_symmetric,
)
from deepgraphgen.datageneration import generated_graph


def test_generate_beta_value():
    """Test beta schedule generation."""
    t_array = np.linspace(0, 1, 1000)
    beta = generate_beta_value(0.1, 10.0, t_array)

    assert beta.shape == (1000,)
    assert np.isclose(beta[0], 0.1)
    assert np.isclose(beta[-1], 10.0)


def test_symmetric_matrix():
    """Test matrix symmetrization."""
    matrix = np.random.randn(10, 10)
    sym = transform_to_symmetric(matrix)
    assert np.allclose(sym, sym.T)


def test_noise_addition():
    """Test noise addition to adjacency matrix."""
    graph = generated_graph("grid_graph", nx=5, ny=5)
    adj = nx.adjacency_matrix(graph).todense()
    adj = 2 * adj - 1

    noisy, gradient = add_noise_to_graph(adj, 0.5, 0.1)
    assert noisy.shape == adj.shape
    assert gradient.shape == adj.shape
    # Noisy matrix should be symmetric
    assert np.allclose(noisy, noisy.T, atol=1e-10)
