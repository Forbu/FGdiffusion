"""Tests for dataset classes."""

import torch
from deepgraphgen.datasets import TreeDataset, SpectreGraphDataset
from deepgraphgen.datageneration import generate_dataset


def test_tree_dataset():
    """Test TreeDataset produces correctly shaped outputs."""
    dataset = TreeDataset(nb_graphs=10, n=10, edges_to_nodes_ratio=5)
    assert len(dataset) == 10

    sample = dataset[0]
    assert "nodes" in sample
    assert "edges" in sample
    assert sample["nodes"].shape[0] == 10
    # Edges should be padded to n * edges_to_nodes_ratio
    assert sample["edges"].shape[0] == 10 * 5
    assert sample["edges"].shape[1] == 2


def test_tree_dataset_edge_values():
    """Test that edge values are valid node indices or padding."""
    dataset = TreeDataset(nb_graphs=5, n=10, edges_to_nodes_ratio=5)
    sample = dataset[0]
    edges = sample["edges"]

    # All values should be in [0, n] where n=10 is the padding index
    assert edges.min() >= 0
    assert edges.max() <= 10


def test_spectre_dataset_download():
    """Test that Spectre dataset can be instantiated (downloads if needed)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = SpectreGraphDataset(
            dataset_name="planar",
            download_dir=tmpdir,
            edges_to_node_ratio=5,
        )
        assert len(dataset) > 0

        sample = dataset[0]
        assert "nodes" in sample
        assert "edges" in sample
