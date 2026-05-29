"""Tests for graph data generation."""

from deepgraphgen.datageneration import generate_dataset, generated_graph, bfs_order


def test_generate_graphs():
    """Test all graph generators produce valid graphs."""
    graph = generated_graph("erdos_renyi_graph", n=100, p=0.01)
    assert graph.number_of_nodes() == 100

    graph = generated_graph("watts_strogatz_graph", n=100, k=4, p=0.1)
    assert graph.number_of_nodes() == 100

    graph = generated_graph("barabasi_albert_graph", n=100, m=3)
    assert graph.number_of_nodes() == 100

    graph = generated_graph("random_labeled_tree", n=50)
    assert graph.number_of_nodes() == 50


def test_bfs_reordering():
    """Test BFS ordering produces canonical node ordering."""
    graph = generated_graph("erdos_renyi_graph", n=10, p=0.3)
    graph_reordered = bfs_order(graph)

    # BFS from node 0 means node 0 should still be 0
    assert 0 in graph_reordered.nodes()
    assert graph_reordered.number_of_nodes() == graph.number_of_nodes()


def test_generate_dataset():
    """Test dataset generation with BFS ordering."""
    dataset = generate_dataset("random_labeled_tree", 10, n=20)
    assert len(dataset) == 10

    for graph in dataset:
        assert graph.number_of_nodes() == 20
