"""
Utilities for generating synthetic graph data for training and evaluation.

Supports various NetworkX graph generators:
- Erdos-Renyi random graphs
- Watts-Strogatz small-world graphs
- Barabasi-Albert scale-free graphs
- Random lobster graphs
- Grid graphs
- Random labeled trees
"""

from networkx import (
    erdos_renyi_graph,
    watts_strogatz_graph,
    barabasi_albert_graph,
    random_lobster,
    grid_graph,
    bfs_edges,
    relabel_nodes,
    convert_node_labels_to_integers,
    random_labeled_tree,
)


def generated_graph(
    graph_name, n=None, p=None, k=None, m=None, p1=None, p2=None, nx=None, ny=None
):
    """
    Generate a single graph from a named generator.

    Args:
        graph_name: One of 'erdos_renyi_graph', 'watts_strogatz_graph',
                    'barabasi_albert_graph', 'random_lobster', 'grid_graph',
                    'random_labeled_tree'
        n: Number of nodes
        p: Probability parameter (for Erdos-Renyi, Watts-Strogatz)
        k: Number of neighbors (Watts-Strogatz)
        m: Number of edges to attach (Barabasi-Albert)
        p1, p2: Lobster parameters
        nx, ny: Grid dimensions

    Returns:
        A NetworkX graph
    """
    if graph_name == "erdos_renyi_graph":
        assert p is not None, "define p"
        return erdos_renyi_graph(n, p)
    elif graph_name == "watts_strogatz_graph":
        assert p is not None, "define p"
        return watts_strogatz_graph(n, k, p)
    elif graph_name == "barabasi_albert_graph":
        assert m is not None, "define m"
        return barabasi_albert_graph(n, m)
    elif graph_name == "random_lobster":
        assert p1 is not None and p2 is not None, "define p1 and p2"
        return random_lobster(n, p1, p2)
    elif graph_name == "grid_graph":
        assert nx is not None and ny is not None, "define nx and ny"
        return grid_graph(dim=[nx, ny])
    elif graph_name == "random_labeled_tree":
        return random_labeled_tree(n)
    else:
        raise ValueError(f"Unknown graph type: {graph_name}")


def bfs_order(graph):
    """
    Reorder graph nodes using BFS starting from node 0.

    This canonical ordering significantly improves generation quality
    by providing a consistent node ordering across similar graphs.

    Args:
        graph: A NetworkX graph

    Returns:
        A relabeled NetworkX graph with BFS-ordered nodes
    """
    graph = convert_node_labels_to_integers(graph)
    list_edges_visited = list(bfs_edges(graph, 0))
    nodes_ordering = [0] + [edge[1] for edge in list_edges_visited]
    mapping = {node: nodes_ordering.index(node) for node in nodes_ordering}
    return relabel_nodes(graph, mapping)


def generate_dataset(
    graph_name,
    nb_graphs,
    n=None,
    p=None,
    k=None,
    m=None,
    p1=None,
    p2=None,
    nx=None,
    ny=None,
    bfs_order_activation=True,
):
    """
    Generate a dataset of graphs.

    Args:
        graph_name: Type of graph to generate
        nb_graphs: Number of graphs to generate
        bfs_order_activation: Whether to apply BFS ordering (default: True)
        **kwargs: Parameters passed to the graph generator

    Returns:
        List of NetworkX graphs
    """
    list_graphs = []
    for _ in range(nb_graphs):
        list_graphs.append(
            generated_graph(graph_name, n, p, k, m, p1, p2, nx, ny)
        )

    if bfs_order_activation:
        list_graphs = [
            bfs_order(graph)
            for graph in list_graphs
            if graph.number_of_nodes() > 1
        ]

    return list_graphs
