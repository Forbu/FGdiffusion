"""
Evaluate KG generation with graph validity checking.

Usage:
    python scripts/evaluate_kg.py --checkpoint path/to/checkpoint.ckpt
"""

import argparse
import numpy as np
from scipy.stats import wasserstein_distance

import networkx as nx

from deepgraphgen.trainers.trainer_llada_kg import TrainerG2PTKG
from deepgraphgen.datasets import NasaKGDataset


def check_kg_validity(graph, rules):
    """
    Check if all edges in a KG respect the given type rules.

    Args:
        graph: NetworkX graph with 'labels' on nodes and edges
        rules: Nx3 array of (edge_label, source_label, target_label)

    Returns:
        1 if valid, 0 otherwise
    """
    rules_set = {tuple(rule) for rule in rules}

    for u, v, data in graph.edges(data=True):
        try:
            source_label = graph.nodes[u]["labels"]
            target_label = graph.nodes[v]["labels"]
            edge_label = data["labels"]

            current = (edge_label.item(), source_label.item(), target_label.item())
            inverse = (edge_label.item(), target_label.item(), source_label.item())

            if current not in rules_set and inverse not in rules_set:
                return 0
        except KeyError:
            return 0

    return 1


def compute_kg_metrics(edges, node_labels, edge_labels, nb_max_node, rules):
    """Compute KG-specific metrics."""
    validity_list = []
    degree_list = []
    cluster_list = []

    for idx in range(edges.shape[0]):
        U = edges[idx, ::2].long().cpu().numpy()
        V = edges[idx, 1::2].long().cpu().numpy()

        G = nx.Graph()
        for i in range(U.shape[0]):
            if U[i] < nb_max_node and V[i] < nb_max_node:
                G.add_edge(U[i], V[i], labels=edge_labels[idx][i])
                G.add_node(U[i], labels=node_labels[idx][U[i]].item())
                G.add_node(V[i], labels=node_labels[idx][V[i]].item())

        if G.number_of_nodes() == 0:
            continue

        validity_list.append(check_kg_validity(G, rules))
        degree_list.append(sum(dict(G.degree()).values()) / nb_max_node)
        cluster_list.append(nx.average_clustering(G))

    n = edges.shape[0]
    print(f"KG Validity: {sum(validity_list)}/{n} ({sum(validity_list)/n:.2%})")
    print(f"Avg degree: {np.mean(degree_list):.4f}")
    print(f"Avg clustering: {np.mean(cluster_list):.4f}")

    return validity_list, degree_list, cluster_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # Example rules for NASA KG
    rules = np.array([
        [2, 1, 0], [8, 1, 5], [1, 6, 5], [0, 5, 5],
        [7, 4, 1], [4, 3, 1], [5, 6, 1], [3, 2, 3], [6, 6, 6],
    ])

    model = TrainerG2PTKG.load_from_checkpoint(
        args.checkpoint,
        nb_max_node=64,
        hidden_dim=386,
        nb_layer=6,
        heads=8,
        edges_to_node_ratio=5,
    )

    # Generate and evaluate
    batch = model.generate_graphs(32, 400, logging=False)

    # Compute metrics
    validity, degrees, clusters = compute_kg_metrics(
        batch["noisy_edges"],
        batch["noisy_nodes_label"],
        batch["noisy_edges_label"],
        64,
        rules,
    )
