"""
Evaluate generated graphs using MMD metrics (degree, clustering, orbits).

Usage:
    python scripts/evaluate.py --checkpoint path/to/checkpoint.ckpt
"""

import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance

import networkx as nx
import torch
import lightning.pytorch as pl

from deepgraphgen.trainers.trainer_llada import TrainerG2PTLLaDA
from deepgraphgen.datasets import SpectreGraphDataset, SpectreIterableDataset


def count_orbit(nx_graph):
    """Compute number of orbits using pynauty."""
    try:
        from pynauty import Graph as NautyGraph, autgrp
    except ImportError:
        return 0

    num_nodes = nx_graph.number_of_nodes()
    if num_nodes == 0:
        return 0

    node_list = list(nx_graph.nodes())
    node_to_int = {node: i for i, node in enumerate(node_list)}
    int_to_node = {i: node for i, node in enumerate(node_list)}

    adj_dict = {}
    for i in range(num_nodes):
        u = int_to_node[i]
        adj_dict[i] = [node_to_int[v] for v in nx_graph.neighbors(u)]

    g = NautyGraph(
        number_of_vertices=num_nodes,
        directed=nx.is_directed(nx_graph),
        adjacency_dict=adj_dict,
    )
    _, _, _, _, numorbits = autgrp(g)
    return numorbits


def compute_eval_metrics(output_concat, nb_max_node):
    """Compute degree, clustering, and orbit metrics for generated graphs."""
    degree_list = []
    cluster_list = []
    orbit_list = []
    planar_count = 0
    tree_count = 0

    for batch_idx in range(output_concat.shape[0]):
        U = output_concat[batch_idx, ::2].long().cpu().numpy()
        V = output_concat[batch_idx, 1::2].long().cpu().numpy()

        G = nx.Graph()
        for i in range(U.shape[0]):
            if U[i] < nb_max_node and V[i] < nb_max_node:
                G.add_edge(U[i], V[i])

        if G.number_of_nodes() == 0:
            continue

        degrees = dict(G.degree())
        mean_degree = sum(degrees.values()) / nb_max_node
        avg_cluster = nx.average_clustering(G)
        nb_orbit = count_orbit(G)
        is_planar, _ = nx.check_planarity(G)
        is_tree = nx.is_tree(G)

        degree_list.append(mean_degree)
        cluster_list.append(avg_cluster)
        orbit_list.append(nb_orbit)
        planar_count += is_planar
        tree_count += is_tree

    n = output_concat.shape[0]
    print(f"Planar: {planar_count}/{n} ({planar_count/n:.2%})")
    print(f"Tree: {tree_count}/{n} ({tree_count/n:.2%})")
    print(f"Avg degree: {np.mean(degree_list):.4f}")
    print(f"Avg clustering: {np.mean(cluster_list):.4f}")
    print(f"Avg orbits: {np.mean(orbit_list):.4f}")

    return degree_list, cluster_list, orbit_list


def mmd_rbf(X, Y):
    """Compute MMD using Wasserstein distance."""
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(Y, list):
        Y = np.array(Y)
    return wasserstein_distance(X, Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nb_rounds", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nb_step", type=int, default=100)
    parser.add_argument("--edges_to_node_ratio", type=int, default=5)
    parser.add_argument("--nb_max_node", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=386)
    parser.add_argument("--nb_layer", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    args = parser.parse_args()

    model = TrainerG2PTLLaDA.load_from_checkpoint(
        args.checkpoint,
        nb_max_node=args.nb_max_node,
        hidden_dim=args.hidden_dim,
        nb_layer=args.nb_layer,
        heads=args.heads,
        edges_to_node_ratio=args.edges_to_node_ratio,
    )

    training_dataset = SpectreGraphDataset(
        dataset_name="planar",
        download_dir="scripts/datasets",
        edges_to_node_ratio=args.edges_to_node_ratio,
    )

    random.seed(246)
    np.random.seed(4812)

    # Generate
    gen_outputs = []
    for _ in tqdm(range(args.nb_rounds)):
        output = model.generate_graphs(args.batch_size, args.nb_step)
        gen_outputs.append(output)
    gen_concat = torch.cat(gen_outputs, dim=0)

    print(f"\n=== Generated graphs ({gen_concat.shape[0]}) ===")
    gen_degree, gen_cluster, gen_orbits = compute_eval_metrics(gen_concat, args.nb_max_node)

    # Ground truth
    true_outputs = []
    for idx, data in enumerate(training_dataset):
        true_outputs.append(data["edges"].flatten())
        if idx >= args.batch_size * args.nb_rounds:
            break
    true_concat = torch.stack(true_outputs, dim=0)

    print(f"\n=== Ground truth ({true_concat.shape[0]}) ===")
    true_degree, true_cluster, true_orbits = compute_eval_metrics(true_concat, args.nb_max_node)

    print(f"\n=== MMD Metrics ===")
    print(f"MMD degree:      {mmd_rbf(true_degree, gen_degree):.9f}")
    print(f"MMD clustering:  {mmd_rbf(true_cluster, gen_cluster):.9f}")
    print(f"MMD orbits:      {mmd_rbf(true_orbits, gen_orbits):.9f}")
