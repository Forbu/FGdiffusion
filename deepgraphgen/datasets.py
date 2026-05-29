"""
Dataset classes for discrete diffusion graph generation.

Provides:
- SpectreGraphDataset: Planar, SBM, Community benchmark graphs (edge-index format with BFS ordering)
- TreeDataset: Random labeled trees (edge-index format)
- TreeAdjacencyDataset: Random labeled trees (adjacency matrix format)
- NasaKGDataset: NASA Knowledge Graph with typed nodes/edges (subgraph sampling)
"""

import os
import os.path as osp
import ssl
import sys
import urllib
import heapq
from typing import Optional
from collections import Counter

import networkx as nx
from networkx import convert_node_labels_to_integers, bfs_edges, relabel_nodes

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

import torch_geometric.utils

from deepgraphgen.datageneration import generate_dataset


# ─── Download helper ─────────────────────────────────────────────────────────

def download_url(url: str, folder: str, log: bool = True, filename: Optional[str] = None):
    """Download a file from URL to folder."""
    import fsspec

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)
    if os.path.exists(path):
        return path

    if log:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(folder, exist_ok=True)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with fsspec.open(path, "wb") as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


# ─── Spectre benchmark datasets ─────────────────────────────────────────────

class SpectreGraphDataset(Dataset):
    """
    Benchmark graph datasets from the SPECTRE paper.
    Downloads from https://github.com/KarolisMart/SPECTRE

    Each sample returns a flattened edge-index sequence with BFS node ordering,
    padded to a fixed length determined by edges_to_node_ratio.

    Supported datasets: 'planar', 'sbm', 'comm20'
    """

    FILE_MAPPING = {
        "sbm": "sbm_200.pt",
        "planar": "planar_64_200.pt",
        "comm20": "community_12_21_100.pt",
    }

    URL_MAPPING = {
        "sbm": "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt",
        "planar": "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt",
        "comm20": "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt",
    }

    def __init__(self, dataset_name, download_dir, edges_to_node_ratio=5):
        super().__init__()
        self.dataset_name = dataset_name
        self.download_dir = download_dir
        self.edges_to_node_ratio = edges_to_node_ratio

        self._download()
        self._preprocess()

    def _download(self):
        if os.path.exists(
            os.path.join(self.download_dir, self.FILE_MAPPING[self.dataset_name])
        ):
            return
        download_url(self.URL_MAPPING[self.dataset_name], self.download_dir)

    def _preprocess(self):
        self.list_graph = []
        data = torch.load(
            os.path.join(self.download_dir, self.FILE_MAPPING[self.dataset_name]),
            weights_only=False,
        )
        adjs = data[0]
        self.n_nodes = data[3]

        for i in range(len(adjs)):
            adjs[i], _ = torch_geometric.utils.dense_to_sparse(adjs[i])
            G = nx.Graph()
            G.add_edges_from(zip(adjs[i].T[:, 0].tolist(), adjs[i].T[:, 1].tolist()))
            self.list_graph.append(G)

    def __len__(self):
        return len(self.list_graph)

    def _apply_bfs_ordering(self, graph, nb_nodes):
        """Apply BFS ordering starting from a random node."""
        graph = convert_node_labels_to_integers(graph)
        id_node_start = torch.randint(0, nb_nodes, (1,)).item()

        list_edges_visited = list(bfs_edges(graph, id_node_start))
        nodes_ordering = [id_node_start] + [edge[1] for edge in list_edges_visited]
        mapping = {node: nodes_ordering.index(node) for node in nodes_ordering}
        return relabel_nodes(graph, mapping)

    def __getitem__(self, idx):
        nb_nodes = self.n_nodes[idx]
        nodes = torch.arange(nb_nodes)
        graph = self._apply_bfs_ordering(self.list_graph[idx], nb_nodes)

        edges = torch.tensor(list(graph.edges)).T.long()
        nb_edges = edges.shape[1]

        # Sort edges by source node
        indices = torch.argsort(edges[0, :], dim=-1)
        edges = edges[:, indices]
        nb_edges = edges.shape[1]

        # Pad or truncate to fixed size
        target_size = int(nb_nodes * self.edges_to_node_ratio)
        if nb_edges < target_size:
            pad = torch.ones((2, target_size - nb_edges), dtype=torch.long) * nb_nodes
            edges = torch.cat([edges, pad], dim=1)
        elif nb_edges > target_size:
            edges = edges[:, :target_size]

        return {"nodes": nodes, "edges": edges.T}


class SpectreIterableDataset(IterableDataset):
    """Wraps a SpectreGraphDataset for infinite iteration."""

    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        i = 0
        while True:
            yield self.dataset[i % len(self.dataset)]
            i += 1


# ─── Tree datasets ───────────────────────────────────────────────────────────

class TreeDataset(Dataset):
    """Dataset of random labeled trees in edge-index format (for discrete diffusion)."""

    def __init__(self, nb_graphs, n, edges_to_nodes_ratio=10):
        self.nb_graphs = nb_graphs
        self.n = n
        self.edges_to_nodes_ratio = edges_to_nodes_ratio
        self.list_graphs = generate_dataset("random_labeled_tree", nb_graphs, n=n)

    def __len__(self):
        return self.nb_graphs

    def __getitem__(self, idx):
        graph = self.list_graphs[idx]
        nodes = torch.tensor(list(range(self.n)))
        edges = torch.tensor(list(graph.edges))

        # Pad to fixed size
        edges = torch.cat([
            edges,
            torch.ones(
                (self.n * self.edges_to_nodes_ratio - edges.shape[0], 2),
                dtype=torch.long,
            ) * self.n,
        ])

        return {"nodes": nodes, "edges": edges}


class TreeAdjacencyDataset(Dataset):
    """Tree dataset returning adjacency matrices."""

    def __init__(self, nb_graphs, n):
        self.nb_graphs = nb_graphs
        self.n = n
        self.list_graphs = generate_dataset("random_labeled_tree", nb_graphs, n=n)

    def __len__(self):
        return self.nb_graphs

    def __getitem__(self, idx):
        graph = self.list_graphs[idx]
        A = nx.adjacency_matrix(graph)
        return {"adjacency": A.toarray()}


# ─── Knowledge Graph dataset ─────────────────────────────────────────────────

def create_graph_from_edges(nodes_df, edges_df):
    """Create a NetworkX graph from nodes and edges DataFrames."""
    graph = nx.Graph()
    for _, row in nodes_df.iterrows():
        graph.add_node(int(row["id"]), label=row["labels_int"])
    for _, row in edges_df.iterrows():
        graph.add_edge(int(row["source"]), int(row["target"]), label=row["labels_int"])
    return graph


def sampling_local_subgraph_optimized(G, start_node, nb_nodes_to_sample):
    """
    Sample a local subgraph using greedy expansion.
    Iteratively adds the node with the most connections to the current subgraph.
    Uses a priority queue for efficiency.
    """
    if start_node not in G:
        raise ValueError(f"Start node {start_node} not found in graph.")
    if nb_nodes_to_sample <= 0:
        return G.subgraph([]).copy()
    if nb_nodes_to_sample == 1:
        return G.subgraph([start_node]).copy()

    subgraph_nodes = {start_node}
    candidate_counts = Counter()
    priority_queue = []

    for neighbor in G.neighbors(start_node):
        if neighbor not in subgraph_nodes:
            candidate_counts[neighbor] += 1
            heapq.heappush(priority_queue, (-candidate_counts[neighbor], neighbor))

    while len(subgraph_nodes) < nb_nodes_to_sample:
        next_node = None

        while priority_queue:
            current_priority, node = heapq.heappop(priority_queue)
            current_count = -current_priority
            if node not in subgraph_nodes and candidate_counts[node] == current_count:
                next_node = node
                break

        if next_node is None:
            break

        subgraph_nodes.add(next_node)

        for neighbor in G.neighbors(next_node):
            if neighbor not in subgraph_nodes:
                candidate_counts[neighbor] += 1
                heapq.heappush(priority_queue, (-candidate_counts[neighbor], neighbor))

    return G.subgraph(list(subgraph_nodes)).copy()


class NasaKGDataset(Dataset):
    """
    NASA Knowledge Graph dataset for typed KG generation.

    Samples local subgraphs from a large knowledge graph,
    returning node labels, edge indices, and edge labels.
    """

    def __init__(self, path_nodes, path_edges, nb_nodes_to_sample=64, edges_to_nodes_ratio=3):
        self.nb_nodes_to_sample = nb_nodes_to_sample
        self.edges_to_nodes_ratio = edges_to_nodes_ratio

        self.nodes = pd.read_parquet(path_nodes)
        self.edges = pd.read_parquet(path_edges)
        self.graph = create_graph_from_edges(self.nodes, self.edges)

    def __len__(self):
        return len(self.graph.nodes)

    def __getitem__(self, idx):
        try:
            subgraph = sampling_local_subgraph_optimized(
                self.graph, idx, self.nb_nodes_to_sample
            )
            subgraph = convert_node_labels_to_integers(subgraph, first_label=0)

            edges = torch.tensor(list(subgraph.edges)).T.long()
            edges_label = torch.tensor(
                [subgraph.edges[e]["label"] for e in subgraph.edges]
            ).long()
            nodes_label = torch.tensor(
                [subgraph.nodes[n]["label"] for n in subgraph.nodes]
            ).long()

            nb_edges = edges.shape[1]
            target_size = self.nb_nodes_to_sample * self.edges_to_nodes_ratio

            if nb_edges < target_size:
                pad = torch.ones((2, target_size - nb_edges), dtype=torch.long) * self.nb_nodes_to_sample
                edges = torch.cat([edges, pad], dim=1)
                edges_label = torch.cat([
                    edges_label,
                    torch.ones((target_size - nb_edges,), dtype=torch.long) * 10,
                ])
            elif nb_edges > target_size:
                edges = edges[:, :target_size]
                edges_label = edges_label[:target_size]

            if edges.shape[1] != target_size or edges_label.shape[0] != target_size:
                return self.__getitem__((idx + 1) % self.__len__())
            if nodes_label.shape[0] != self.nb_nodes_to_sample:
                return self.__getitem__((idx + 1) % self.__len__())

        except Exception as e:
            print(f"Error: {e}")
            return self.__getitem__((idx + 1) % self.__len__())

        return {
            "nodes_label": nodes_label,
            "edges": edges.T,
            "edges_label": edges_label,
        }
