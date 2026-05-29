"""
G2PT + LLaDA discrete diffusion trainer.

Implements the main discrete diffusion approach from the paper:
- Transformer encoder processes flattened edge-index sequences
- Mask-token discrete diffusion (LLaDA-style) for training
- Iterative demasking with low-entropy remasking for generation

Reference: LLaDA - https://arxiv.org/abs/2502.09992
"""

import os

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn import functional as F

import lightning.pytorch as pl

from x_transformers import Encoder
from heavyball import ForeachSOAP

from deepgraphgen.utils import stratified_uniform_sample

torch.set_float32_matmul_precision("medium")


class TrainerG2PTLLaDA(pl.LightningModule):
    """
    G2PT trainer with LLaDA-style discrete diffusion.

    Graphs are represented as flattened edge-index sequences.
    During training, random tokens are masked and the model predicts them.
    During generation, all tokens start masked and are iteratively demasked
    using a low-entropy remasking strategy.
    """

    def __init__(
        self,
        in_dim_node=1,
        out_dim_node=1,
        hidden_dim=768,
        nb_layer=12,
        heads=12,
        nb_max_node=100,
        edges_to_node_ratio=5,
    ):
        super().__init__()

        self.nb_layer = nb_layer
        self.hidden_dim = hidden_dim
        self.nb_max_node = nb_max_node
        self.edges_to_node_ratio = edges_to_node_ratio
        self.nb_edges = edges_to_node_ratio * nb_max_node

        # Transformer encoder
        self.model_core = Encoder(dim=hidden_dim, depth=nb_layer, heads=heads)

        # Position embedding for all tokens (nodes + edges)
        self.position_embedding = torch.nn.Embedding(
            nb_max_node + self.nb_edges, hidden_dim
        )

        # Embedding for edge token values (node indices)
        self.nodes_embedding = torch.nn.Embedding(nb_max_node + 3, hidden_dim // 2)

        # Output heads for source and destination node predictions
        self.head_1 = torch.nn.Linear(hidden_dim, self.nb_max_node + 3)
        self.head_2 = torch.nn.Linear(hidden_dim, self.nb_max_node + 3)

        self.epoch_current = 0

    @property
    def mask_token(self):
        """Token ID used for masking (one past the last valid node)."""
        return self.nb_max_node + 2

    def forward(self, batch):
        """
        Forward pass through the transformer.

        Args:
            batch: dict with 'noisy_edges' (batch_size, num_edges*2) and 'edges'

        Returns:
            (targets, logits): ground-truth tokens and predicted logits
        """
        batch_size = batch["noisy_edges"].shape[0]

        noisy_edges = batch["noisy_edges"].reshape(
            batch_size, self.nb_max_node * self.edges_to_node_ratio, 2
        )
        edges_int = batch["edges"].long()

        nb_positions = self.nb_max_node * self.edges_to_node_ratio + self.nb_max_node

        # Position embedding
        position_integer = torch.arange(nb_positions, device=self.device).unsqueeze(0)
        position_integer = position_integer.repeat(batch_size, 1)
        global_embedding = self.position_embedding(position_integer)

        # Edge token embeddings (split into two halves for src/dst)
        nodes_embedding_range = self.nodes_embedding(noisy_edges)
        half = self.hidden_dim // 2
        global_embedding[:, self.nb_max_node:, :half] = (
            global_embedding[:, self.nb_max_node:, :half]
            + nodes_embedding_range[:, :, 0, :]
        )
        global_embedding[:, self.nb_max_node:, half:] = (
            global_embedding[:, self.nb_max_node:, half:]
            + nodes_embedding_range[:, :, 1, :]
        )

        # Transformer
        output_global = self.model_core(global_embedding)
        edges_output = output_global[:, self.nb_max_node:]

        # Predict source and destination
        edges_output_1 = self.head_1(edges_output)
        edges_output_2 = self.head_2(edges_output)
        edges_output = torch.stack([edges_output_1, edges_output_2], dim=-1).reshape(
            edges_output.shape[0],
            self.nb_max_node * self.edges_to_node_ratio * 2,
            self.nb_max_node + 3,
        )

        return edges_int, edges_output

    def training_step(self, batch, batch_idx):
        """Apply token masking and compute cross-entropy loss on masked positions."""
        edges_element = batch["edges"]
        batch_size = edges_element.shape[0]

        # Flatten (batch_size, num_edges, 2) -> (batch_size, num_edges * 2)
        edges_element = edges_element.reshape(batch_size, -1)

        # Apply discrete diffusion masking
        batch, mask = self._token_masking(edges_element, batch_size, return_mask=True)

        # Forward
        targets, logits = self(batch)

        # Loss only on masked tokens
        loss = F.cross_entropy(logits.transpose(1, 2), targets.long(), reduction="none")
        loss = (loss * mask.float()).sum() / mask.sum()

        self.log("train_loss", loss)

        if self.global_step % 200 == 0 and self.global_step > 0:
            with torch.no_grad():
                self.generate_graphs(10, 639, remasking="low_entropy")

        return loss

    def _token_masking(
        self, edges_elements, batch_size, time_stamp=None, return_mask=False, stratified=True
    ):
        """
        Apply LLaDA-style random masking.

        Each token is independently replaced with a mask token with probability t,
        where t is sampled uniformly (or stratified) per batch element.
        """
        if time_stamp is None:
            if stratified:
                time_stamp = stratified_uniform_sample(batch_size, device=self.device)
            else:
                time_stamp = torch.rand((batch_size,), device=self.device)

        proba_compute = torch.cat([time_stamp.unsqueeze(1), 1 - time_stamp.unsqueeze(1)], dim=1)
        masking = torch.multinomial(
            proba_compute, num_samples=edges_elements.shape[1], replacement=True
        )

        batch = {
            "time_stamp": time_stamp,
            "noisy_edges": ((1 - masking) * edges_elements + masking * self.mask_token).long(),
            "edges": edges_elements,
        }

        if return_mask:
            return batch, masking
        return batch

    def on_train_epoch_end(self):
        if self.epoch_current % 100 == 0:
            with torch.no_grad():
                self.generate_graphs(1, 500, remasking="low_entropy")
        self.epoch_current += 1

    def generate_graphs(self, batch_size, nb_step, remasking="low_entropy"):
        """
        Generate graphs via iterative demasking.

        1. Start with all tokens masked
        2. At each step, predict all tokens and demask the most confident ones
        3. Repeat until all tokens are demasked
        """
        self.eval()

        time_stamp = torch.zeros((batch_size,), device=self.device)
        edges_element = torch.ones(
            (batch_size, self.nb_max_node * self.edges_to_node_ratio * 2),
            device=self.device,
        )
        batch = self._token_masking(edges_element, batch_size, time_stamp)

        for i in range(nb_step):
            targets, logits = self(batch)

            non_mask_token = batch["noisy_edges"] != self.mask_token
            delta_choose = (~non_mask_token).sum(dim=1)[0] // (nb_step - i)

            softmax_p = F.softmax(logits, dim=2)
            max_proba_index = torch.multinomial(softmax_p.flatten(0, 1), num_samples=1)
            max_proba_index = max_proba_index.reshape(batch_size, -1)

            if remasking == "low_confidence":
                score = torch.max(softmax_p, dim=2)[0]
                score = torch.where(non_mask_token, torch.full_like(score, -1000), score)
            elif remasking == "low_entropy":
                score = torch.sum(softmax_p * torch.log(softmax_p + 1e-8), dim=2)
                score = torch.where(non_mask_token, torch.full_like(score, -1000), score)

            _, indices = torch.topk(score, k=delta_choose, dim=1)

            new_noisy = batch["noisy_edges"].clone()
            idx_batch = torch.arange(batch_size, device=self.device).unsqueeze(1).expand_as(indices)
            new_noisy[idx_batch.flatten().long(), indices.flatten().long()] = (
                max_proba_index[idx_batch.flatten().long(), indices.flatten().long()]
            )
            batch["noisy_edges"] = new_noisy

        output = batch["noisy_edges"].long()
        self._plot_graphs(output, self.nb_max_node)
        self.train()

        return output

    def _plot_graphs(self, output, nb_max_node):
        """Visualize generated graphs."""
        batch_size = output.shape[0]
        for batch_idx in range(batch_size):
            U = output[batch_idx, ::2].long().cpu().numpy()
            V = output[batch_idx, 1::2].long().cpu().numpy()

            G = nx.Graph()
            for i in range(U.shape[0]):
                if U[i] < nb_max_node and V[i] < nb_max_node:
                    G.add_edge(U[i], V[i])

            if G.number_of_nodes() == 0:
                continue

            pos = nx.spring_layout(G, seed=42)
            is_planar, _ = nx.check_planarity(G)
            print(f"Graph {batch_idx} - planar: {is_planar}, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

            try:
                w, eigvecs = np.linalg.eigh(nx.normalized_laplacian_matrix(G).toarray())
                node_color = eigvecs[:, 1]
                m = max(np.abs(node_color.min()), np.abs(node_color.max()))
                vmin, vmax = -m, m
            except Exception:
                node_color = range(G.number_of_nodes())
                vmin, vmax = None, None

            plt.figure(figsize=(10, 10))
            nx.draw(
                G, pos, font_size=5, node_size=60, with_labels=False,
                node_color=node_color, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax,
                edge_color="grey",
            )

            name_img = f"graph_visu/graph_auto_epoch_{self.epoch_current}_{batch_idx}.png"
            os.makedirs("graph_visu", exist_ok=True)
            plt.savefig(name_img)
            plt.clf()

            try:
                img = plt.imread(name_img)[:, :, :3].transpose((2, 0, 1))
                self.logger.experiment.add_image("pred_image", img, global_step=self.global_step)
                os.remove(name_img)
            except Exception:
                pass

    def configure_optimizers(self):
        return ForeachSOAP(
            self.parameters(), lr=1e-3, foreach=False, warmup_steps=1000,
        )


def add_gumbel_noise(logits, temperature):
    """
    Gumbel max trick for categorical sampling.
    Uses float64 for improved generation quality (cf. arXiv:2409.02908).
    """
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
