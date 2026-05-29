"""
G2PT + Score-based discrete diffusion trainer (SEDD).

Implements score-entropy discrete diffusion with a DiT backbone:
- Score network predicts the ratio of the data distribution
- Uses a continuous-time transition matrix Q for the masking process
- Tau-leaping for the reverse sampling process

Reference: SEDD - https://arxiv.org/abs/2310.16834
"""

import os
import math

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


class TrainerG2PTScore(pl.LightningModule):
    """
    G2PT trainer with score-based discrete diffusion (SEDD).

    Unlike LLaDA which uses simple mask/unmask, this approach:
    - Learns a score function s(x)_ij estimating the ratio p(x)/p_t(x)
    - Uses a transition matrix Q with absorbing state (mask token)
    - Applies tau-leaping for discrete-time reverse sampling
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
        add_degree_feature=True,
        add_spectral_feature=True,
        k_eigvecs=4,
    ):
        super().__init__()

        self.nb_layer = nb_layer
        self.hidden_dim = hidden_dim
        self.nb_max_node = nb_max_node
        self.edges_to_node_ratio = edges_to_node_ratio
        self.add_degree_feature = add_degree_feature
        self.add_spectral_feature = add_spectral_feature
        self.k_eigvecs = k_eigvecs
        self.nb_edges = edges_to_node_ratio * nb_max_node

        # Transformer encoder
        self.model_core = Encoder(dim=hidden_dim, depth=nb_layer, heads=heads)

        # Time projection
        self.time_projection = torch.nn.Linear(1, hidden_dim)

        # Position embedding
        self.position_embedding = torch.nn.Embedding(
            nb_max_node + self.nb_edges, hidden_dim
        )

        if self.add_degree_feature:
            self.degree_projection = torch.nn.Linear(1, hidden_dim)

        if self.add_spectral_feature:
            self.spectral_projection = torch.nn.Linear(self.k_eigvecs, hidden_dim)

        # Edge token embedding
        self.nodes_embedding = torch.nn.Embedding(nb_max_node + 3, hidden_dim // 2)

        # Output heads
        self.head_1 = torch.nn.Linear(hidden_dim, self.nb_max_node + 3)
        self.head_2 = torch.nn.Linear(hidden_dim, self.nb_max_node + 3)

        # Transition matrix Q: absorbing to mask state
        self.Q = -torch.eye(self.nb_max_node + 3)
        self.Q[-1, :] = 1
        self.Q[-1, -1] = 0
        self.register_buffer("ref_Q", self.Q)

        self.sigma_min = 1.0
        self.sigma_max = 5.0

        self.mask_matrix_init = torch.zeros((self.nb_max_node + 3, self.nb_max_node + 3))
        self.mask_matrix_init[-1, :] = 1
        self.register_buffer("mask_matrix", self.mask_matrix_init)

        self.epoch_current = 0

    def _calculate_degree_features(self, noisy_edges, batch_size):
        """Compute degree-based node features from current edges."""
        noisy_edges_flat = noisy_edges.flatten(start_dim=1)
        valid_mask = noisy_edges_flat < self.nb_max_node
        masked_edges = torch.where(valid_mask, noisy_edges_flat, self.nb_max_node)

        one_hot = F.one_hot(masked_edges, num_classes=self.nb_max_node + 1)
        degrees = one_hot.sum(dim=1)[:, : self.nb_max_node]
        return self.degree_projection(degrees.unsqueeze(-1).float())

    def _calculate_spectral_features(self, noisy_edges, batch_size):
        """Compute spectral (Laplacian eigenvector) features."""
        with torch.no_grad():
            adj = torch.zeros(batch_size, self.nb_max_node, self.nb_max_node, device=self.device)
            u = noisy_edges[:, :, 0]
            v = noisy_edges[:, :, 1]
            num_edges = noisy_edges.shape[1]

            batch_idx = torch.arange(batch_size, device=self.device).view(-1, 1).expand(batch_size, num_edges)
            valid_mask = (u < self.nb_max_node) & (v < self.nb_max_node)

            indices = torch.stack([batch_idx[valid_mask], u[valid_mask], v[valid_mask]], dim=0)
            values = torch.ones(indices.shape[1], device=self.device, dtype=torch.float)
            adj_sparse = torch.sparse_coo_tensor(
                indices, values, (batch_size, self.nb_max_node, self.nb_max_node)
            )
            adj = adj_sparse.to_dense()
            adj = (adj + adj.transpose(1, 2)).clamp(0, 1)

            d = adj.sum(dim=-1)
            d_inv_sqrt = torch.pow(d, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
            D_inv_sqrt = torch.diag_embed(d_inv_sqrt)
            L_norm = (
                torch.eye(self.nb_max_node, device=self.device).unsqueeze(0)
                - torch.bmm(D_inv_sqrt, torch.bmm(adj, D_inv_sqrt))
            )

            eigvals, eigvecs = torch.linalg.eigh(L_norm)

        spectral_features = eigvecs[:, :, 1 : self.k_eigvecs + 1].detach()
        return self.spectral_projection(spectral_features)

    def forward(self, batch):
        """Forward pass: compute score for each token."""
        batch_size = batch["noisy_edges"].shape[0]

        noisy_edges = batch["noisy_edges"].reshape(
            batch_size, self.nb_max_node * self.edges_to_node_ratio, 2
        )

        nb_positions = self.nb_max_node * self.edges_to_node_ratio + self.nb_max_node

        # Position embedding
        position_integer = torch.arange(nb_positions, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        global_embedding = self.position_embedding(position_integer)

        # Node features (degree + spectral)
        node_features_emb = torch.zeros(
            batch_size, self.nb_max_node, self.hidden_dim, device=self.device
        )
        if self.add_degree_feature:
            node_features_emb += self._calculate_degree_features(noisy_edges, batch_size)
        if self.add_spectral_feature:
            node_features_emb += self._calculate_spectral_features(noisy_edges, batch_size)
        global_embedding[:, : self.nb_max_node, :] += node_features_emb

        # Edge token embeddings
        nodes_emb = self.nodes_embedding(noisy_edges)
        half = self.hidden_dim // 2
        global_embedding[:, self.nb_max_node:, :half] += nodes_emb[:, :, 0, :]
        global_embedding[:, self.nb_max_node:, half:] += nodes_emb[:, :, 1, :]

        # Time-conditioned transformer
        time_emb = self.time_projection(batch["time_stamp"].unsqueeze(1))
        output_global = self.model_core(global_embedding)

        edges_output = output_global[:, self.nb_max_node:]
        edges_output_1 = self.head_1(edges_output)
        edges_output_2 = self.head_2(edges_output)
        edges_output = torch.stack([edges_output_1, edges_output_2], dim=-1).reshape(
            batch_size, self.nb_max_node * self.edges_to_node_ratio * 2, self.nb_max_node + 3
        )

        return F.elu(edges_output) + 1.0

    def training_step(self, batch, batch_idx):
        edges_element = batch["edges"].reshape(batch["edges"].shape[0], -1)
        batch_size = edges_element.shape[0]

        batch = self._token_masking(edges_element, batch_size)
        edges_score = self(batch)

        loss = (
            edges_score
            - torch.log(edges_score + 1e-8) * batch["proba_y"] / batch["proba_x"]
        )
        loss = (loss * batch["mask_x"]).sum(dim=-1).mean()

        self.log("train_loss", loss)

        if self.global_step % 200 == 0 and self.global_step > 0:
            with torch.no_grad():
                self.generate_graphs(2, 300)

        return loss

    def _token_masking(self, edges_elements, batch_size, stratified=True):
        """
        Apply score-based transition matrix masking.

        Uses the continuous-time transition kernel exp(tQ) where Q is
        the absorbing transition matrix.
        """
        with torch.no_grad():
            time_stamp = (
                stratified_uniform_sample(batch_size, device=self.device)
                if stratified
                else torch.rand((batch_size,), device=self.device)
            )

            exp_sigma_t = calculate_exp_neg_F(time_stamp).unsqueeze(-1).unsqueeze(-1)
            diago = exp_sigma_t * torch.eye(
                self.nb_max_node + 3, device=exp_sigma_t.device
            ).unsqueeze(0)
            diago[:, -1, -1] = 0

            full_transition = diago + (1 - exp_sigma_t) * self.mask_matrix.unsqueeze(0)

            edges_one_hot = F.one_hot(
                edges_elements, num_classes=self.nb_max_node + 3
            ).transpose(1, 2).float()

            edges_proba = torch.bmm(full_transition, edges_one_hot).transpose(1, 2)
            edges_proba = torch.clamp(edges_proba, min=1e-8, max=1.0)

            sampler = torch.distributions.categorical.Categorical(probs=edges_proba)
            edges_sample = sampler.sample()

            proba_x = torch.gather(edges_proba, 2, edges_sample.unsqueeze(-1))

            mask_x = torch.ones_like(edges_proba, dtype=torch.float32)
            mask_x.scatter_(dim=2, index=edges_sample.unsqueeze(-1), value=0.0)

        return {
            "noisy_edges": edges_sample,
            "time_stamp": time_stamp,
            "proba_y": edges_proba,
            "proba_x": proba_x,
            "mask_x": mask_x,
        }

    def generate_graphs(self, batch_size, nb_step, log=True):
        """Generate graphs via tau-leaping reverse process."""
        self.eval()

        mask_id = self.nb_max_node + 2
        nb_elements = self.nb_max_node * self.edges_to_node_ratio * 2

        edges_elements = torch.ones((batch_size, nb_elements), device=self.device).long() * mask_id

        batch = {
            "noisy_edges": edges_elements,
            "time_stamp": torch.ones((batch_size,), device=self.device),
        }

        for i in range(nb_step):
            time = torch.ones((batch_size,), device=self.device) * (1 - i / nb_step)
            batch["time_stamp"] = time

            score_output = self(batch)

            sigma_t = torch.exp(
                time.unsqueeze(1).unsqueeze(1) * math.log(20.0 / 0.001)
            ) * 0.001

            proba_inverse = self.ref_Q[edges_elements, :] * sigma_t * score_output

            idx_tensor = edges_elements.unsqueeze(-1)
            zeros = torch.zeros_like(idx_tensor, dtype=score_output.dtype)
            proba_inverse.scatter_(dim=2, index=idx_tensor, src=zeros)

            norm_vals = -torch.sum(proba_inverse, dim=2, keepdim=True)
            proba_inverse.scatter_(dim=2, index=idx_tensor, src=norm_vals)

            proba_inverse = (1 / nb_step) * proba_inverse

            ones = torch.ones_like(idx_tensor, dtype=score_output.dtype)
            proba_inverse.scatter_add_(dim=2, index=idx_tensor, src=ones)
            proba_inverse = torch.clamp(proba_inverse, min=0.0, max=1.0)

            # Renormalize
            threshold = 1e-6
            filtered = proba_inverse.clone()
            filtered[filtered < threshold] = 0.0
            s = filtered.sum(dim=-1, keepdim=True)
            s[s == 0] = 1.0
            renormalized = filtered / s

            sampler = torch.distributions.categorical.Categorical(probs=renormalized)
            edges_elements = sampler.sample().long()
            batch["noisy_edges"] = edges_elements

        if log:
            self._plot_graphs(edges_elements, self.nb_max_node)

        self.train()
        return edges_elements

    def _plot_graphs(self, output, nb_max_node):
        """Visualize generated graphs."""
        batch_size = output.shape[0]
        nb_planar = 0

        for batch_idx in range(batch_size):
            U = output[batch_idx, ::2].long().cpu().numpy()
            V = output[batch_idx, 1::2].long().cpu().numpy()

            G = nx.Graph()
            for i in range(U.shape[0]):
                if U[i] < nb_max_node and V[i] < nb_max_node:
                    G.add_edge(U[i], V[i])

            if G.number_of_nodes() == 0:
                continue

            is_planar, _ = nx.check_planarity(G)
            if is_planar:
                nb_planar += 1
            print(f"Graph {batch_idx} - planar: {is_planar}")

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
                G, nx.spring_layout(G, seed=42),
                font_size=5, node_size=60, with_labels=False,
                node_color=node_color, cmap="coolwarm", vmin=vmin, vmax=vmax,
                edge_color="grey",
            )

            name_img = f"graph_visu/graph_score_epoch_{self.epoch_current}_{batch_idx}.png"
            os.makedirs("graph_visu", exist_ok=True)
            plt.savefig(name_img)
            plt.clf()
            plt.close()

        print(f"Planar proportion: {nb_planar / batch_size}")

    def configure_optimizers(self):
        return ForeachSOAP(
            self.parameters(), lr=1e-3, foreach=False, warmup_steps=200,
        )


def calculate_exp_neg_F(t_batch, A=20, B=0.001):
    """Compute exp(-F(t)) where F(t) = (B / ln(A/B)) * ((A/B)^t - 1)."""
    A_t = torch.tensor(A, dtype=t_batch.dtype)
    B_t = torch.tensor(B, dtype=t_batch.dtype)
    A_over_B = A_t / B_t
    F_t = (B_t / torch.log(A_over_B)) * (torch.pow(A_over_B, t_batch) - 1)
    return torch.exp(-F_t)
