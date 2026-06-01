"""
G2PT + Fisher-Rao Flow Matching for discrete graph generation.

Implements continuous flow matching on the Fisher-Rao manifold (positive sphere S⁺_d)
for discrete token generation, using endpoint parametrization.

Key ideas:
- Discrete tokens are mapped to points on S⁺_d via sqrt(one_hot).
- The prior is the uniform distribution 1/sqrt(d) * 1_d (barycenter).
- Geodesic interpolation between prior (x0) and data (x1) via spherical slerp.
- The network predicts x1_hat from x_t (endpoint parametrization).
- Training loss: cross-entropy on the predicted token (most stable),
  velocity matching, or geodesic distance.
- Inference: Euler steps on the sphere using log/exp maps.

The velocity on the sphere:
    u_t(x_t | x_1) = log_{x_t}(x_1) / (1 - t)
where log_{x_t}(x_1) = (ψ/sin(ψ)) · (x_1 - cos(ψ) · x_t), ψ = arccos(x_t · x_1)

The exp map update:
    x_{t+dt} = cos(||v||) · x_t + sin(||v||) · (v / ||v||)
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
try:
    from heavyball import ForeachSOAP as SOAP
except ImportError:
    from heavyball import SOAP

from deepgraphgen.fisher_rao import (
    token_to_sphere,
    token_to_sphere_data,
    sphere_to_token,
    geodesic_interp,
    log_map,
    exp_map,
    endpoint_velocity,
    euler_step,
    safe_acos,
    sinc_inv,
    sample_random_prior,
)


def stratified_uniform_sample(batch_size, device=None, dtype=torch.float32):
    """Generate stratified uniform samples in [0, 1)."""
    if device is None:
        device = torch.device("cpu")
    t = torch.arange(batch_size, device=device, dtype=dtype)
    offsets = torch.rand(batch_size, device=device, dtype=dtype)
    return (t + offsets) / batch_size

torch.set_float32_matmul_precision("medium")


# ═══════════════════════════════════════════════════════════════════════════
#  Fisher-Rao FM for simple (untyped) graph generation
# ═══════════════════════════════════════════════════════════════════════════


class TrainerG2PTFisherFM(pl.LightningModule):
    """
    G2PT trainer with Fisher-Rao Flow Matching on the positive sphere.

    Instead of discrete diffusion (mask/unmask or score-based), this approach
    models the generation process as a continuous flow on the Fisher-Rao manifold
    (positive orthant of the unit sphere S⁺_d).

    Pipeline:
    1. Data tokens → S⁺_d via one-hot (e_i)
    2. Prior = 1/√d · 1 (uniform barycenter)
    3. Geodesic interpolation: x_t = slerp(prior, x_1, t)
    4. Network predicts x_1_hat from (x_t, t) — endpoint parametrization
    5. Inference: Euler steps on the sphere via log/exp maps

    Compatible with the same datasets and evaluation as LLaDA/score trainers.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 384,
        nb_layer: int = 6,
        heads: int = 8,
        nb_max_node: int = 100,
        edges_to_node_ratio: int = 5,
        loss_type: str = "ce",
        inference_steps: int = 128,
        use_time_emb: bool = True,
    ):
        """
        Args:
            vocab_size: number of discrete token classes
            hidden_dim: transformer hidden dimension
            nb_layer: number of transformer layers
            heads: number of attention heads
            nb_max_node: maximum number of nodes
            edges_to_node_ratio: edges per node
            loss_type: "ce" (cross-entropy), "velocity", or "geodesic"
            inference_steps: Euler steps at inference
            use_time_emb: whether to use time conditioning
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.nb_max_node = nb_max_node
        self.edges_to_node_ratio = edges_to_node_ratio
        self.nb_edges = edges_to_node_ratio * nb_max_node
        self.loss_type = loss_type
        self.inference_steps = inference_steps
        self.use_time_emb = use_time_emb
        self.nb_layer = nb_layer

        # Prior: uniform point on S⁺_d
        self.register_buffer(
            "prior",
            torch.ones(vocab_size) / math.sqrt(vocab_size),
        )

        # ─── Network ─────────────────────────────────────────────────

        # Project sphere point (vocab_size) to hidden_dim
        self.input_projection = torch.nn.Linear(vocab_size, hidden_dim)

        # Time embedding (sinusoidal + MLP)
        if use_time_emb:
            self.time_projection = torch.nn.Sequential(
                torch.nn.Linear(1, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )

        # Position embedding: we process nb_edges * 2 tokens (flattened src/dst)
        self.position_embedding = torch.nn.Embedding(
            self.nb_edges * 2, hidden_dim
        )

        # Transformer encoder backbone
        self.model_core = Encoder(dim=hidden_dim, depth=nb_layer, heads=heads)

        # Output head: predict logits over vocabulary
        self.output_head = torch.nn.Linear(hidden_dim, vocab_size)

        self.epoch_current = 0

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict endpoint logits from current sphere point and time.

        Args:
            x_t: (B, L, V) current point on S⁺_d
            t: (B,) time in [0, 1]

        Returns:
            (B, L, V) logits for predicted endpoint x1
        """
        batch_size, seq_len, _ = x_t.shape

        # Project sphere coordinates to hidden dim
        h = self.input_projection(x_t)

        # Add position embeddings
        positions = (
            torch.arange(seq_len, device=x_t.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        h = h + self.position_embedding(positions[:, :seq_len])

        # Add time embedding
        if self.use_time_emb:
            t_emb = self.time_projection(t.unsqueeze(-1))
            h = h + t_emb.unsqueeze(1)

        # Transformer
        h = self.model_core(h)

        # Output logits
        return self.output_head(h)

    def _logits_to_sphere(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert output logits to a point on S⁺_d via sqrt(softmax)."""
        p = F.softmax(logits, dim=-1).clamp(min=1e-8)
        return F.normalize(torch.sqrt(p), dim=-1)

    def training_step(self, batch, batch_idx):
        """
        Training step with Fisher-Rao flow matching.

        1. Map data tokens → sphere x1
        2. Sample t ~ Uniform(0, 1)
        3. Interpolate: x_t = slerp(prior, x1, t)
        4. Network predicts x1_hat from (x_t, t)
        5. Compute loss
        """
        edges_element = batch["edges"]  # (B, num_edges, 2)
        batch_size = edges_element.shape[0]
        edges_flat = edges_element.reshape(batch_size, -1)

        # Map tokens → sphere
        x1 = token_to_sphere_data(edges_flat, self.vocab_size)

        # Random prior on S⁺_d (Dirichlet-based, provides stochasticity)
        x0 = sample_random_prior(
            x1.shape, self.device, noise=0.1
        )

        # Sample time
        t = stratified_uniform_sample(batch_size, device=self.device)

        # Geodesic interpolation
        x_t = geodesic_interp(x0, x1, t.unsqueeze(-1))
        x_t = F.normalize(x_t, dim=-1)

        # Forward: predict endpoint logits
        logits = self.forward(x_t, t)

        # Compute loss
        target_tokens = edges_flat.long()

        if self.loss_type == "ce":
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                target_tokens.reshape(-1),
            )

        elif self.loss_type == "velocity":
            with torch.no_grad():
                u_true = endpoint_velocity(x_t, x1, t)
            x1_hat = self._logits_to_sphere(logits)
            u_pred = endpoint_velocity(x_t, x1_hat, t)
            loss = ((u_pred - u_true) ** 2).sum(-1).mean()

        elif self.loss_type == "geodesic":
            x1_hat = self._logits_to_sphere(logits)
            cos_sim = (x1_hat * x1).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            loss = (torch.acos(cos_sim) ** 2).mean()

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        self.log("train_loss", loss, prog_bar=True)

        if self.global_step % 200 == 0 and self.global_step > 0:
            with torch.no_grad():
                self.generate_graphs(2)

        return loss

    @torch.no_grad()
    def generate_graphs(
        self,
        batch_size: int,
        nb_steps: int | None = None,
        log: bool = True,
    ) -> torch.Tensor:
        """
        Generate graphs via Euler integration on the Fisher-Rao sphere.

        1. x_0 = 1/√d · 1 (uniform prior)
        2. Loop:
           a. Predict x1_hat from (x_t, t)
           b. Velocity: u_t = log_{x_t}(x1_hat) / (1-t)
           c. Update: x_{t+dt} = exp_{x_t}(u_t · dt)
        3. Token = argmax(x_T²)

        Returns:
            (B, num_edges * 2) integer tensor of predicted tokens
        """
        self.eval()
        if nb_steps is None:
            nb_steps = self.inference_steps

        seq_len = self.nb_max_node * self.edges_to_node_ratio * 2

        # Random starting point on S⁺_d (provides diversity across samples)
        x_t = sample_random_prior(
            (batch_size, seq_len, self.vocab_size),
            self.device,
            noise=0.1,
        )

        dt = 1.0 / nb_steps

        for step in range(nb_steps):
            t_val = step / nb_steps
            if t_val >= 1.0 - 1e-5:
                break

            t = torch.full((batch_size,), t_val, device=self.device)

            # Predict endpoint
            logits = self.forward(x_t, t)
            x1_hat = self._logits_to_sphere(logits)

            # Euler step
            x_t = euler_step(x_t, x1_hat, t_val, dt)

        # Decode: square to get probabilities, argmax
        predicted_tokens = sphere_to_token(x_t)

        if log:
            self._plot_graphs(predicted_tokens, self.nb_max_node)

        self.train()
        return predicted_tokens

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
            print(
                f"Graph {batch_idx} - planar: {is_planar}, "
                f"nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}"
            )

            try:
                w, eigvecs = np.linalg.eigh(
                    nx.normalized_laplacian_matrix(G).toarray()
                )
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
                node_color=node_color, cmap="coolwarm",
                vmin=vmin, vmax=vmax, edge_color="grey",
            )

            name_img = (
                f"graph_visu/graph_fisher_fm_epoch_"
                f"{self.epoch_current}_{batch_idx}.png"
            )
            os.makedirs("graph_visu", exist_ok=True)
            plt.savefig(name_img)
            plt.clf()
            plt.close()

        print(f"Planar proportion: {nb_planar / max(batch_size, 1):.2f}")

    def on_train_epoch_end(self):
        if self.epoch_current % 100 == 0:
            with torch.no_grad():
                self.generate_graphs(1)
        self.epoch_current += 1

    def configure_optimizers(self):
        return SOAP(
            self.parameters(), lr=1e-3, foreach=False, warmup_steps=200,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Fisher-Rao FM for Knowledge Graph generation (typed nodes + edges)
# ═══════════════════════════════════════════════════════════════════════════


class TrainerKGFisherFM(pl.LightningModule):
    """
    Fisher-Rao Flow Matching for typed Knowledge Graph generation.

    Extends Fisher-Rao FM to predict three modalities jointly:
    - Edge indices (source, destination node IDs)
    - Edge labels (relation types)
    - Node labels (entity types)

    Each modality lives on its own Fisher-Rao manifold S⁺_{V_k} where V_k
    is the vocabulary size for that modality. The Transformer processes all
    modalities in a unified sequence with positional embeddings.
    """

    def __init__(
        self,
        vocab_size_edges: int,
        vocab_size_edge_labels: int,
        vocab_size_node_labels: int,
        hidden_dim: int = 384,
        nb_layer: int = 6,
        heads: int = 8,
        nb_max_node: int = 64,
        edges_to_node_ratio: int = 5,
        loss_type: str = "ce",
        inference_steps: int = 128,
    ):
        super().__init__()

        self.vocab_size_edges = vocab_size_edges
        self.vocab_size_edge_labels = vocab_size_edge_labels
        self.vocab_size_node_labels = vocab_size_node_labels
        self.hidden_dim = hidden_dim
        self.nb_max_node = nb_max_node
        self.edges_to_node_ratio = edges_to_node_ratio
        self.nb_edges = edges_to_node_ratio * nb_max_node
        self.loss_type = loss_type
        self.inference_steps = inference_steps

        # Priors
        self.register_buffer(
            "prior_edges",
            torch.ones(vocab_size_edges) / math.sqrt(vocab_size_edges),
        )
        self.register_buffer(
            "prior_edge_labels",
            torch.ones(vocab_size_edge_labels) / math.sqrt(vocab_size_edge_labels),
        )
        self.register_buffer(
            "prior_node_labels",
            torch.ones(vocab_size_node_labels) / math.sqrt(vocab_size_node_labels),
        )

        # ─── Network ─────────────────────────────────────────────────

        # Input projections (one per modality)
        self.edge_input_proj = torch.nn.Linear(vocab_size_edges, hidden_dim)
        self.edge_label_input_proj = torch.nn.Linear(vocab_size_edge_labels, hidden_dim)
        self.node_label_input_proj = torch.nn.Linear(vocab_size_node_labels, hidden_dim)

        # Time embedding
        self.time_projection = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        # Position embedding
        nb_positions = nb_max_node + self.nb_edges
        self.position_embedding = torch.nn.Embedding(nb_positions, hidden_dim)

        # Transformer encoder
        self.model_core = Encoder(dim=hidden_dim, depth=nb_layer, heads=heads)

        # Output heads
        self.edge_head_1 = torch.nn.Linear(hidden_dim, vocab_size_edges)
        self.edge_head_2 = torch.nn.Linear(hidden_dim, vocab_size_edges)
        self.edge_label_head = torch.nn.Linear(hidden_dim, vocab_size_edge_labels)
        self.node_label_head = torch.nn.Linear(hidden_dim, vocab_size_node_labels)

        self.epoch_current = 0

    def _build_sequence(
        self,
        edge_x_t: torch.Tensor,  # (B, num_edges, 2, V_edge)
        edge_label_x_t: torch.Tensor,  # (B, num_edges, V_el)
        node_label_x_t: torch.Tensor,  # (B, num_nodes, V_nl)
    ) -> torch.Tensor:
        """Build unified transformer input from all modalities."""
        batch_size = edge_x_t.shape[0]
        num_edges = edge_x_t.shape[1]
        num_nodes = node_label_x_t.shape[1]

        # Edge indices → embed and combine src + dst
        edge_src = self.edge_input_proj(edge_x_t[:, :, 0, :])  # (B, E, H)
        edge_dst = self.edge_input_proj(edge_x_t[:, :, 1, :])  # (B, E, H)
        edge_label_emb = self.edge_label_input_proj(edge_label_x_t)  # (B, E, H)

        # Combine: src + dst + label for each edge position
        edge_seq = edge_src + edge_dst + edge_label_emb

        # Node labels
        node_seq = self.node_label_input_proj(node_label_x_t)  # (B, N, H)

        # Concatenate: [nodes | edges]
        h = torch.cat([node_seq, edge_seq], dim=1)  # (B, N+E, H)
        return h, num_nodes, num_edges

    def forward(
        self,
        edge_x_t: torch.Tensor,
        edge_label_x_t: torch.Tensor,
        node_label_x_t: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Forward pass.

        Args:
            edge_x_t: (B, num_edges, 2, V_edge) edge index sphere points
            edge_label_x_t: (B, num_edges, V_el) edge label sphere points
            node_label_x_t: (B, num_nodes, V_nl) node label sphere points
            t: (B,) time

        Returns:
            Tuple of logits for each modality
        """
        batch_size = edge_x_t.shape[0]

        h, num_nodes, num_edges = self._build_sequence(
            edge_x_t, edge_label_x_t, node_label_x_t
        )

        # Position embedding
        total_len = num_nodes + num_edges
        positions = (
            torch.arange(total_len, device=h.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        h = h + self.position_embedding(positions[:, :total_len])

        # Time embedding
        t_emb = self.time_projection(t.unsqueeze(-1))
        h = h + t_emb.unsqueeze(1)

        # Transformer
        h = self.model_core(h)

        # Decode
        node_out = h[:, :num_nodes]
        edge_out = h[:, num_nodes:]

        return (
            self.edge_head_1(edge_out),       # (B, E, V_edge)
            self.edge_head_2(edge_out),       # (B, E, V_edge)
            self.edge_label_head(edge_out),   # (B, E, V_el)
            self.node_label_head(node_out),   # (B, N, V_nl)
        )

    def _logits_to_sphere(self, logits: torch.Tensor) -> torch.Tensor:
        return F.normalize(torch.sqrt(F.softmax(logits, dim=-1).clamp(min=1e-8)), dim=-1)

    def training_step(self, batch, batch_idx):
        edges = batch["edges"]          # (B, num_edges, 2)
        edge_labels = batch["edges_label"]  # (B, num_edges)
        node_labels = batch["nodes_label"]  # (B, num_nodes)
        batch_size = edges.shape[0]
        num_edges = edges.shape[1]
        num_nodes = node_labels.shape[1]

        # Flatten edges for sphere mapping
        edges_flat = edges.reshape(batch_size, -1)  # (B, E*2)

        # Map tokens → sphere
        x1_edges = token_to_sphere_data(edges_flat, self.vocab_size_edges)
        x1_el = token_to_sphere_data(edge_labels, self.vocab_size_edge_labels)
        x1_nl = token_to_sphere_data(node_labels, self.vocab_size_node_labels)

        # Priors (random on S⁺_d for stochastic training)
        x0_edges = sample_random_prior(x1_edges.shape, self.device, noise=0.1)
        x0_el = sample_random_prior(x1_el.shape, self.device, noise=0.1)
        x0_nl = sample_random_prior(x1_nl.shape, self.device, noise=0.1)

        # Sample time
        t = stratified_uniform_sample(batch_size, device=self.device)
        t_unsq = t.unsqueeze(-1)

        # Interpolate
        x_t_edges = F.normalize(geodesic_interp(x0_edges, x1_edges, t_unsq), dim=-1)
        x_t_el = F.normalize(geodesic_interp(x0_el, x1_el, t_unsq), dim=-1)
        x_t_nl = F.normalize(geodesic_interp(x0_nl, x1_nl, t_unsq), dim=-1)

        # Reshape edges: (B, E*2, V) → (B, E, 2, V)
        x_t_edges_2d = x_t_edges.reshape(batch_size, num_edges, 2, self.vocab_size_edges)

        # Forward
        el1, el2, el_logits, nl_logits = self.forward(
            x_t_edges_2d, x_t_el, x_t_nl, t
        )

        # Edge logits flat
        edge_logits_flat = torch.stack([el1, el2], dim=-1).reshape(
            batch_size, num_edges * 2, self.vocab_size_edges
        )

        # CE losses
        loss_edges = F.cross_entropy(
            edge_logits_flat.reshape(-1, self.vocab_size_edges),
            edges_flat.reshape(-1).long(),
        )
        loss_el = F.cross_entropy(
            el_logits.reshape(-1, self.vocab_size_edge_labels),
            edge_labels.reshape(-1).long(),
        )
        loss_nl = F.cross_entropy(
            nl_logits.reshape(-1, self.vocab_size_node_labels),
            node_labels.reshape(-1).long(),
        )

        loss = loss_edges + loss_el + loss_nl

        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_edges", loss_edges)
        self.log("loss_el", loss_el)
        self.log("loss_nl", loss_nl)

        if self.global_step % 200 == 0 and self.global_step > 0:
            with torch.no_grad():
                self.generate_graphs(2)

        return loss

    @torch.no_grad()
    def generate_graphs(self, batch_size: int, nb_steps: int | None = None, log: bool = True):
        """Generate typed KGs via Euler integration on Fisher-Rao manifolds."""
        self.eval()
        if nb_steps is None:
            nb_steps = self.inference_steps

        num_edges = self.nb_edges
        num_nodes = self.nb_max_node

        # Random starting points on S⁺_d (provides diversity)
        x_t_edges_flat = sample_random_prior(
            (batch_size, num_edges * 2, self.vocab_size_edges),
            self.device, noise=0.1,
        )
        x_t_edges_2d = x_t_edges_flat.reshape(
            batch_size, num_edges, 2, self.vocab_size_edges
        )

        x_t_el = sample_random_prior(
            (batch_size, num_edges, self.vocab_size_edge_labels),
            self.device, noise=0.1,
        )

        x_t_nl = sample_random_prior(
            (batch_size, num_nodes, self.vocab_size_node_labels),
            self.device, noise=0.1,
        )

        dt = 1.0 / nb_steps

        for step in range(nb_steps):
            t_val = step / nb_steps
            if t_val >= 1.0 - 1e-5:
                break

            t = torch.full((batch_size,), t_val, device=self.device)

            # Forward
            el1, el2, el_logits, nl_logits = self.forward(
                x_t_edges_2d, x_t_el, x_t_nl, t
            )

            # Predicted endpoints on sphere
            x1_hat_e1 = self._logits_to_sphere(el1)
            x1_hat_e2 = self._logits_to_sphere(el2)
            x1_hat_el = self._logits_to_sphere(el_logits)
            x1_hat_nl = self._logits_to_sphere(nl_logits)

            # Euler step for each modality
            x_t_e1_new = euler_step(x_t_edges_2d[:, :, 0, :], x1_hat_e1, t_val, dt)
            x_t_e2_new = euler_step(x_t_edges_2d[:, :, 1, :], x1_hat_e2, t_val, dt)
            x_t_edges_2d = torch.stack([x_t_e1_new, x_t_e2_new], dim=2)

            x_t_el = euler_step(x_t_el, x1_hat_el, t_val, dt)
            x_t_nl = euler_step(x_t_nl, x1_hat_nl, t_val, dt)

        # Decode
        x_t_flat = x_t_edges_2d.reshape(batch_size, -1, self.vocab_size_edges)
        pred_edges = sphere_to_token(x_t_flat)
        pred_el = sphere_to_token(x_t_el)
        pred_nl = sphere_to_token(x_t_nl)

        if log:
            self._plot_kg(pred_edges, pred_el, pred_nl)

        self.train()
        return {
            "edges": pred_edges,
            "edge_labels": pred_el,
            "node_labels": pred_nl,
        }

    def _plot_kg(self, edges, edge_labels, node_labels):
        """Visualize generated KGs."""
        batch_size = edges.shape[0]
        for b in range(batch_size):
            U = edges[b, ::2].cpu().numpy()
            V = edges[b, 1::2].cpu().numpy()
            G = nx.Graph()
            for i in range(len(U)):
                if U[i] < self.nb_max_node and V[i] < self.nb_max_node:
                    G.add_edge(int(U[i]), int(V[i]))
                    G.nodes[int(U[i])]["label"] = node_labels[b, U[i]].item()
                    G.nodes[int(V[i])]["label"] = node_labels[b, V[i]].item()

            if G.number_of_nodes() == 0:
                continue

            plt.figure(figsize=(10, 10))
            nx.draw(
                G, nx.spring_layout(G, seed=42),
                font_size=5, node_size=100, with_labels=False,
                node_color=node_labels[b].cpu().numpy(),
                cmap=plt.cm.coolwarm, edge_color="grey",
            )
            name_img = f"graph_visu/kg_fisher_fm_epoch_{self.epoch_current}_{b}.png"
            os.makedirs("graph_visu", exist_ok=True)
            plt.savefig(name_img)
            plt.clf()
            plt.close()

    def on_train_epoch_end(self):
        if self.epoch_current % 100 == 0:
            with torch.no_grad():
                self.generate_graphs(1)
        self.epoch_current += 1

    def configure_optimizers(self):
        return SOAP(
            self.parameters(), lr=1e-3, foreach=False, warmup_steps=200,
        )
