"""
Tests for Fisher-Rao geometric primitives and trainers.

Verifies:
- Sphere mapping (token → sphere → token roundtrip)
- Geodesic interpolation endpoints
- Log/exp map consistency
- Euler step stability
- Basic forward/backward pass through Fisher-FM trainers
"""

import math
import pytest
import torch
from torch.nn import functional as F

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
)


class TestSphereMapping:
    """Test token ↔ sphere point conversions."""

    def test_one_hot_on_sphere(self):
        """One-hot tokens should already be unit-norm (on S⁺_d)."""
        tokens = torch.tensor([0, 1, 2, 3])
        x = token_to_sphere_data(tokens, vocab_size=5)
        norms = torch.norm(x, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-6)

    def test_roundtrip_data_tokens(self):
        """token → sphere → token should be identity for data tokens."""
        tokens = torch.tensor([[0, 3, 1], [2, 0, 4]])
        x = token_to_sphere_data(tokens, vocab_size=5)
        recovered = sphere_to_token(x)
        assert torch.equal(tokens, recovered)

    def test_mask_to_uniform(self):
        """Mask/pad tokens should map to the uniform point 1/√d."""
        vocab_size = 5
        tokens = torch.tensor([vocab_size, vocab_size + 1])
        x = token_to_sphere(tokens, vocab_size)
        expected = 1.0 / math.sqrt(vocab_size)
        assert torch.allclose(x, torch.full_like(x, expected), atol=1e-6)
        # Should be unit norm
        norms = torch.norm(x, dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-6)

    def test_roundtrip_with_mask(self):
        """Mask tokens should NOT roundtrip to the original (they become uniform)."""
        vocab_size = 5
        tokens = torch.tensor([0, vocab_size, 2])
        x = token_to_sphere(tokens, vocab_size)
        recovered = sphere_to_token(x)
        # Only non-mask tokens roundtrip
        assert recovered[0].item() == 0
        assert recovered[2].item() == 2
        # Mask token maps to uniform → argmax of uniform (could be any)


class TestGeodesicInterp:
    """Test spherical geodesic interpolation."""

    def test_endpoints(self):
        """slerp at t=0 should give x0, slerp at t=1 should give x1."""
        vocab_size = 5
        x0 = torch.tensor([[1.0, 0, 0, 0, 0]])
        x1 = torch.tensor([[0, 1.0, 0, 0, 0]])

        xt_0 = geodesic_interp(x0, x1, torch.tensor([[0.0]]))
        xt_1 = geodesic_interp(x0, x1, torch.tensor([[1.0]]))

        assert torch.allclose(xt_0, x0, atol=1e-5)
        assert torch.allclose(xt_1, x1, atol=1e-5)

    def test_midpoint_on_sphere(self):
        """slerp midpoint should be unit-norm."""
        x0 = torch.tensor([[1.0, 0, 0, 0, 0]])
        x1 = torch.tensor([[0, 1.0, 0, 0, 0]])
        xt = geodesic_interp(x0, x1, torch.tensor([[0.5]]))
        norm = torch.norm(xt, dim=-1)
        assert torch.allclose(norm, torch.ones(1), atol=1e-5)

    def test_batched(self):
        """Should work with batch dimensions."""
        B, L, V = 4, 10, 20
        x0 = F.normalize(torch.randn(B, L, V), dim=-1).abs()  # positive sphere
        x1 = F.normalize(torch.randn(B, L, V), dim=-1).abs()
        x0 = F.normalize(x0, dim=-1)
        x1 = F.normalize(x1, dim=-1)
        t = torch.rand(B, 1)

        xt = geodesic_interp(x0, x1, t)
        assert xt.shape == (B, L, V)
        norms = torch.norm(xt, dim=-1)
        assert torch.allclose(norms, torch.ones(B, L), atol=1e-4)


class TestLogExpMap:
    """Test log map and exp map on the sphere."""

    def test_log_exp_roundtrip(self):
        """exp_{x_t}(log_{x_t}(x1)) should give x1."""
        x_t = torch.tensor([[1.0, 0, 0, 0, 0]])
        x1 = torch.tensor([[0, 1.0, 0, 0, 0]])

        v = log_map(x_t, x1)
        x1_reconstructed = exp_map(x_t, v)

        assert torch.allclose(x1_reconstructed, x1, atol=1e-5)

    def test_log_map_zero_distance(self):
        """log_{x}(x) should be ~zero."""
        x = torch.tensor([[0.6, 0.8, 0, 0, 0]])
        x = F.normalize(x, dim=-1)
        v = log_map(x, x)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-3)

    def test_exp_map_preserves_norm(self):
        """exp map result should be unit-norm."""
        x_t = torch.tensor([[1.0, 0, 0, 0, 0]])
        v = torch.tensor([[0.3, 0.7, 0.1, 0, 0]])  # tangent vector
        v = v - (v * x_t).sum(-1, keepdim=True) * x_t  # project to tangent plane

        x_next = exp_map(x_t, v)
        norm = torch.norm(x_next, dim=-1)
        assert torch.allclose(norm, torch.ones(1), atol=1e-5)

    def test_batched_log_exp(self):
        """Batched log/exp map roundtrip."""
        B, L, V = 4, 10, 20
        x_t = F.normalize(torch.randn(B, L, V), dim=-1)
        x1 = F.normalize(torch.randn(B, L, V), dim=-1)
        x_t = x_t.abs()
        x1 = x1.abs()
        x_t = F.normalize(x_t, dim=-1)
        x1 = F.normalize(x1, dim=-1)

        v = log_map(x_t, x1)
        x1_rec = exp_map(x_t, v)

        cos_sim = (x1_rec * x1).sum(-1)
        assert torch.allclose(cos_sim, torch.ones(B, L), atol=1e-4)


class TestNumericalStability:
    """Test edge cases for numerical stability."""

    def test_sinc_inv_small_psi(self):
        """sinc_inv should handle psi → 0."""
        psi = torch.tensor([0.0, 1e-8, 1e-10, 0.01])
        result = sinc_inv(psi)
        assert torch.isfinite(result).all()
        # At psi=0, should be ~1
        assert torch.allclose(result[0:1], torch.tensor([1.0]), atol=1e-3)

    def test_safe_acos_boundary(self):
        """safe_acos should handle values at ±1."""
        x = torch.tensor([1.0, -1.0, 1.0 + 1e-4, -1.0 - 1e-4])
        result = safe_acos(x)
        assert torch.isfinite(result).all()
        # safe_acos clamps to ±(1-eps), so acos(1) = acos(1-eps) ≈ small
        assert result[0].item() < 0.01
        assert result[1].item() > math.pi - 0.01


class TestEulerStep:
    """Test Euler integration on the sphere."""

    def test_step_preserves_norm(self):
        """Each Euler step should keep the point on the sphere."""
        B, L, V = 2, 5, 10
        x_t = F.normalize(torch.rand(B, L, V), dim=-1)
        x1_hat = F.normalize(torch.rand(B, L, V), dim=-1)

        x_next = euler_step(x_t, x1_hat, t=0.5, dt=0.1)
        norms = torch.norm(x_next, dim=-1)
        assert torch.allclose(norms, torch.ones(B, L), atol=1e-4)

    def test_full_trajectory_reaches_target(self):
        """Multiple Euler steps from t=0 to t=1 should approach x1."""
        torch.manual_seed(42)
        V = 10
        x0 = torch.ones(1, 1, V) / math.sqrt(V)
        x0 = F.normalize(x0, dim=-1)
        x1 = F.normalize(torch.rand(1, 1, V), dim=-1)

        x_t = x0.clone()
        nb_steps = 200
        dt = 1.0 / nb_steps

        for step in range(nb_steps):
            t_val = step / nb_steps
            if t_val >= 1.0 - 1e-5:
                break
            x_t = euler_step(x_t, x1, t_val, dt)

        # Should be close to x1
        cos_sim = (x_t * x1).sum(dim=-1)
        assert cos_sim.item() > 0.95, f"Expected cos_sim > 0.95, got {cos_sim.item()}"


class TestTrainerG2PTFisherFM:
    """Smoke tests for the Fisher-Rao FM trainer."""

    @pytest.fixture
    def model(self):
        from deepgraphgen.trainers.trainer_fisher_fm import TrainerG2PTFisherFM
        return TrainerG2PTFisherFM(
            vocab_size=10,
            hidden_dim=32,
            nb_layer=1,
            heads=2,
            nb_max_node=4,
            edges_to_node_ratio=2,
            loss_type="ce",
            inference_steps=10,
        )

    def test_forward_shape(self, model):
        """Forward pass should output correct shape."""
        B = 3
        seq_len = model.nb_max_node * model.edges_to_node_ratio * 2  # 4*2*2=16
        V = 10
        x_t = F.normalize(torch.rand(B, seq_len, V), dim=-1)
        t = torch.rand(B)

        logits = model.forward(x_t, t)
        assert logits.shape == (B, seq_len, V)

    def test_training_step(self, model):
        """Training step should return a scalar loss."""
        B, num_edges = 2, model.nb_edges  # 4*2=8 edges
        edges = torch.randint(0, 8, (B, num_edges, 2))
        batch = {"edges": edges}

        loss = model.training_step(batch, 0)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_generate(self, model):
        """Generation should return correct shape."""
        model.eval()
        output = model.generate_graphs(batch_size=2, nb_steps=5, log=False)
        expected_len = model.nb_max_node * model.edges_to_node_ratio * 2
        assert output.shape == (2, expected_len)


class TestTrainerKGFisherFM:
    """Smoke tests for the KG Fisher-Rao FM trainer."""

    @pytest.fixture
    def model(self):
        from deepgraphgen.trainers.trainer_fisher_fm import TrainerKGFisherFM
        return TrainerKGFisherFM(
            vocab_size_edges=10,
            vocab_size_edge_labels=5,
            vocab_size_node_labels=5,
            hidden_dim=32,
            nb_layer=1,
            heads=2,
            nb_max_node=4,
            edges_to_node_ratio=2,
            loss_type="ce",
            inference_steps=10,
        )

    def test_training_step(self, model):
        """KG training step should return scalar loss."""
        B, num_edges, num_nodes = 2, 8, 4
        batch = {
            "edges": torch.randint(0, 7, (B, num_edges, 2)),
            "edges_label": torch.randint(0, 4, (B, num_edges)),
            "nodes_label": torch.randint(0, 4, (B, num_nodes)),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_generate(self, model):
        """KG generation should return all modalities."""
        model.eval()
        result = model.generate_graphs(batch_size=2, nb_steps=5, log=False)
        assert "edges" in result
        assert "edge_labels" in result
        assert "node_labels" in result
        assert result["edges"].shape[0] == 2
