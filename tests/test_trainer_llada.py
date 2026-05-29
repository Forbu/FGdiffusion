"""Tests for the LLaDA trainer."""

import torch
from deepgraphgen.trainers.trainer_llada import TrainerG2PTLLaDA


def test_trainer_init():
    """Test trainer can be instantiated."""
    model = TrainerG2PTLLaDA(
        nb_max_node=10,
        hidden_dim=64,
        nb_layer=2,
        heads=4,
        edges_to_node_ratio=3,
    )
    assert model.nb_max_node == 10
    assert model.mask_token == 12  # nb_max_node + 2


def test_token_masking():
    """Test the masking mechanism."""
    model = TrainerG2PTLLaDA(
        nb_max_node=10,
        hidden_dim=64,
        nb_layer=2,
        heads=4,
        edges_to_node_ratio=3,
    )

    batch_size = 4
    edges = torch.randint(0, 10, (batch_size, 10 * 3 * 2))

    batch, mask = model._token_masking(edges, batch_size, return_mask=True)

    assert batch["noisy_edges"].shape == edges.shape
    assert mask.shape == edges.shape

    # Mask tokens should be nb_max_node + 2
    assert model.mask_token == 12

    # At t=0, nothing should be masked
    batch_t0, mask_t0 = model._token_masking(
        edges, batch_size, time_stamp=torch.ones(batch_size), return_mask=True
    )
    assert (batch_t0["noisy_edges"] == edges).all()

    # At t=0 (all masked), everything should be mask token
    batch_t1, mask_t1 = model._token_masking(
        edges, batch_size, time_stamp=torch.zeros(batch_size), return_mask=True
    )
    assert (batch_t1["noisy_edges"] == model.mask_token).all()
