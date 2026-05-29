"""Tests for utility functions."""

import torch
from deepgraphgen.utils import stratified_uniform_sample


def test_stratified_sampling():
    """Test stratified uniform sampling produces values in [0, 1)."""
    batch_size = 100
    samples = stratified_uniform_sample(batch_size)

    assert samples.shape == (batch_size,)
    assert samples.min() >= 0.0
    assert samples.max() < 1.0

    # Check stratification: each sample should fall in its stratum
    for i, s in enumerate(samples):
        assert i / batch_size <= s.item() < (i + 1) / batch_size


def test_stratified_sampling_device():
    """Test sampling works on different devices."""
    samples_cpu = stratified_uniform_sample(32, device=torch.device("cpu"))
    assert samples_cpu.device == torch.device("cpu")
