"""
Fisher-Rao geometric primitives for flow matching on the positive sphere.

The Fisher-Rao metric on the probability simplex Δ^{d-1} is isometric to
the round metric on the positive orthant of the sphere S⁺_d via the
embedding p ↦ √p.

This module provides:
- Token-to-sphere mapping (discrete → continuous)
- Geodesic interpolation (slerp)
- Log map and Exp map on the sphere
- Numerically stable helper functions (sinc_inv, safe_acos)

These primitives are used by the Fisher-Rao Flow Matching trainers to
model discrete token generation as continuous flows on S⁺_d.
"""

import math
import torch
from torch.nn import functional as F


# ─── Token ↔ Sphere mapping ────────────────────────────────────────────────


def token_to_sphere(token_indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Map discrete token indices to points on the positive sphere S⁺_d.

    Token i → e_i (standard basis vector, already unit-norm).
    The mask/prior token (index = vocab_size or higher) → 1/√d · 1 (uniform).

    Args:
        token_indices: (...,) integer tensor of token IDs
        vocab_size: number of valid token classes

    Returns:
        (..., vocab_size) float tensor on S⁺_d
    """
    device = token_indices.device
    is_mask = token_indices >= vocab_size
    clamped = token_indices.clamp(max=vocab_size - 1)
    x = F.one_hot(clamped, num_classes=vocab_size).float()
    uniform = torch.ones(vocab_size, device=device) / math.sqrt(vocab_size)
    x[is_mask] = uniform
    return x


def token_to_sphere_data(token_indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Map data tokens to S⁺_d (no mask handling).

    Args:
        token_indices: (...,) integer tensor
        vocab_size: vocabulary size

    Returns:
        (..., vocab_size) float tensor on S⁺_d
    """
    clamped = token_indices.clamp(max=vocab_size - 1)
    return F.one_hot(clamped, num_classes=vocab_size).float()


def sphere_to_token(x: torch.Tensor) -> torch.Tensor:
    """
    Map sphere points back to discrete tokens via argmax.

    x ∈ S⁺_d → token = argmax(x²) since x² recovers the probability.

    Args:
        x: (..., vocab_size) point on S⁺_d

    Returns:
        (...,) integer tensor of token IDs
    """
    return (x ** 2).argmax(dim=-1)


def sample_random_prior(
    shape: tuple, device: torch.device, noise: float = 0.1
) -> torch.Tensor:
    """
    Sample noisy barycenter points on S⁺_d as a stochastic prior.

    Starts from the uniform barycenter 1/√d · 1, adds Gaussian noise
    projected onto the tangent plane, then re-normalizes to the positive sphere.

    Args:
        shape: (..., vocab_size) desired output shape
        device: torch device
        noise: scale of the tangent noise (default 0.1)

    Returns:
        Same shape as input, points on S⁺_d near the barycenter
    """
    vocab_size = shape[-1]
    # Barycenter
    x0 = torch.ones(shape, device=device) / math.sqrt(vocab_size)
    # Gaussian noise in ambient space
    v = torch.randn(shape, device=device) * noise
    # Project onto tangent plane at x0 (orthogonal to x0)
    v = v - (v * x0).sum(dim=-1, keepdim=True) * x0
    # Move along tangent and re-normalize onto positive sphere
    x0 = F.normalize(x0 + v, dim=-1).abs()
    return x0


# ─── Numerically stable helpers ────────────────────────────────────────────


def sinc_inv(psi: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Compute ψ / sin(ψ) with Taylor expansion for small ψ.

    sinc_inv(ψ) = ψ/sin(ψ) → 1 + ψ²/6 + 7ψ⁴/360 as ψ → 0

    Args:
        psi: angle tensor of any shape

    Returns:
        ψ/sin(ψ), numerically stable
    """
    return torch.where(
        psi.abs() < eps,
        1.0 + psi ** 2 / 6.0 + 7.0 * psi.pow(4) / 360.0,
        psi / torch.sin(psi.clamp(min=1e-30)),
    )


def safe_acos(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """arccos clamped to [-1+eps, 1-eps] for numerical stability."""
    return torch.acos(x.clamp(-1.0 + eps, 1.0 - eps))


# ─── Sphere geometry ──────────────────────────────────────────────────────


def geodesic_interp(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Spherical linear interpolation (slerp) between x0 and x1.

    x_t = sin((1-t)θ)/sin(θ) · x0 + sin(tθ)/sin(θ) · x1

    Args:
        x0: (..., d) start point on sphere
        x1: (..., d) end point on sphere
        t: (...) or (..., 1) interpolation parameter in [0, 1]

    Returns:
        (..., d) interpolated point on sphere
    """
    theta = safe_acos((x0 * x1).sum(dim=-1, keepdim=True))  # (..., 1)
    sin_theta = torch.sin(theta).clamp(min=1e-30)

    if t.dim() < x0.dim():
        t = t.unsqueeze(-1)
    coeff_0 = torch.sin((1 - t) * theta) / sin_theta
    coeff_1 = torch.sin(t * theta) / sin_theta

    return coeff_0 * x0 + coeff_1 * x1


def log_map(x_t: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map on the sphere: tangent vector at x_t pointing toward x1.

    log_{x_t}(x1) = (ψ/sin(ψ)) · (x1 - cos(ψ) · x_t)

    where ψ = arccos(x_t · x1) is the geodesic distance.

    Args:
        x_t: (..., d) current point on sphere
        x1: (..., d) target point on sphere

    Returns:
        (..., d) tangent vector at x_t in the direction of x1
    """
    psi = safe_acos((x_t * x1).sum(dim=-1, keepdim=True))
    sinc = sinc_inv(psi)
    return sinc * (x1 - torch.cos(psi) * x_t)


def exp_map(x_t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on the sphere: move from x_t along tangent vector v.

    exp_{x_t}(v) = cos(||v||) · x_t + sin(||v||) · (v / ||v||)

    Args:
        x_t: (..., d) current point on sphere
        v: (..., d) tangent vector at x_t

    Returns:
        (..., d) new point on sphere
    """
    norm_v = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-30)
    unit_v = v / norm_v
    return torch.cos(norm_v) * x_t + torch.sin(norm_v) * unit_v


def endpoint_velocity(
    x_t: torch.Tensor, x1_hat: torch.Tensor, t: torch.Tensor, eps: float = 1e-2
) -> torch.Tensor:
    """
    Compute the endpoint-parametrized velocity on the sphere.

    u_t(x_t | x1_hat) = log_{x_t}(x1_hat) / (1 - t)

    When t → 1: x_t ≈ x1 so log_{x_t}(x1) → 0, and (1-t) → 0.
    The ratio is well-defined but numerically 0/0. Clamping (1-t) ≥ eps
    makes this 0/eps ≈ 0 near t=1, which is correct (no signal needed).

    Args:
        x_t: (..., d) current point on sphere
        x1_hat: (..., d) predicted endpoint on sphere
        t: (...) time in [0, 1]
        eps: minimum value for (1-t) to prevent singularity

    Returns:
        (..., d) velocity tangent vector at x_t
    """
    log = log_map(x_t, x1_hat)
    one_minus_t = (1 - t).clamp(min=eps)
    if one_minus_t.dim() < log.dim():
        one_minus_t = one_minus_t.unsqueeze(-1)
    return log / one_minus_t


def euler_step(
    x_t: torch.Tensor, x1_hat: torch.Tensor, t: float, dt: float
) -> torch.Tensor:
    """
    One Euler integration step on the Fisher-Rao sphere.

    1. Compute velocity: u_t = log_{x_t}(x1_hat) / (1 - t)
    2. Step: x_{t+dt} = exp_{x_t}(u_t · dt)
    3. Re-normalize for numerical stability

    Args:
        x_t: (..., d) current point on sphere
        x1_hat: (..., d) predicted endpoint
        t: current time (float)
        dt: step size

    Returns:
        (..., d) updated point on sphere
    """
    t_tensor = torch.tensor(t, device=x_t.device)
    one_minus_t = max(1.0 - t, 1e-2)

    log = log_map(x_t, x1_hat)
    u_t = log / one_minus_t
    step = u_t * dt
    x_next = exp_map(x_t, step)
    return F.normalize(x_next, dim=-1)
