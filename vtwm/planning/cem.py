"""Cross-Entropy Method planning in VT-WM imagination (paper Algorithm 1).

Goal-conditioned, vision-only objective: cost is the l2 distance between the final
predicted visual latent and the goal latent. Tactile is not used as a goal signal; it
only enters through the initial context to disambiguate contact states.
"""
from __future__ import annotations

from typing import Tuple

import torch

from vtwm.models.predictor import VTWMPredictor


def qpos_sigma_ramp(
    horizon: int,
    action_chunk: int,
    action_dim: int,
    sigma_step: float,
    frame_stride: int,
    device: str = "cuda",
    chunk_accumulate: bool = True,
    floor: float = 1e-4,
) -> torch.Tensor:
    """Per-position CEM init std for an ABSOLUTE joint-qpos action space.

    The search is seeded at the current pose, so the std needed to reach action position
    (keyframe h, chunk index c) scales with how far that position drifts from the seed. By the
    dataset convention (`univtac_dataset.py`) that position is `joint[k0 + h*frame_stride + c + 1]`,
    i.e. `h*frame_stride + c + 1` source (~60Hz) steps out; the qpos deviation grows ~linearly with
    that count (verified on lift_bottle: keyframe |delta| == frame_stride x per-step |delta|), so
    `sigma(h,c) = sigma_step * steps_out`. A single per-step scalar `sigma_step` is ramped along
    BOTH the chunk (c) and horizon (h) axes.

    `chunk_accumulate=False` drops the within-chunk `c+1` term for a seed that already carries the
    chunk structure (the train-val / openloop seed is the first GT chunk `a[b,0]`, so the c offset
    cancels and only the horizon term `h*frame_stride` survives). Returns (horizon, chunk, dim).
    """
    h = torch.arange(horizon, device=device, dtype=torch.float32)
    c = torch.arange(action_chunk, device=device, dtype=torch.float32)
    if chunk_accumulate:
        steps = h[:, None] * frame_stride + (c[None, :] + 1.0)        # (H, chunk)
    else:
        steps = (h[:, None] * frame_stride).expand(horizon, action_chunk)
    sigma = (sigma_step * steps).clamp_min(floor)
    return sigma[..., None].expand(horizon, action_chunk, action_dim).contiguous()


@torch.no_grad()
def cem_plan(
    predictor: VTWMPredictor,
    s_ctx: torch.Tensor,        # (1, Tc, 16, 12, 20) initial visual context
    t_ctx: torch.Tensor,        # (1, Tc, 4, 196, 768) initial tactile context
    s_goal: torch.Tensor,       # (1, 16, 12, 20) goal visual latent
    horizon: int,               # H * f action steps
    action_chunk: int = 5,
    action_dim: int = 7,
    particles: int = 36,
    iters: int = 10,
    elites: int = 5,
    max_context: int = 9,
    device: str = "cuda",
    mu_init: torch.Tensor | None = None,
    sigma_init: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, list]:
    """Returns (best_action_sequence (horizon, action_chunk, action_dim), cost_history).

    `mu_init`/`sigma_init` seed the search distribution. They default to the
    zero-mean/unit-std prior (right for normalized delta actions); for an absolute action
    space (e.g. joint qpos) pass the current state broadcast over the horizon as `mu_init`
    and a small per-dim `sigma_init`. Each may be (horizon, chunk, dim) or broadcastable to it.
    """
    Tc = s_ctx.shape[1]
    base = torch.zeros(horizon, action_chunk, action_dim, device=device)
    mu = base + (mu_init.to(device) if mu_init is not None else 0.0)
    sigma = (sigma_init.to(device) if sigma_init is not None
             else torch.ones(horizon, action_chunk, action_dim, device=device))
    sigma = (base + sigma).clone()
    best_action = None
    best_cost = float("inf")
    cost_history = []

    for _ in range(iters):
        # Sample action particles ~ N(mu, sigma^2): (P, horizon, chunk, dim)
        noise = torch.randn(particles, horizon, action_chunk, action_dim, device=device)
        actions = mu[None] + sigma[None] * noise

        # Build per-particle context + action stream. Context actions (frames before the
        # horizon) are zeros; planned actions drive frames Tc-1 .. Tc-1+horizon-1.
        s_rep = s_ctx.expand(particles, -1, -1, -1, -1)
        t_rep = t_ctx.expand(particles, -1, -1, -1, -1)
        ctx_actions = torch.zeros(particles, Tc - 1, action_chunk, action_dim, device=device)
        full_actions = torch.cat([ctx_actions, actions], dim=1)  # (P, Tc-1+horizon, chunk, dim)

        pred_s, _ = predictor.rollout(s_rep, t_rep, full_actions, horizon=horizon, max_context=max_context)
        final_s = pred_s[:, -1]  # (P, 16, 12, 20)
        costs = (final_s - s_goal).flatten(1).pow(2).sum(dim=1).sqrt()  # l2

        topk = torch.topk(costs, k=min(elites, particles), largest=False).indices
        elite = actions[topk]
        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0).clamp_min(1e-4)  # floor below the ~1e-3 per-step qpos delta scale

        if costs.min().item() < best_cost:
            best_cost = costs.min().item()
            best_action = actions[costs.argmin()].clone()
        cost_history.append(best_cost)

    return best_action, cost_history
