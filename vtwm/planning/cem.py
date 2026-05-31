"""Cross-Entropy Method planning in VT-WM imagination (paper Algorithm 1).

Goal-conditioned, vision-only objective: cost is the l2 distance between the final
predicted visual latent and the goal latent. Tactile is not used as a goal signal; it
only enters through the initial context to disambiguate contact states.
"""
from __future__ import annotations

from typing import Tuple

import torch

from vtwm.models.predictor import VTWMPredictor


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
) -> Tuple[torch.Tensor, list]:
    """Returns (best_action_sequence (horizon, action_chunk, action_dim), cost_history)."""
    Tc = s_ctx.shape[1]
    mu = torch.zeros(horizon, action_chunk, action_dim, device=device)
    sigma = torch.ones(horizon, action_chunk, action_dim, device=device)
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
        sigma = elite.std(dim=0).clamp_min(1e-3)

        if costs.min().item() < best_cost:
            best_cost = costs.min().item()
            best_action = actions[costs.argmin()].clone()
        cost_history.append(best_cost)

    return best_action, cost_history
