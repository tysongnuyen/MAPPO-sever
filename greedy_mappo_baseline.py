"""Grid-greedy placement followed by the existing attention-only MAPPO allocator.

The planner uses physical QoS feasibility, never the learned allocation policy.
Its table is private to placement: allocation inputs/masks/losses are unchanged.
"""
from __future__ import annotations

import numpy as np

import env as physics
from mappo_attention_baseline import AttentionMAPPOBaseline

PLANNER_VERSION = "grid_qos_coverage_then_cost_xyz_v1"


def candidate_service_costs(environment):
    """Minimum one-UE cost at each candidate, infinity if no grid action works.

    This is an individual feasibility estimate, NOT a simultaneous allocation
    solution. Dynamic/shared budgets are enforced later by the unchanged env.
    Use the actual double-precision UE coordinates and env.gain, not normalized
    neural observations. Chunk the rate grid to bound temporary memory.
    """
    if environment.coordination_mode != "none":
        raise ValueError("Greedy MAPPO requires coordination_mode='none'")
    if len(environment.ues) != environment.num_ues:
        raise ValueError("Reset the environment before planning")
    if not np.all(environment.max_power_w_per_uav == environment.max_power_w_per_uav[0]):
        raise ValueError("This planner assumes equal UAV power budgets")
    prbs = np.repeat(environment.prb_levels, environment.num_power_levels)
    powers = np.tile(environment.power_levels_w, environment.max_prbs)
    resource_ok = ((prbs <= environment.total_max_prbs)
                   & (powers <= environment.max_power_w_per_uav[0] + 1e-12))
    action_costs = (environment.power_weight * powers / environment.power_penalty_denominator_w
                    + environment.prb_weight * prbs / environment.total_max_prbs)
    qos = np.array([ue.require_mbps for ue in environment.ues], dtype=np.float64)
    costs = np.full((environment.num_solutions, environment.num_ues), np.inf)
    bandwidth = prbs * physics.phys.Bandwidth_BRP
    noise = physics.phys.k_B * physics.phys.T * bandwidth * physics.phys.noise_figure_linear
    for start in range(0, environment.num_solutions, 128):
        positions = environment.solutions[start:start + 128]
        covered = np.array([[physics.dis_2d(p.x_u, p.y_u, u.x_m, u.y_m) <= p.r_u
                             for u in environment.ues] for p in positions])
        gains = np.array([[physics.gain(p, u) if covered[i, j] else 0.
                           for j, u in enumerate(environment.ues)]
                          for i, p in enumerate(positions)])
        rates = bandwidth * np.log2(1. + gains[..., None] * powers / noise) / 1e6
        # Match scalar env checks even at a floating-point QoS boundary.
        for i, j, a in np.argwhere(np.abs(rates - qos[None, :, None]) < 1e-10):
            rates[i, j, a] = physics.data_rate_mbps(float(powers[a]), int(prbs[a]),
                                                   float(gains[i, j]))[0]
        feasible = covered[..., None] & resource_ok & (rates >= qos[None, :, None])
        costs[start:start + len(positions)] = np.where(feasible, action_costs, np.inf).min(axis=-1)
    return costs


def select_greedy_candidates(costs, xyz, num_uavs):
    """Maximize new QoS-coverable UEs, then minimize union service cost.

    Exact ties (cost rounded to 12 decimals) use candidate index. Different
    angles at the same xyz cannot be used twice, matching the placement space.
    """
    costs, xyz = np.asarray(costs, dtype=np.float64), np.asarray(xyz)
    if (costs.ndim != 2 or xyz.shape != (len(costs), 3) or costs.shape[1] == 0
            or np.isnan(costs).any() or (costs < 0).any() or not np.isfinite(xyz).all()):
        raise ValueError("Expected nonnegative candidate/UE costs (inf allowed) and finite xyz")
    if num_uavs <= 0 or len(np.unique(xyz, axis=0)) < num_uavs:
        raise ValueError("Not enough distinct xyz positions")
    best_cost = np.full(costs.shape[1], np.inf)
    available = np.ones(len(costs), dtype=bool)
    selected = []
    for _ in range(num_uavs):
        merged = np.minimum(costs, best_cost)
        new_coverage = (np.isfinite(merged) & ~np.isfinite(best_cost)).sum(axis=1)
        ids = np.flatnonzero(available)
        ids = ids[new_coverage[ids] == new_coverage[ids].max()]
        sums = np.where(np.isfinite(merged[ids]), merged[ids], 0.).sum(axis=1).round(12)
        chosen = int(ids[np.argmin(sums)])
        selected.append(chosen)
        best_cost = merged[chosen]
        available[np.all(xyz == xyz[chosen], axis=1)] = False
    return np.asarray(selected, dtype=np.int64)


class GreedyAttentionMAPPO(AttentionMAPPOBaseline):
    """Bind to one env; reset/evaluate/train that env through the shared helpers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.CHECKPOINT_FRAMEWORK = "greedy_placement_attention_allocation_ppo_gae_v1"
        # ponytail: retain unused placement modules for inherited checkpoint
        # compatibility; they are frozen and excluded from active parameter counts.
        self.placement_actors.requires_grad_(False)
        self.placement_critic.requires_grad_(False)
        self._planner_environment = None
        self._planner_key = None
        self._planner_candidates = None

    @classmethod
    def from_env(cls, environment, **kwargs):
        model = super().from_env(environment, **kwargs)
        model._planner_environment = environment
        return model

    def act_placement(self, global_state, actor_observations, *, deterministic=False,
                      store_for_update=True, selection_order=None):
        environment = self._planner_environment
        if environment is None or environment.phase != environment.PLACEMENT_PHASE:
            raise ValueError("Use from_env and reset the bound env before greedy placement")
        if store_for_update:
            raise ValueError("Greedy placement has no PPO rollout; use update_placement=False")
        raw_ues = np.array([(u.x_m, u.y_m, u.require_mbps) for u in environment.ues])
        if not np.array_equal(global_state["global_ues"], raw_ues.astype(np.float32)):
            raise ValueError("Greedy planner was called with a different environment's state")
        order = np.arange(self.num_uavs) if selection_order is None else np.asarray(selection_order)
        if (order.shape != (self.num_uavs,) or not np.issubdtype(order.dtype, np.integer)
                or set(order.tolist()) != set(range(self.num_uavs))):
            raise ValueError("selection_order must permute all UAV IDs")
        key = (raw_ues.tobytes(), environment.candidate_positions.tobytes(),
               environment.power_levels_w.tobytes(), environment.prb_levels.tobytes(),
               environment.power_weight, environment.prb_weight, environment.total_max_prbs,
               environment.power_penalty_denominator_w, environment.max_power_w_per_uav.tobytes())
        if key != self._planner_key:
            costs = candidate_service_costs(environment)
            self._planner_candidates = select_greedy_candidates(
                costs, environment.candidate_positions[:, :3], self.num_uavs)
            self._planner_key = key
        actions = np.empty(self.num_uavs, dtype=np.int64)
        actions[order] = self._planner_candidates
        if not environment.placement_action_space.contains(actions):
            raise RuntimeError("Greedy generated an invalid placement")
        return actions
