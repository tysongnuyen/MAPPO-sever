"""Attention MAPPO: sampled PPO/GAE, no local bypass, PopArt or cost table.

Reuse the first baseline's local observations, attention, hierarchical action
heads and training losses. Critics receive global physical state only.
"""
from types import SimpleNamespace

import torch
from torch import nn

from local_mappo_baseline import LocalMAPPOBaseline
from strict_local_independent_mappo import (
    CentralizedGlobalCritic, LocalAutoregressiveAllocationActor,
)


class AttentionAllocationActor(LocalAutoregressiveAllocationActor):
    def encode(self, observation):
        local_context = self.local_ue_encoder(
            observation["local_ue_features"], observation["local_ue_mask"])
        # Keep current QoS/link/resources as inputs to the mixer. Remove only
        # the direct addition of vector_context AFTER that mixer.
        local_vector = torch.cat((
            observation["own_uav_features"][..., 5:7],
            observation["current_ue_features"][..., 2:3],
            observation["current_link_features"],
            observation["agent_id_one_hot"],
        ), dim=-1)
        vector_context = self.local_vector_encoder(local_vector)
        return self.context_encoder(torch.cat((local_context, vector_context), dim=-1))


class AttentionPlacementCritic(CentralizedGlobalCritic):
    # Identity compatibility hooks for the existing placement PPO loop.
    # There are no PopArt buffers, target standardization or weight rescaling.
    normalized_value = CentralizedGlobalCritic.forward

    def update_popart(self, targets):
        pass

    def normalize_popart_targets(self, targets):
        return targets

    @property
    def popart_head(self):
        return SimpleNamespace(mean=0.0, std=1.0)


class AttentionMAPPOBaseline(LocalMAPPOBaseline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.CHECKPOINT_FRAMEWORK = "local_mappo_attention_no_bypass_no_popart_no_cost_ppo_gae_v2"
        self.allocation_actor_feature_version = "compact_qos_link_mixed_no_bypass_v1"
        self.allocation_critic_feature_version = "global_attention_no_cost_v1"
        # Preserve the initial weights and RNG of the architecture-matched
        # baseline. Replacement modules are constructed on CPU before .to().
        with torch.random.fork_rng(devices=[]):
            actors = nn.ModuleList([
                AttentionAllocationActor(
                    self.num_uavs, self.max_prbs, self.coordination_feature_dim,
                    self.embed_dim, self.num_heads, num_power_levels=self.num_power_levels)
                for _ in range(self.num_uavs)])
            actors.load_state_dict(self.allocation_actors.state_dict())
            critic = AttentionPlacementCritic(self.embed_dim, self.num_heads, popart_beta=None)
            state = self.placement_critic.state_dict()
            # The only removed state is the three PopArt statistics buffers.
            for name in ("mean", "std", "sample_count"):
                del state[f"value_head.4.{name}"]
            critic.load_state_dict(state)
            allocation_critic = CentralizedGlobalCritic(
                self.embed_dim, self.num_heads, include_candidate_features=False,
                allocation_cost_dim=0)
            allocation_state = self.allocation_critic.state_dict()
            # Costs were the final edge features. Preserve the initial weights
            # of every retained input/module, removing just those columns.
            input_width = allocation_critic.relation_encoder.edge_encoder[0].in_features
            key = "relation_encoder.edge_encoder.0.weight"
            allocation_state[key] = allocation_state[key][:, :input_width]
            allocation_critic.load_state_dict(allocation_state)
        actor_lr = self.allocation_actor_optimizers[0].param_groups[0]["lr"]
        critic_lr = self.placement_critic_optimizer.param_groups[0]["lr"]
        self.allocation_actors = actors.to(self.device)
        self.placement_critic = critic.to(self.device)
        self.allocation_critic = allocation_critic.to(self.device)
        self.allocation_actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.allocation_actors]
        self.placement_critic_optimizer = torch.optim.Adam(self.placement_critic.parameters(), lr=critic_lr)
        self.allocation_critic_optimizer = torch.optim.Adam(self.allocation_critic.parameters(), lr=critic_lr)

    def _global_state_to_tensors(self, state, *, include_candidate_features=True,
                                include_allocation_cost_features=True):
        return super()._global_state_to_tensors(
            state, include_candidate_features=include_candidate_features,
            include_allocation_cost_features=False)
