"""Run: python -m unittest -v test_greedy_mappo (CPU; temporary output only)."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import env as physics
import train_greedy_mappo as launch
from greedy_mappo_baseline import candidate_service_costs, select_greedy_candidates, GreedyAttentionMAPPO
from learning_credit import parameter_vector
from mappo_attention_baseline import AttentionMAPPOBaseline
import strict_local_independent_mappo as original
from ver2 import gym_env


class GreedyBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_defaults_and_greedy_objective(self):
        args = launch.parse_arguments([])
        self.assertEqual((args.placement_episodes, args.placement_only_episodes, args.training_seed), (710, 0, 2027))
        self.assertEqual((args.allocation_max_inner_updates, args.allocation_inner_patience,
                          args.allocation_minibatch_size, args.allocation_ppo_epochs), (20, 10, 500, 5))
        self.assertEqual((args.power_weight, args.prb_weight, args.allocation_entropy_coef,
                          args.allocation_entropy_final_coef, args.gamma, args.gae_lambda), (.2, 1., .003, .003, 1., .95))
        self.assertEqual(args.train_seeds, tuple(range(3000, 4620)))
        schedule = launch.shared.build_interleaved_seed_schedule(args.train_seeds, args.episodes_per_seed, args.training_seed)[:1420]
        self.assertEqual(len(set(schedule)), 1420)
        self.assertFalse(set(schedule) & set(args.validation_seeds + args.diagnostic_seeds + args.test_seeds))
        # Same xyz, different angles cannot occupy two UAV slots. Once all UEs
        # are covered, a redundant position may still reduce union service cost.
        xyz = np.array([[0, 0, 250], [0, 0, 250], [0, 0, 260], [10, 0, 250]])
        costs = np.array([[.1, np.inf, np.inf], [.1, .1, np.inf],
                          [np.inf, np.inf, .2], [.05, np.inf, np.inf]])
        np.testing.assert_array_equal(select_greedy_candidates(costs, xyz, 3), [1, 2, 3])
        np.testing.assert_array_equal(select_greedy_candidates(np.full((4, 2), np.inf), xyz, 3), [0, 2, 3])
        with self.assertRaises(ValueError):
            select_greedy_candidates(costs, xyz, 4)
        with self.assertRaises(ValueError):
            select_greedy_candidates(np.full((4, 2), np.nan), xyz, 3)

    def test_costs_match_scalar_env_and_id_permutations(self):
        environment = gym_env(coordination_mode="none", power_weight=.2)
        state, _ = environment.reset(options={"ue_seed":3000})
        before = [(u.x_m, u.y_m, u.require_mbps) for u in environment.ues]
        costs = candidate_service_costs(environment)
        selected = select_greedy_candidates(costs, environment.candidate_positions[:, :3], 5)
        # All 50 UEs at selected positions plus grid endpoints: scalar oracle.
        for index in [0, environment.num_solutions-1, *selected]:
            pos = environment.solutions[index]
            for j, ue in enumerate(environment.ues):
                feasible = []
                if physics.dis_2d(pos.x_u, pos.y_u, ue.x_m, ue.y_m) <= pos.r_u:
                    for n in environment.prb_levels:
                        for power in environment.power_levels_w:
                            rate, _ = physics.data_rate_mbps(power, n, physics.gain(pos, ue))
                            if rate >= ue.require_mbps:
                                feasible.append(.2*power/2 + n/135)
                self.assertAlmostEqual(costs[index, j], min(feasible, default=np.inf), places=13)
        model = GreedyAttentionMAPPO.from_env(environment, embed_dim=8, num_heads=2)
        for shift in range(5):
            order = launch.shared.cyclic_placement_order(shift, 5)
            actions = model.act_placement(state, environment.get_actor_observations(), store_for_update=False,
                                          selection_order=order)
            self.assertEqual(actions.dtype, np.int64)
            self.assertTrue(environment.placement_action_space.contains(actions))
            np.testing.assert_array_equal(actions[list(order)], selected)
        self.assertEqual(before, [(u.x_m, u.y_m, u.require_mbps) for u in environment.ues])
        self.assertEqual(environment.phase, environment.PLACEMENT_PHASE)
        self.assertEqual(model.placement_rollout_size, 0)
        with self.assertRaises(ValueError):
            model.act_placement(state, environment.get_actor_observations(), store_for_update=True)

    def test_attention_allocation_updates_without_placement_or_counterfactuals(self):
        with patch.object(physics, "num_ues", 6):
            environment = gym_env(coordination_mode="none", power_weight=.2)
        model = GreedyAttentionMAPPO.from_env(environment, embed_dim=8, num_heads=2,
                                             allocation_ppo_epochs=1, allocation_minibatch_size=60)
        placement_before = parameter_vector(model.placement_actors).clone()
        allocation_before = parameter_vector(model.allocation_actors).clone()
        critic_before = parameter_vector(model.allocation_critic).clone()
        self.assertFalse(any(isinstance(m, original.PopArtLinear) for m in model.modules()))
        self.assertEqual(model.allocation_critic.allocation_cost_dim, 0)
        with patch.object(physics, "num_ues", 6), \
             patch.object(environment, "preview_allocation_step", side_effect=AssertionError("counterfactual")), \
             patch.object(model, "_counterfactual_values", side_effect=AssertionError("Q_cf")), \
             patch.object(original, "expected_clipped_loss", side_effect=AssertionError("all-action PPO")), \
             patch.object(model, "restore_allocation_snapshot", side_effect=AssertionError("rollback")), \
             patch.object(model, "update_placement_rollout", side_effect=AssertionError("placement update")):
            result = original.train_frozen_placement_batch(
                environment, model, ue_seeds=[3000]*5+[3001]*5,
                placement_orders=[launch.shared.cyclic_placement_order(i,5) for _ in range(2) for i in range(5)],
                max_inner_updates=1, inner_patience=1, update_placement=False, update_allocation=True)
            state, _ = environment.reset(options={"ue_seed":3000})
            placement = model.act_placement(state, environment.get_actor_observations(), store_for_update=False)
            state, *_ = environment.step(placement)
        self.assertEqual(result["allocation_inner_episodes"], 10)
        self.assertEqual(result["allocation_updates"][0]["allocation_rollout_size"], 60)
        self.assertEqual(model.optimizer_step_counts()["placement_actor"], 0)
        self.assertEqual(model.optimizer_step_counts()["placement_critic"], 0)
        torch.testing.assert_close(placement_before, parameter_vector(model.placement_actors), rtol=0, atol=0)
        self.assertFalse(torch.equal(allocation_before, parameter_vector(model.allocation_actors)))
        self.assertFalse(torch.equal(critic_before, parameter_vector(model.allocation_critic)))
        # Exactly the same allocation forward pass as the existing attention baseline.
        reference = AttentionMAPPOBaseline.from_env(environment, embed_dim=8, num_heads=2)
        reference.load_state_dict(model.state_dict())
        observations = environment.get_actor_observations()
        a, _ = model.act_allocation(state, observations, deterministic=True, store_for_update=False)
        b, _ = reference.act_allocation(state, observations, deterministic=True, store_for_update=False)
        np.testing.assert_array_equal(a, b)
        for raw in observations:
            if raw["current_link_features"][1] >= .5:
                self.assertTrue(raw["allocation_pair_mask"].all())
        tensors = model._global_state_to_tensors(state, include_candidate_features=False)
        self.assertNotIn("allocation_cost_features", tensors)
        poisoned = {**state, "allocation_cost_features": np.full_like(state["allocation_cost_features"], np.nan)}
        bad_tensors = model._global_state_to_tensors(poisoned, include_candidate_features=False)
        torch.testing.assert_close(model.allocation_critic(tensors), model.allocation_critic(bad_tensors), rtol=0, atol=0)
        for actor, raw in zip(model.allocation_actors, observations):
            clean = model._allocation_observation_to_tensors(raw)
            changed = {key: value.clone() for key,value in clean.items()}
            changed["own_uav_features"][:5] = float("nan")
            changed["current_ue_features"][:2] = float("nan")
            changed["local_ue_features"][~changed["local_ue_mask"].bool()] = float("nan")
            torch.testing.assert_close(actor.encode(clean), actor.encode(changed), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "greedy.pt"
            model.save_checkpoint(path, 1)
            with self.assertRaises(ValueError):
                reference.load_checkpoint(path)
            for count in (20, 50, 70):
                with patch.object(physics, "num_ues", count):
                    other_env = gym_env(coordination_mode="none", power_weight=.2)
                    other = GreedyAttentionMAPPO.from_env(other_env, embed_dim=8, num_heads=2,
                                                          reward_scale=model.reward_scale)
                    other.load_checkpoint(path)
                    state, _ = other_env.reset(options={"ue_seed":434})
                    placement = other.act_placement(state, other_env.get_actor_observations(), store_for_update=False)
                    state, *_ = other_env.step(placement)
                    actions, _ = other.act_allocation(state, other_env.get_actor_observations(), deterministic=True, store_for_update=False)
                    self.assertEqual(actions.shape, (5,3))
                    tensors = other._global_state_to_tensors(state, include_candidate_features=False)
                    self.assertTrue(torch.isfinite(other.allocation_critic(tensors)).all())

    def test_patience_counts_ties_and_keeps_last_inner_model(self):
        with patch.object(physics, "num_ues", 6):
            environment = gym_env(coordination_mode="none", power_weight=.2)
            snapshots = []
            model = GreedyAttentionMAPPO.from_env(
                environment, embed_dim=8, num_heads=2, allocation_ppo_epochs=1,
                diagnostics_callback=lambda phase, metrics: snapshots.append(parameter_vector(model.allocation_critic)))
            calls = 0
            def controlled_evaluation(*args, **kwargs):
                nonlocal calls
                # Ten problems per evaluation pass: improve, tie, worsen.
                score = [2., 3., 3., 2., 4., 5.][calls // 10]
                calls += 1
                return {"placement_return": score, "served_count": 4,
                        "final_info": {"total_used_power_w": .4, "total_used_prbs": 12}}
            with patch.object(original, "evaluate_fixed_placement_episode", side_effect=controlled_evaluation), \
                 patch.object(model, "restore_allocation_snapshot", side_effect=AssertionError("rollback")):
                result = original.train_frozen_placement_batch(
                    environment, model, ue_seeds=[3000]*5+[3001]*5,
                    placement_orders=[launch.shared.cyclic_placement_order(i,5) for _ in range(2) for i in range(5)],
                    max_inner_updates=5, inner_patience=2, update_placement=False, update_allocation=True)
            self.assertEqual(result["allocation_inner_updates"], 3)
            self.assertEqual(result["allocation_stop_reason"], "no_improvement")
            self.assertEqual([r["checks_without_improvement"] for r in result["allocation_evaluation_history"]], [0,0,1,2])
            self.assertEqual(result["final_evaluations"][0]["placement_return"], 2.)
            self.assertEqual(result["best_evaluations"][0]["placement_return"], 3.)
            torch.testing.assert_close(parameter_vector(model.allocation_critic), snapshots[-1], rtol=0, atol=0)
            self.assertFalse(torch.equal(snapshots[-1], snapshots[-2]))

    def test_portable_launcher_and_exact_resume(self):
        common = ["--device", "cpu", "--torch-num-threads", "1", "--placement-episodes", "3",
                  "--placement-batch-episodes", "1", "--num-ues", "6", "--embed-dim", "8", "--num-heads", "2",
                  "--allocation-inner-updates", "1", "--allocation-inner-patience", "1",
                  "--allocation-ppo-epochs", "1", "--evaluation-every", "2", "--checkpoint-every", "1",
                  "--train-seeds", "3000,3001,3002,3003,3004,3005", "--validation-seeds", "214,215,216,217,218",
                  "--diagnostic-seeds", "224,225,226,227,228", "--test-seeds", "434,435,436,437,438"]
        command = [sys.executable, str(launch.ROOT / "train_greedy_mappo.py")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, resumed = root / "full", root / "resumed"
            def run(options, expected=0):
                result = subprocess.run(command + options, cwd=root, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, expected, result.stdout[-3000:]+result.stderr[-3000:])
            run(common + ["--output-dir", str(full)])
            run(common + ["--output-dir", str(resumed), "--stop-after-global", "1"])
            checkpoint = resumed / "checkpoints/latest_resume.pt"
            checkpoint_before = checkpoint.read_bytes()
            log = resumed / "allocation_metrics.csv"
            original_log = log.read_bytes()
            lines = original_log.splitlines(keepends=True)
            lines[1] = b"9" + lines[1][1:]  # same byte length, wrong episode number
            log.write_bytes(b"".join(lines))
            run(["--resume", str(checkpoint)], expected=1)
            self.assertEqual(checkpoint.read_bytes(), checkpoint_before)
            self.assertEqual(log.read_bytes(), b"".join(lines))
            log.write_bytes(original_log)
            full_log = (full / "allocation_metrics.csv").read_bytes()
            run(["--resume", str(checkpoint), "--output-dir", str(full)], expected=2)
            self.assertEqual((full / "allocation_metrics.csv").read_bytes(), full_log)
            # Simulate buffered log writes beyond a completed checkpoint.
            with (resumed / "allocation_metrics.csv").open("a") as handle:
                handle.write("stale trailing data\n")
            run(["--resume", str(checkpoint)])
            left = torch.load(full / "checkpoints/latest_resume.pt", weights_only=False)
            right = torch.load(checkpoint, weights_only=False)
            for key in ("model_state_dict", "allocation_actor_optimizers", "allocation_critic_optimizer", "optimizer_step_totals", "rng_state"):
                self.assertTrue(original._nested_state_equal(left[key], right[key]), key)
            for name in (*launch.LOGS, "test_metrics.csv", "diagnostic_metrics.csv"):
                self.assertEqual((full/name).read_bytes(), (resumed/name).read_bytes(), name)
            initial = torch.load(full / "checkpoints/resume_global_episode_1.pt", weights_only=False)
            final = left["model_state_dict"]
            self.assertTrue(any(not torch.equal(v, final[k]) for k,v in initial["model_state_dict"].items() if k.startswith("allocation_actors.")))
            self.assertTrue(all(torch.equal(v, final[k]) for k,v in initial["model_state_dict"].items() if k.startswith("placement_")))
            budget = json.loads((full/"training_budget_summary.json").read_text())
            self.assertEqual((budget["allocation_training_episodes"], budget["real_allocation_training_transitions"], budget["allocation_freeze_globals"]), (30,180,0))
            self.assertEqual(budget["counterfactual_preview_transitions"], 0)
            run(["--resume", str(checkpoint), "--power-weight", "0.3"], expected=1)
            self.assertEqual((full/"allocation_metrics.csv").read_bytes(), (resumed/"allocation_metrics.csv").read_bytes())
            # Rewind, then stop immediately: the advertised latest checkpoint
            # must match the truncated logs even before another PPO update.
            run(["--resume", str(full / "checkpoints/resume_global_episode_1.pt"),
                 "--stop-after-global", "1"])
            rewound = torch.load(full / "checkpoints/latest_resume.pt", weights_only=False)
            self.assertEqual(rewound["episode"], 1)
            run(["--resume", str(full / "checkpoints/latest_resume.pt")])
            rerun = torch.load(full / "checkpoints/latest_resume.pt", weights_only=False)
            self.assertTrue(original._nested_state_equal(left["model_state_dict"], rerun["model_state_dict"]))


if __name__ == "__main__":
    unittest.main()
