"""Train grid-greedy placement + attention-only MAPPO allocation, without freeze.

Reuse the server seed schedule, env, complete-episode PPO/GAE collector and
reward-patience rule. No placement learning and no inner best-model rollback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import env as physics
import train_strict_local_mappo as shared
from greedy_mappo_baseline import GreedyAttentionMAPPO, PLANNER_VERSION
from strict_local_independent_mappo import train_frozen_placement_batch
from ver2 import gym_env

ROOT = Path(__file__).resolve().parent
ALGORITHM = "greedy_placement_mappo_attention_only_v1"
SOURCES = ("env.py", "ver2.py", "learning_credit.py", "kmeans_baseline.py",
           "strict_local_independent_mappo.py", "local_mappo_baseline.py",
           "mappo_attention_baseline.py", "train_strict_local_mappo.py",
           "greedy_mappo_baseline.py", "train_greedy_mappo.py")
LOGS = ("training_metrics.csv", "allocation_metrics.csv", "validation_metrics.csv",
        "learning_diagnostics.jsonl", "inner_evaluations.jsonl")


def parse_arguments(argv=None):
    parser = shared.build_argument_parser()
    parser.description = __doc__
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--num-ues", type=int, default=50)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--stop-after-global", type=int, help="Stop cleanly at a global boundary; resume keeps the original total budget")
    # Hyperparameters from the requested 10/400/200 run; seed/splits/schedule
    # from MAPPO-sever. Placement settings are retained only as CLI compatibility.
    parser.set_defaults(placement_episodes=710, placement_only_episodes=0,
                        placement_ppo_epochs=5, allocation_ppo_epochs=5,
                        allocation_max_inner_updates=20, allocation_inner_patience=10,
                        allocation_entropy_coef=.003, allocation_entropy_final_coef=.003,
                        power_weight=.2, prb_weight=1., served_reward=1.,
                        no_greedy_baseline=True, no_kmeans_baseline=True,
                        output_dir=str(ROOT / "training_outputs_greedy_mappo_attention_700_seed2027"))
    args = parser.parse_args(argv)
    if args.resume:
        run = args.resume.expanduser().resolve().parent.parent
        saved = json.loads((run / "training_config.json").read_text())
        if saved.get("algorithm") != ALGORITHM:
            parser.error("Resume requires this greedy MAPPO baseline's checkpoint")
        excluded = {"help", "resume", "output_dir", "print_config", "stop_after_global"}
        parser.set_defaults(**{a.dest: saved[a.dest] for a in parser._actions
                              if a.dest in saved and a.dest not in excluded}, output_dir=str(run))
        args = parser.parse_args(argv)
        if Path(args.output_dir).expanduser().resolve() != run:
            parser.error("Resume output-dir must be the run containing the checkpoint; move/copy the whole run together")
    shared.validate_arguments(args)
    shared.validate_seed_splits(args.train_seeds, args.validation_seeds, args.test_seeds, args.diagnostic_seeds)
    if args.coordination_mode != "none" or args.placement_only_episodes != 0:
        parser.error("This baseline requires coordination-mode none and placement-only-episodes 0")
    if not math.isfinite(args.gae_lambda) or not 0 <= args.gae_lambda <= 1 or args.num_ues <= 0:
        parser.error("gae-lambda must be in [0,1] and num-ues must be positive")
    if args.stop_after_global is not None and not 1 <= args.stop_after_global <= args.placement_episodes:
        parser.error("stop-after-global must be within the configured global budget")
    if args.kmeans_baseline_only or args.benchmark_inference_episodes:
        parser.error("K-means-only and the inherited inference benchmark are not used by this launcher")
    return args


def configuration(args):
    options = vars(args).copy()
    for name in ("print_config", "stop_after_global"):
        options.pop(name)
    fingerprint = shared.resume_configuration_fingerprint(argparse.Namespace(**options))
    return {**options, "algorithm": ALGORITHM, "planner_version": PLANNER_VERSION,
            "configuration_fingerprint": fingerprint,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
            "total_global_episodes": args.placement_episodes,
            "training_map_episodes": 2 * args.placement_episodes,
            "maps_per_global": 2, "placements_per_map": 5,
            "reference_warmup_globals": args.placement_batch_episodes,
            "allocation_active_globals": args.placement_episodes,
            "placement_trainable": False, "allocation_freeze": False,
            "seed_schedule": "server shuffled maps, two maps/global, five cyclic UAV-ID assignments/map",
            "placement_geometry": "one deterministic greedy geometry per map; five ID permutations, not five distinct geometries",
            "placement_score": "maximum new individually QoS-feasible UEs; ties minimum union service cost, then candidate ID",
            "placement_unused_options": ["placement_minibatch_size", "placement_ppo_epochs", "placement_entropy_coef", "placement_popart_beta"],
            "network_architecture": "attention allocation actors and global V critic; no COMA, local bypass, PopArt or critic service-cost features",
            "allocation_critic_feature_version": "global_attention_no_cost_v1",
            "allocation_actor_feature_version": "compact_qos_link_mixed_no_bypass_v1",
            "allocation_credit": "sampled PPO + team GAE; no simulator counterfactuals",
            "allocation_mask_scope": "coverage only, reject + 100 pairs when covered",
            "reward_scale": args.num_ues,
            "reward": "served_reward - power_weight*winner_power/2 - prb_weight*winner_PRBs/135",
            "inner_stop_rule": "consecutive deterministic no-improvement patience or max updates; no rollback",
            "allocation_kl_stop_rule": "per actor, within PPO update only",
            "checkpoint_resume_scope": "completed global boundaries; latest_resume.pt restores model, optimizers and RNG"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def append_csv(path, rows):
    if not rows:
        return
    populated = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not populated:
            writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def extend_log_digest(path, digest, start, end):
    """Hash only newly appended bytes; resume checks the saved prefix once."""
    if not 0 <= start <= end:
        raise ValueError(f"Invalid log byte range: {path.name}")
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start
        while remaining:
            block = handle.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError(f"Log is shorter than its checkpoint: {path.name}")
            digest.update(block)
            remaining -= len(block)


def make_model(args, environment, device, callback=None):
    return GreedyAttentionMAPPO.from_env(
        environment, embed_dim=args.embed_dim, num_heads=args.num_heads,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=args.gamma,
        gae_lambda=args.gae_lambda, placement_batch_episodes=10,
        placement_minibatch_size=10, placement_ppo_epochs=args.placement_ppo_epochs,
        allocation_ppo_epochs=args.allocation_ppo_epochs,
        allocation_minibatch_size=args.allocation_minibatch_size,
        allocation_entropy_coef=args.allocation_entropy_coef,
        target_kl=args.target_kl, diagnostics_callback=callback, device=device)


def plot_results(output):
    with (output / "allocation_metrics.csv").open() as handle:
        rewards = np.array([float(r["episode_return"]) for r in csv.DictReader(handle)])
    prefix = np.r_[0., np.cumsum(rewards)]
    end = np.arange(1, len(rewards) + 1)
    start = np.maximum(0, end - 1000)
    fig, ax = shared.plt.subplots(figsize=(12, 4))
    ax.plot(end, rewards, alpha=.25, linewidth=.5, label="All allocation episodes")
    ax.plot(end, (prefix[end] - prefix[start]) / (end - start), label="Moving mean: 1,000 episodes")
    ax.set(xlabel="Training episode", ylabel="Episode return", title="Greedy + MAPPO allocation training")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "allocation_training.png", dpi=180)
    shared.plt.close(fig)
    with (output / "validation_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    episodes = sorted({int(r["global_episode"]) for r in rows})
    means = [np.mean([float(r["sum_reward"]) for r in rows if int(r["global_episode"]) == g]) for g in episodes]
    fig, ax = shared.plt.subplots(figsize=(12, 4))
    ax.scatter([int(r["global_episode"]) for r in rows], [float(r["sum_reward"]) for r in rows], alpha=.25, s=8, label="All validation maps")
    ax.plot(episodes, means, label="Mean over validation maps")
    ax.set(xlabel="Global episode", ylabel="Episode return", title="Validation")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "validation.png", dpi=180)
    shared.plt.close(fig)


def main(argv=None):
    args = parse_arguments(argv)
    config = configuration(args)
    if args.print_config:
        print(json.dumps(config, indent=2, default=str))
        return
    output = Path(args.output_dir).expanduser().resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Fresh training requires an empty output directory: {output}")
    shared.configure_deterministic_training()
    torch.set_num_threads(args.torch_num_threads)
    shared.seed_training_randomness(args.training_seed)
    device = shared.choose_device(args.device)
    physics.num_ues = args.num_ues
    environment = gym_env(coordination_mode="none", seed=args.train_seeds,
                          shuffle_ue_order=not args.no_shuffle_ue_order,
                          served_reward=args.served_reward, power_weight=args.power_weight, prb_weight=args.prb_weight)
    global_episode = allocation_episodes = updates = 0
    active_seconds = 0.
    best = None
    log_offsets = dict.fromkeys(LOGS, 0)
    log_digests = {name: hashlib.sha256() for name in LOGS}

    def diagnostics(phase, metrics):
        if phase != "allocation":
            raise RuntimeError("Greedy baseline unexpectedly updated placement")
        nonlocal updates
        updates += 1
        append_jsonl(output / "learning_diagnostics.jsonl", {"global_episode": global_episode, "update": updates, **metrics})
        print(f"[global {global_episode:03d} allocation update {updates}] "
              f"mean_return={metrics['allocation_rollout_return']/10:.4f} "
              f"KL={metrics['allocation_policy_kl']:.6f}", flush=True)

    learner = make_model(args, environment, device, diagnostics)
    if args.resume:
        saved = json.loads((output / "training_config.json").read_text())
        for key in ("configuration_fingerprint", "source_sha256"):
            if saved.get(key) != config[key]:
                raise ValueError(f"Exact resume rejected: {key} changed")
        global_episode = learner.load_checkpoint(args.resume, restore_training_state=True)
        progress = learner.loaded_trainer_state
        if progress.get("configuration_fingerprint") != config["configuration_fingerprint"] or progress.get("source_sha256") != config["source_sha256"]:
            raise ValueError("Resume checkpoint does not belong to this configuration/source")
        if args.stop_after_global is not None and args.stop_after_global < global_episode:
            raise ValueError("stop-after-global precedes the loaded checkpoint")
        if progress.get("global_episode") != global_episode:
            raise ValueError("Checkpoint model and trainer global counters disagree")
        allocation_episodes, updates = progress["allocation_episodes"], progress["allocation_updates"]
        best, active_seconds = progress["best_validation"], progress["active_wall_time_seconds"]
        if best:
            checkpoint = output / "checkpoints" / best["checkpoint"]
            if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != best["checkpoint_sha256"]:
                raise ValueError("Missing or modified best-validation checkpoint")
        # Validate ALL prefixes before mutating any artifact. Length alone can
        # silently accept edited history or logs copied from a different run.
        if set(progress["log_offsets"]) != set(LOGS) or set(progress.get("log_sha256", {})) != set(LOGS):
            raise ValueError("Resume checkpoint has incomplete log integrity metadata")
        log_offsets = progress["log_offsets"].copy()
        for name, offset in log_offsets.items():
            path = output / name
            if type(offset) is not int or offset < 0:
                raise ValueError(f"Invalid resume log offset: {name}")
            if not path.is_file() or path.stat().st_size < offset:
                raise ValueError(f"Resume log missing or shorter than its checkpoint: {name}")
            extend_log_digest(path, log_digests[name], 0, offset)
            if log_digests[name].hexdigest() != progress["log_sha256"][name]:
                raise ValueError(f"Resume log history differs from its checkpoint: {name}")
        # Publish the restored boundary BEFORE truncation. A rewind that stops
        # here (or is interrupted) must never advertise a future latest model.
        learner.save_checkpoint(output / "checkpoints" / "latest_resume.pt", global_episode, trainer_state=progress)
        for name, offset in log_offsets.items():
            with (output / name).open("r+b") as handle:
                handle.truncate(offset)
        if best:
            write_json(output / "best_validation_summary.json", best)
        else:
            (output / "best_validation_summary.json").unlink(missing_ok=True)
        if global_episode < args.placement_episodes:
            # Completed-run results are stale when rewinding to an earlier boundary.
            stale = [output / n for n in ("test_metrics.csv", "test_summary.json", "diagnostic_metrics.csv", "diagnostic_summary.json", "training_budget_summary.json", "allocation_training.png", "validation.png")]
            stale.append(output / "checkpoints" / "final_model.pt")
            for pattern in ("resume_global_episode_*.pt", "best_validation_global_episode_*.pt"):
                stale.extend(p for p in (output / "checkpoints").glob(pattern)
                             if int(p.stem.rsplit("_", 1)[1]) > global_episode)
            existing = [p for p in stale if p.exists()]
            if existing:
                archive = output / f"archived_before_resume_{time.time_ns()}"
                archive.mkdir()
                for path in existing:
                    path.replace(archive / path.name)
    else:
        (output / "checkpoints").mkdir(parents=True, exist_ok=True)
        for name in LOGS:
            (output / name).touch()
        config["runtime"] = shared.runtime_provenance(device)
        config["active_allocation_parameters"] = sum(p.numel() for p in learner.parameters() if p.requires_grad)
        write_json(output / "training_config.json", config)
    schedule = shared.build_interleaved_seed_schedule(args.train_seeds, args.episodes_per_seed, args.training_seed)
    end = args.stop_after_global or args.placement_episodes
    started = time.monotonic()
    print(f"Greedy + MAPPO attention-only | {args.placement_episodes} globals, "
          f"{args.placement_episodes*2} map visits | allocation active throughout | {device}", flush=True)
    for global_episode in range(global_episode + 1, end + 1):
        maps = schedule[2*(global_episode-1):2*global_episode]
        seeds = [seed for seed in maps for _ in range(5)]
        orders = [shared.cyclic_placement_order(i, 5) for _ in maps for i in range(5)]
        base_episodes = allocation_episodes
        result = train_frozen_placement_batch(
            environment, learner, ue_seeds=seeds, placement_orders=orders,
            max_inner_updates=args.allocation_max_inner_updates, inner_patience=args.allocation_inner_patience,
            update_placement=False, update_allocation=True,
            allocation_entropy_schedule=lambda n: shared.allocation_entropy_coefficient(
                base_episodes + n, initial=args.allocation_entropy_coef, final=args.allocation_entropy_final_coef,
                anneal_start_episode=args.allocation_entropy_anneal_start_episode,
                anneal_end_episode=args.allocation_entropy_anneal_end_episode))
        rows = []
        for inner in range(result["allocation_inner_updates"]):
            for problem, seed in enumerate(seeds):
                episode = result["allocation_episode_results_by_problem"][problem][inner]
                info = episode["final_info"]
                allocation_episodes += 1
                rows.append({"allocation_episode": allocation_episodes, "global_episode": global_episode,
                             "inner_update": inner + 1, "ue_seed": seed,
                             "placement_order": shared.format_placement_order(orders[problem]),
                             "episode_return": episode["allocation_return"], "served_count": episode["served_count"],
                             "total_used_prbs": info["total_used_prbs"], "total_used_power_w": info["total_used_power_w"]})
        append_csv(output / "allocation_metrics.csv", rows)
        append_jsonl(output / "inner_evaluations.jsonl", {"global_episode": global_episode, "history": result["allocation_evaluation_history"]})
        append_csv(output / "training_metrics.csv", [{
            "global_episode": global_episode, "map_seeds": ",".join(map(str, maps)),
            "allocation_training_active": 1, "placement_updates": 0,
            "allocation_inner_updates": result["allocation_inner_updates"],
            "allocation_stop_reason": result["allocation_stop_reason"],
            "initial_mean_return": np.mean([r["placement_return"] for r in result["initial_evaluations"]]),
            "final_mean_return": np.mean([r["placement_return"] for r in result["final_evaluations"]]),
            "allocation_training_episodes": allocation_episodes,
            "real_allocation_training_transitions": allocation_episodes * args.num_ues,
            "placements": json.dumps([p.tolist() for p in result["placement_actions"]])}])
        if global_episode % args.evaluation_every == 0 or global_episode == args.placement_episodes:
            validation = shared.evaluate_seed_set(environment, learner, args.validation_seeds,
                                                  global_episode=global_episode, evaluation_kind="validation")
            append_csv(output / "validation_metrics.csv", validation)
            summary = shared.summarize_evaluation(validation, num_ues=args.num_ues)
            print(f"[validation {global_episode}] reward={summary['mean_sum_reward']:.6f} "
                  f"served={summary['mean_served_count']:.3f}", flush=True)
            if best is None or shared.validation_selection_score(summary) > shared.validation_selection_score(best):
                path = output / "checkpoints" / f"best_validation_global_episode_{global_episode}.pt"
                learner.save_checkpoint(path, global_episode)
                best = {**summary, "global_episode": global_episode, "checkpoint": path.name,
                        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                write_json(output / "best_validation_summary.json", best)
        for name in LOGS:
            end_offset = (output / name).stat().st_size
            extend_log_digest(output / name, log_digests[name], log_offsets[name], end_offset)
            log_offsets[name] = end_offset
        progress = {"resume_boundary": "outer_frozen_batch", "global_episode": global_episode,
                    "configuration_fingerprint": config["configuration_fingerprint"], "source_sha256": config["source_sha256"],
                    "allocation_episodes": allocation_episodes, "allocation_updates": updates, "best_validation": best,
                    "active_wall_time_seconds": active_seconds + time.monotonic() - started,
                    "log_offsets": log_offsets.copy(),
                    "log_sha256": {name: digest.hexdigest() for name, digest in log_digests.items()}}
        learner.save_checkpoint(output / "checkpoints" / "latest_resume.pt", global_episode, trainer_state=progress)
        if global_episode % args.checkpoint_every == 0:
            learner.save_checkpoint(output / "checkpoints" / f"resume_global_episode_{global_episode}.pt", global_episode, trainer_state=progress)
        print(f"global={global_episode:03d}/{args.placement_episodes} maps={maps} "
              f"inner={result['allocation_inner_updates']} stop={result['allocation_stop_reason']} "
              f"allocation_episodes={allocation_episodes}", flush=True)
    if global_episode < args.placement_episodes:
        print(f"Stopped at global {global_episode}; resume with {output / 'checkpoints/latest_resume.pt'}", flush=True)
        return
    learner.save_checkpoint(output / "checkpoints" / "final_model.pt", global_episode)
    write_json(output / "training_budget_summary.json", {
        "global_episodes_completed": global_episode, "training_map_episodes": 2 * global_episode,
        "placement_samples_collected": 10 * global_episode, "placement_ppo_update_batches": 0,
        "allocation_training_episodes": allocation_episodes, "allocation_updates": updates,
        "real_allocation_training_transitions": allocation_episodes * args.num_ues,
        "counterfactual_preview_transitions": 0, "optimizer_step_counts": learner.optimizer_step_counts(),
        "active_wall_time_seconds": active_seconds + time.monotonic() - started,
        "allocation_active_globals": global_episode, "allocation_freeze_globals": 0})
    learner.load_checkpoint(output / "checkpoints" / best["checkpoint"])
    for kind, seeds in (("diagnostic", args.diagnostic_seeds), ("test", args.test_seeds)):
        if seeds:
            rows = shared.evaluate_seed_set(environment, learner, seeds, global_episode=best["global_episode"], evaluation_kind=kind)
            path = output / f"{kind}_metrics.csv"
            # Regenerate post-training evaluations, never append duplicates on resume.
            path.write_text("")
            append_csv(path, rows)
            write_json(output / f"{kind}_summary.json", {**shared.summarize_evaluation(rows, num_ues=args.num_ues),
                                                       "selected_checkpoint": best["checkpoint"]})
    plot_results(output)
    print(f"Completed. Best validation: {output / 'checkpoints' / best['checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
