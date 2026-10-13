"""Run: python test_training_schedule.py (short CPU run in a temporary folder)."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from train_strict_local_mappo import (
    ALLOCATION_WARMUP_BATCHES,
    MAPS_PER_GLOBAL,
    build_argument_parser,
    build_interleaved_seed_schedule,
    training_phase_for_episode,
    validate_arguments,
    validate_seed_splits,
)


def main():
    args = build_argument_parser().parse_args([])
    validate_arguments(args)
    validate_seed_splits(args.train_seeds, args.validation_seeds,
                         args.test_seeds, args.diagnostic_seeds)
    warmup = ALLOCATION_WARMUP_BATCHES * args.placement_batch_episodes
    assert (warmup, args.placement_episodes, args.placement_only_episodes) == (10, 710, 0)
    assert args.training_seed == 2027
    assert args.train_seeds == tuple(range(3000, 4620))
    assert args.validation_seeds == tuple(range(214, 224))
    assert args.diagnostic_seeds == tuple(range(224, 274))
    assert args.test_seeds == tuple(range(434, 484))
    phases = [training_phase_for_episode(
        episode, warmup_samples=warmup,
        joint_training_end=args.placement_episodes - args.placement_only_episodes,
        total_episodes=args.placement_episodes,
    ) for episode in range(1, args.placement_episodes + 1)]
    assert phases == ["allocation_warmup"] * 10 + ["joint_placement_allocation"] * 700
    schedule = build_interleaved_seed_schedule(
        args.train_seeds, args.episodes_per_seed, args.training_seed)
    used_seeds = schedule[:args.placement_episodes * MAPS_PER_GLOBAL]
    assert len(used_seeds) == len(set(used_seeds)) == 1420
    assert schedule == build_interleaved_seed_schedule(
        args.train_seeds, args.episodes_per_seed, args.training_seed)

    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="mappo-no-freeze-") as directory:
        output = Path(directory) / "run"
        command = [
            sys.executable, "-B", str(root / "train_strict_local_mappo.py"),
            "--device", "cpu", "--torch-num-threads", "1",
            "--placement-episodes", "3", "--placement-batch-episodes", "1",
            "--placement-minibatch-size", "10",
            "--allocation-inner-updates", "1", "--allocation-inner-patience", "1",
            "--allocation-ppo-epochs", "1", "--placement-ppo-epochs", "1",
            "--embed-dim", "16", "--num-heads", "4",
            "--train-seeds", ",".join(map(str, args.train_seeds[:6])),
            "--validation-seeds", "214,215,216,217,218",
            "--diagnostic-seeds", "224,225,226,227,228",
            "--test-seeds", "434,435,436,437,438",
            "--checkpoint-every", "1", "--evaluation-every", "1",
            "--no-kmeans-baseline", "--no-greedy-baseline",
            "--output-dir", str(output),
        ]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=300)
        assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
        config = json.loads((output / "training_config.json").read_text())
        assert [config[key] for key in (
            "allocation_warmup_global_episodes", "joint_training_global_episodes",
            "placement_only_global_episodes",
        )] == [1, 2, 0]
        checkpoints = [torch.load(
            output / "checkpoints" / f"global_episode_{episode}.pt",
            map_location="cpu", weights_only=False,
        ) for episode in (1, 2, 3)]
        for scope in ("placement_actor", "placement_critic", "allocation_actor", "allocation_critic"):
            steps = [checkpoint["optimizer_step_totals"][scope] for checkpoint in checkpoints]
            assert steps[0] < steps[1] < steps[2], (scope, steps)
            assert steps[0] == 0 if scope.startswith("placement") else steps[0] > 0
        before, after = [checkpoint["model_state_dict"] for checkpoint in checkpoints[1:]]
        for prefix in ("placement_actors.", "placement_critic.", "allocation_actors.", "allocation_critic."):
            keys = [key for key in after if key.startswith(prefix)]
            assert keys and any(not torch.equal(before[key], after[key]) for key in keys), prefix
        for value in after.values():
            assert torch.isfinite(value).all()
        with (output / "training_metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 30
        assert [row["training_phase"] for row in rows] == (
            ["allocation_warmup"] * 10 + ["joint_placement_allocation"] * 20)
        assert (output / "test_summary.json").is_file()
    print("PASS: 10 warm-up + 700 joint, disjoint seeds, CPU training/checkpoints/evaluation; no freeze.")


if __name__ == "__main__":
    main()
