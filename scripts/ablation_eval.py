#!/usr/bin/env python3
"""
Unified evaluation script for all ablation experiments.

Evaluates every trained ablation checkpoint on all test datasets
and saves per-experiment metrics to results/ablation/.

Usage:
  # Evaluate all experiments:
  python scripts/ablation_eval.py --all

  # Evaluate specific experiments:
  python scripts/ablation_eval.py --experiments ablation_baseline ablation_full

  # Dry run (list what would be evaluated):
  python scripts/ablation_eval.py --all --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results" / "ablation"
METRICS_FILE = RESULTS_DIR / "metrics.json"

DINO_MODEL_PATH = os.environ.get("DINO_MODEL_PATH", "facebook/dinov3-vitb16-pretrained-lvd-142M")
GPUS = os.environ.get("GPUS", "0,1")
NPROC = int(os.environ.get("NPROC", "2"))

# Reference baseline: pre-trained checkpoint from baseline git branch.
# Not trained by ablation scripts; evaluated via --include-baseline-ref.
BASELINE_REF_CKPT = ROOT / "checkpoints" / "errnet" / "errnet_060_00463920.pt"
BASELINE_REF = {
    "baseline_ref": {"hyper": True, "fusion_mode": "none", "priority": "P0", "label": "baseline (ref)", "ckpt": str(BASELINE_REF_CKPT)},
}

# All ablation experiments with their config
ALL_EXPERIMENTS = {
    # ---- P0 ----
    "ablation_dino":       {"hyper": True, "fusion_mode": "both",   "priority": "P0", "label": "+DINO"},
    "ablation_synth":      {"hyper": True, "fusion_mode": "none",   "priority": "P0", "label": "+Synth"},
    "ablation_train":      {"hyper": True, "fusion_mode": "none",   "priority": "P0", "label": "+Train"},
    "ablation_full":       {"hyper": True, "fusion_mode": "both",   "priority": "P0", "label": "Full"},
    # ---- P1 ----
    "ablation_film_only":     {"hyper": True, "fusion_mode": "film", "priority": "P1", "label": "FiLM only"},
    "ablation_cross_only":    {"hyper": True, "fusion_mode": "cross", "priority": "P1", "label": "Cross-Attn only"},
    "ablation_dino_none":     {"hyper": True, "fusion_mode": "none", "priority": "P1", "label": "DINO none (VGG only)"},
    "ablation_both_strided":  {"hyper": True, "fusion_mode": "both", "priority": "P1", "label": "Both + strided"},
    # ---- P2 ----
    "ablation_synth_legacy":       {"hyper": True, "fusion_mode": "both", "priority": "P2", "label": "Synth: legacy"},
    "ablation_synth_reflection2":  {"hyper": True, "fusion_mode": "both", "priority": "P2", "label": "Synth: reflection2"},
    "ablation_synth_advanced":     {"hyper": True, "fusion_mode": "both", "priority": "P2", "label": "Synth: advanced"},
    "ablation_synth_mixed5050":    {"hyper": True, "fusion_mode": "both", "priority": "P2", "label": "Synth: mixed 50/50"},
    # ---- P3 ----
    "ablation_adam_manual":    {"hyper": True, "fusion_mode": "both", "priority": "P3", "label": "Adam + manual"},
    "ablation_adamw_cosine":   {"hyper": True, "fusion_mode": "both", "priority": "P3", "label": "AdamW + cosine"},
    "ablation_grad_clip":      {"hyper": True, "fusion_mode": "both", "priority": "P3", "label": "+ grad clip"},
    "ablation_ema":            {"hyper": True, "fusion_mode": "both", "priority": "P3", "label": "+ EMA"},
}

TEST_DATASETS = ["ceilnet_table2", "real20", "objects", "postcard", "wild", "sir2_withgt"]


def find_checkpoint(exp_name: str) -> Path | None:
    """Find the best checkpoint for an experiment.

    Priority: best_psnr_val.pt > errnet_latest.pt > any .pt file.
    For baseline_ref and any experiment with explicit 'ckpt' in its config,
    use that path directly.
    """
    # Baseline reference has an explicit checkpoint path
    if exp_name in BASELINE_REF:
        ckpt = Path(BASELINE_REF[exp_name]["ckpt"])
        return ckpt if ckpt.exists() else None

    ckpt_dir = CHECKPOINTS_DIR / exp_name
    if not ckpt_dir.is_dir():
        return None

    candidates = [
        ckpt_dir / "errnet_best_psnr_val.pt",
        ckpt_dir / "errnet_latest.pt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    # fallback: find any .pt file
    pt_files = sorted(ckpt_dir.glob("*.pt"))
    return pt_files[-1] if pt_files else None


def parse_metrics_from_output(stdout: str) -> dict[str, dict[str, float]]:
    """Parse test_errnet.py output for per-dataset metrics."""
    results = {}
    for line in stdout.splitlines():
        # Format: "real20: LMSE: 0.0170 | NCC: 0.8992 | PSNR: 24.3593 | SSIM: 0.8391 |"
        # or: "real20: {'LMSE': 0.0170, 'NCC': 0.8992, 'PSNR': 24.3593, 'SSIM': 0.8391}"
        line = line.strip()
        if ":" not in line:
            continue
        parts = line.split(":", 1)
        dataset = parts[0].strip()
        if dataset not in TEST_DATASETS and dataset not in [
            "testdata_table2", "testdata_real", "testdata_objects",
            "testdata_postcard", "testdata_wild", "testdata_sir2",
        ]:
            continue

        # normalize dataset names
        name_map = {
            "testdata_table2": "ceilnet_table2",
            "testdata_real": "real20",
            "testdata_objects": "objects",
            "testdata_postcard": "postcard",
            "testdata_wild": "wild",
            "testdata_sir2": "sir2_withgt",
        }
        dataset = name_map.get(dataset, dataset)

        metrics_str = parts[1].strip()
        try:
            metrics = eval(metrics_str)
            if isinstance(metrics, dict):
                results[dataset] = {k: float(v) for k, v in metrics.items()}
        except Exception:
            pass
    return results


def run_eval(exp_name: str, dry_run: bool = False) -> dict | None:
    """Run test_errnet.py for a single experiment on all datasets."""
    ckpt = find_checkpoint(exp_name)
    if ckpt is None:
        print(f"  [SKIP] {exp_name}: no checkpoint found")
        return None

    exp_cfg = {**ALL_EXPERIMENTS, **BASELINE_REF}.get(exp_name, {})
    fusion_mode = exp_cfg.get("fusion_mode", "both")
    use_hyper = exp_cfg.get("hyper", True)

    args = [
        sys.executable, str(ROOT / "test_errnet.py"),
        "--name", exp_name,
        "--dataset", "all",
        "--data_root", str(ROOT / "datasets" / "data"),
        "--result_dir", str(RESULTS_DIR / exp_name),
        "-r", "--icnn_path", str(ckpt),
        "--feature_model_path", DINO_MODEL_PATH,
    ]
    if use_hyper:
        args.append("--hyper")
    args.extend(["--fusion_mode", fusion_mode])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUS

    if dry_run:
        print(f"  [DRY RUN] {exp_name}: ckpt={ckpt}")
        return None

    print(f"\n--- Evaluating {exp_name} (ckpt: {ckpt}) ---")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--master_port", "29777",
         "--nproc_per_node", str(NPROC),
         *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=3600,
    )

    combined = {}
    if result.stdout:
        combined = parse_metrics_from_output(result.stdout)
    if result.stderr:
        print(result.stderr[-500:])

    if combined:
        print(f"  {exp_name}: {combined}")
    else:
        print(f"  [WARN] {exp_name}: no metrics parsed from output")
        if result.stdout:
            print(f"  stdout (last 500 chars): {result.stdout[-500:]}")

    return combined


def aggregate_metrics(all_results: dict) -> dict:
    """Compute mean metrics across datasets for each experiment."""
    aggregated = {}
    for exp_name, per_dataset in all_results.items():
        if not per_dataset:
            continue
        agg = {}
        for metric in ["PSNR", "SSIM", "NCC", "LMSE"]:
            values = [v[metric] for v in per_dataset.values() if metric in v]
            if values:
                agg[metric] = sum(values) / len(values)
        aggregated[exp_name] = agg
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Evaluate ablation experiments")
    parser.add_argument("--all", action="store_true", help="Evaluate all experiments")
    parser.add_argument("--include-baseline-ref", action="store_true",
                        help="Include the pre-trained baseline checkpoint as 'baseline_ref'")
    parser.add_argument("--priority", type=str, nargs="*", choices=["P0", "P1", "P2", "P3"],
                        help="Evaluate experiments of specific priority")
    parser.add_argument("--experiments", type=str, nargs="*",
                        help="Evaluate specific experiments by name")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done")
    parser.add_argument("--gpus", type=str, default=GPUS, help="GPU IDs")
    parser.add_argument("--nproc", type=int, default=NPROC, help="Number of processes")
    args = parser.parse_args()

    global GPUS, NPROC
    GPUS = args.gpus
    NPROC = args.nproc

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which experiments to run
    to_eval = set()
    if args.include_baseline_ref:
        to_eval.add("baseline_ref")
    if args.all:
        to_eval |= set(ALL_EXPERIMENTS.keys())
    elif args.priority:
        for p in args.priority:
            to_eval.update(
                name for name, cfg in ALL_EXPERIMENTS.items()
                if cfg["priority"] == p
            )
    elif args.experiments:
        to_eval = set(args.experiments)
    elif not to_eval:
        parser.error("Must specify --all, --priority, --experiments, or --include-baseline-ref")

    # Validate experiment names
    all_valid = set(ALL_EXPERIMENTS.keys()) | set(BASELINE_REF.keys())
    invalid = to_eval - all_valid
    if invalid:
        print(f"Unknown experiments: {invalid}")
        print(f"Available: {sorted(all_valid)}")
        sys.exit(1)

    # Sort by priority then name
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    sorted_exps = sorted(to_eval, key=lambda n: (priority_order.get(ALL_EXPERIMENTS[n]["priority"], 9), n))

    # Load existing results
    existing = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            existing = json.load(f)

    # Evaluate
    for exp_name in sorted_exps:
        if exp_name in existing and not args.dry_run:
            print(f"[SKIP] {exp_name}: already evaluated (delete {METRICS_FILE} to re-evaluate)")
            continue
        result = run_eval(exp_name, dry_run=args.dry_run)
        if result is not None:
            existing[exp_name] = result
            # Save incrementally
            with open(METRICS_FILE, "w") as f:
                json.dump(existing, f, indent=2)

    # Print summary
    if not args.dry_run and existing:
        print("\n========== Summary ==========")
        agg = aggregate_metrics(existing)
        for exp_name in sorted_exps:
            if exp_name in agg:
                label = ALL_EXPERIMENTS.get(exp_name, {}).get("label", exp_name)
                metrics_str = " | ".join(
                    f"{m}={agg[exp_name][m]:.4f}" for m in ["PSNR", "SSIM", "LMSE"]
                    if m in agg[exp_name]
                )
                print(f"  [{ALL_EXPERIMENTS.get(exp_name, {}).get('priority', '?')}] {label:30s} | {metrics_str}")


if __name__ == "__main__":
    main()
