"""Rebuild one continuous W&B run from a checkpoint's log_history.

A long streaming run gets interrupted (network stalls, NCCL timeouts, restarts), and
every resume starts a fresh W&B run, so the dashboard ends up as a pile of fragments.

The Trainer's own state is not fragmented: `TrainerState` is restored from the
checkpoint on resume, so `log_history` is truncated back to the checkpoint step and then
re-appended. The newest checkpoint therefore holds one clean series from step 1 to the
end -- no duplicates and no gaps -- which is a better source than stitching the W&B runs
themselves.

    python examples/train/wandb_from_checkpoint.py <checkpoint_dir> \
        --project multimodal-sae --name my-run [--csv out.csv] [--offline]
"""

import argparse
import collections
import csv as csvmod
import json
import os


def load_history(checkpoint_dir: str) -> list[dict]:
    with open(os.path.join(checkpoint_dir, "trainer_state.json")) as f:
        return json.load(f)["log_history"]


def merge_by_step(history: list[dict]) -> list[tuple[int, dict]]:
    """Collapse entries sharing a step into one record.

    compute_loss logs the per-layer FVU separately from the Trainer's own loss line, so a
    single optimizer step can appear as two entries.
    """
    merged: dict[int, dict] = collections.defaultdict(dict)
    for entry in history:
        step = entry.get("step")
        if step is None:
            continue
        for key, value in entry.items():
            if key != "step":
                merged[step][key] = value
    return sorted(merged.items())


def add_friendly_aliases(record: dict, target_module: str | None) -> dict:
    """Give the SAE metrics short names without dropping the originals."""
    out = dict(record)
    for key, value in record.items():
        if key.endswith("/dead_latent_percentage"):
            out["dead_latent_pct"] = value
        elif target_module and key == target_module:
            out["fvu"] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint_dir", help="checkpoint whose trainer_state.json holds the full history")
    ap.add_argument("--project", default="multimodal-sae")
    ap.add_argument("--name", default=None, help="W&B run name (default: checkpoint's parent dir)")
    ap.add_argument("--csv", default=None, help="also write the merged series here")
    ap.add_argument("--offline", action="store_true", help="write the run locally without uploading")
    ap.add_argument("--dry-run", action="store_true", help="summarise only; touch nothing")
    args = ap.parse_args()

    history = load_history(args.checkpoint_dir)
    records = merge_by_step(history)
    if not records:
        raise SystemExit(f"no step-tagged entries in {args.checkpoint_dir}")

    # The SAE's FVU is logged under the target module's name; find it so we can alias it.
    target_module = next(
        (k for _, r in records for k in r if "." in k and not k.endswith("/dead_latent_percentage")),
        None,
    )

    steps = [s for s, _ in records]
    gaps = [(a, b) for a, b in zip(steps, steps[1:]) if b - a > 1]
    print(f"{len(records)} steps, {steps[0]} -> {steps[-1]}, {len(gaps)} gaps")
    if gaps:
        print(f"  first gaps: {gaps[:5]}")
    print(f"  metrics: {sorted({k for _, r in records for k in r})}")

    if args.csv:
        fields = ["step"] + sorted({k for _, r in records for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for step, rec in records:
                w.writerow({"step": step, **rec})
        print(f"  wrote {args.csv}")

    if args.dry_run:
        return

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    import wandb

    name = args.name or os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint_dir)))
    run = wandb.init(project=args.project, name=name, config={"rebuilt_from": args.checkpoint_dir})
    for step, rec in records:
        run.log(add_friendly_aliases(rec, target_module), step=step)
    run.finish()
    print(f"  logged {len(records)} steps to W&B run '{name}'")


if __name__ == "__main__":
    main()
