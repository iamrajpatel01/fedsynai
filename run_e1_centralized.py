"""run_e1_centralized.py — Experiment 1: Centralised Baseline (no Flower).

Trains the credit_risk plugin model on the FULL combined dataset (no federation)
and logs loss, accuracy, F1 and MIA AUC to results/metrics_e1_centralized.json.

This establishes the upper-bound performance ceiling that all FL experiments
(E2–E5) are compared against in the Review 0 / Review 1 results table.

Usage (from the quickstart-pytorch directory):
    python run_e1_centralized.py
    python run_e1_centralized.py --epochs 10 --lr 0.001 --batch-size 64
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from pytorchexample.plugins import load_plugin
from pytorchexample.security_audit import MembershipInferenceAuditor
from pytorchexample.task import test_tabular, train_tabular


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedSynAI E1 – Centralised Baseline")
    p.add_argument("--plugin",      default="credit_risk", help="Plugin name")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--num-clients", type=int,   default=10,
                   help="How many FL partitions to pool into the central dataset")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  FedSynAI — E1: Centralised Baseline")
    print(f"  plugin={args.plugin}  epochs={args.epochs}  lr={args.lr}  device={device}")
    print(f"{'='*60}\n")

    # ---- Load plugin -----------------------------------------------------------
    plugin = load_plugin(args.plugin)
    model = plugin.build_model()
    model.to(device)

    # ---- Pool ALL client partitions into one big training DataLoader ----------
    # This simulates a single entity having access to all data (no privacy).
    all_train_datasets = []
    for pid in range(args.num_clients):
        train_loader, _ = plugin.load_partition(
            partition_id=pid,
            num_partitions=args.num_clients,
            batch_size=args.batch_size,
            non_iid=False,
            use_synthetic=False,
        )
        all_train_datasets.append(train_loader.dataset)

    combined_train = ConcatDataset(all_train_datasets)
    full_trainloader = DataLoader(
        combined_train, batch_size=args.batch_size, shuffle=True
    )
    print(f"[E1] Pooled training set: {len(combined_train):,} samples from {args.num_clients} partitions")

    # ---- Load centralised test set -------------------------------------------
    testloader = plugin.load_centralized_dataset(batch_size=args.batch_size)
    print(f"[E1] Test set: {len(testloader.dataset):,} samples\n")

    # ---- Train ----------------------------------------------------------------
    start = time.time()
    train_loss, _ = train_tabular(
        net=model,
        trainloader=full_trainloader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_dp=False,
        fedprox_mu=0.0,
    )
    elapsed = time.time() - start
    print(f"\n[E1] Training complete in {elapsed:.1f}s  |  avg_train_loss={train_loss:.4f}")

    # ---- Evaluate -------------------------------------------------------------
    metrics = test_tabular(model, testloader, device)
    print(
        f"[E1] Test  ->  loss={metrics['loss']:.4f}  "
        f"acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}"
    )

    # ---- Security Audit (MIA) ------------------------------------------------
    # Members = first partition's training split; non-members = centralised test
    member_loader, _ = plugin.load_partition(
        partition_id=0,
        num_partitions=args.num_clients,
        batch_size=args.batch_size,
        non_iid=False,
        use_synthetic=False,
    )
    auditor = MembershipInferenceAuditor(model, device=device)
    mia_auc = auditor.run(member_loader, testloader)

    # ---- Save results --------------------------------------------------------
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output = {
        "experiment": "e1_centralized",
        "config": {
            "plugin": args.plugin,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_clients_pooled": args.num_clients,
        },
        "metrics": {
            "train_loss":   round(train_loss, 6),
            "test_loss":    round(metrics["loss"], 6),
            "accuracy":     round(metrics["accuracy"], 6),
            "f1":           round(metrics["f1"], 6),
        },
        "mia_auc":          round(mia_auc, 6),
        "training_time_s":  round(elapsed, 2),
        # Mimic the per-round format so the dashboard can treat E1 as round 0
        "rounds": [{"round": 0, **metrics, "mia_auc": mia_auc}],
    }

    out_path = results_dir / "metrics_e1_centralized.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"\n[E1] Results saved -> {out_path}")
    print(f"     accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  MIA AUC={mia_auc:.4f}")
    print(f"\nThis is your Review 0 centralised baseline - compare E2-E5 against these numbers.\n")

    # Save model
    torch.save(model.state_dict(), f"final_model_e1_{args.plugin}.pt")


if __name__ == "__main__":
    main()
