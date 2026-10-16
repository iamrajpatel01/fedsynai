"""FedSynAI: Enterprise Flower ServerApp.

Experiment dispatch:
    E2  Standard FL   -> non-iid=false, use-dp=false, use-synthetic=false
    E3  Non-IID FL    -> non-iid=true,  fedprox-mu=0.01
    E4  DP-FL         -> use-dp=true,   noise-multiplier=1.1, max-grad-norm=1.0
    E5  FedSynAI      -> non-iid=true,  use-dp=true,  use-synthetic=true

All flags are read from [tool.flwr.app.config] in pyproject.toml and can be
overridden at runtime with:
    flwr run . --run-config "use-dp=true noise-multiplier=0.8"
"""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.plugins import load_plugin   # self-registering package
from pytorchexample.task import test_tabular

# ---------------------------------------------------------------------------
# ServerApp
# ---------------------------------------------------------------------------
app = ServerApp()

# Per-run metrics accumulated by global_evaluate() for the Streamlit dashboard
_METRICS_LOG: list[dict] = []


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the FedSynAI ServerApp."""
    cfg = context.run_config

    # ---- Standard FL hyperparameters ----------------------------------------
    fraction_evaluate: float = float(cfg.get("fraction-evaluate", 1.0))
    num_rounds: int          = int(cfg.get("num-server-rounds", 5))
    lr: float                = float(cfg.get("learning-rate", 0.001))
    local_epochs: int        = int(cfg.get("local-epochs", 2))
    batch_size: int          = int(cfg.get("batch-size", 64))

    # ---- FedSynAI enterprise toggles (E2 – E5) ------------------------------
    plugin_name: str         = str(cfg.get("plugin-name", "credit_risk"))
    non_iid: bool            = bool(cfg.get("non-iid", False))
    use_dp: bool             = bool(cfg.get("use-dp", False))
    noise_multiplier: float  = float(cfg.get("noise-multiplier", 1.1))
    max_grad_norm: float     = float(cfg.get("max-grad-norm", 1.0))
    use_synthetic: bool      = bool(cfg.get("use-synthetic", False))
    fedprox_mu: float        = float(cfg.get("fedprox-mu", 0.0))
    save_model: bool         = bool(cfg.get("save-model", True))

    print(
        f"\n[FedSynAI] Experiment config:"
        f"\n  plugin      : {plugin_name}"
        f"\n  rounds      : {num_rounds}  |  lr={lr}  |  epochs={local_epochs}"
        f"\n  non-iid     : {non_iid}  |  fedprox-mu={fedprox_mu}"
        f"\n  use-dp      : {use_dp}  |  noise={noise_multiplier}  |  clip={max_grad_norm}"
        f"\n  use-synthetic: {use_synthetic}\n"
    )

    # ---- Load plugin & initialise global model --------------------------------
    plugin = load_plugin(plugin_name)
    global_model = plugin.build_model()
    arrays = ArrayRecord(global_model.state_dict())

    # ---- Pack every config flag into train_config for ClientApps -------------
    # Clients read these in client_app.py via msg.content["config"][key]
    train_config = ConfigRecord({
        "plugin-name":      plugin_name,
        "lr":               lr,
        "local-epochs":     local_epochs,
        "batch-size":       batch_size,
        "non-iid":          non_iid,
        "use-dp":           use_dp,
        "noise-multiplier": noise_multiplier,
        "max-grad-norm":    max_grad_norm,
        "use-synthetic":    use_synthetic,
        "fedprox-mu":       fedprox_mu,
    })

    # ---- Strategy ------------------------------------------------------------
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Closure captures plugin_name so global_evaluate stays stateless
    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        return _global_evaluate(server_round, arrays, plugin_name)

    # ---- Run federated learning ----------------------------------------------
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    # ---- Persist outputs for Streamlit dashboard & review reports -----------
    if save_model:
        model_path = f"final_model_{plugin_name}.pt"
        torch.save(result.arrays.to_torch_state_dict(), model_path)
        print(f"[FedSynAI] Final model saved -> {model_path}")

    # ---- Security Audit: Membership Inference Attack (MIA) ------------------
    # Measures privacy leakage of the final global model.
    # AUC ~ 0.50 -> perfect privacy  |  AUC ~ 1.00 -> full leakage
    mia_auc: float = -1.0
    try:
        from pytorchexample.security_audit import MembershipInferenceAuditor
        plugin_audit = load_plugin(plugin_name)
        audit_model = plugin_audit.build_model()
        audit_model.load_state_dict(result.arrays.to_torch_state_dict())
        device_audit = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Members   = partition 0's training split (seen during FL)
        # Non-members = centralised holdout (never seen during FL)
        member_loader, _ = plugin_audit.load_partition(
            partition_id=0,
            num_partitions=4,
            batch_size=batch_size,
            non_iid=False,
            use_synthetic=False,    # always audit on real data
        )
        non_member_loader = plugin_audit.load_centralized_dataset(batch_size=batch_size)

        auditor = MembershipInferenceAuditor(audit_model, device=device_audit)
        mia_auc = auditor.run(member_loader, non_member_loader)
        print(f"[FedSynAI] MIA AUC = {mia_auc:.4f}  (0.50=private, 1.00=leaky)")
    except Exception as exc:
        print(f"[FedSynAI] Security audit skipped: {exc}")

    # ---- Write metrics JSON -------------------------------------------------
    exp_tag = f"{plugin_name}_dp{use_dp}_synth{use_synthetic}_noniid{non_iid}"
    filename = f"metrics_{exp_tag}.json"
    summary = {
        "experiment": exp_tag,
        "config": {
            "plugin": plugin_name, "rounds": num_rounds, "lr": lr,
            "non_iid": non_iid, "use_dp": use_dp,
            "noise_multiplier": noise_multiplier, "max_grad_norm": max_grad_norm,
            "use_synthetic": use_synthetic, "fedprox_mu": fedprox_mu,
        },
        "mia_auc": mia_auc,
        "rounds": _METRICS_LOG,
    }

    target_dirs = [
        Path(__file__).resolve().parent.parent / "results",
        Path("c:/Users/iamra/Desktop/work/federatedLearning/quickstart-pytorch/results"),
    ]
    saved = False
    for r_dir in target_dirs:
        try:
            r_dir.mkdir(exist_ok=True, parents=True)
            with open(r_dir / filename, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            print(f"[FedSynAI] Round metrics saved -> {r_dir / filename}")
            saved = True
        except Exception:
            pass
    if not saved:
        Path("results").mkdir(exist_ok=True)
        with open(Path("results") / filename, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)




# ---------------------------------------------------------------------------
# Global evaluation (runs on server after each round)
# ---------------------------------------------------------------------------

def _global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    plugin_name: str,
) -> MetricRecord:
    """Evaluate the global model on the plugin's centralised holdout test set."""
    plugin = load_plugin(plugin_name)
    model = plugin.build_model()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loader = plugin.load_centralized_dataset(batch_size=128)
    metrics = test_tabular(model, test_loader, device)

    print(
        f"[Round {server_round:02d}] "
        f"loss={metrics['loss']:.4f}  "
        f"acc={metrics['accuracy']:.4f}  "
        f"f1={metrics['f1']:.4f}"
    )

    _METRICS_LOG.append({"round": server_round, **metrics})
    return MetricRecord({k: float(v) for k, v in metrics.items()})
