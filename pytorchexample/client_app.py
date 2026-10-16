"""FedSynAI: Enterprise Flower ClientApp.

Experiment dispatch (flags come from server via ConfigRecord):
    E2  Standard FL   -> non-iid=false, use-dp=false, use-synthetic=false
    E3  Non-IID FL    -> non-iid=true,  fedprox-mu=0.01
    E4  DP-FL         -> use-dp=true,   noise-multiplier=1.1, max-grad-norm=1.0
    E5  FedSynAI      -> non-iid=true,  use-dp=true,  use-synthetic=true
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.plugins import load_plugin
from pytorchexample.task import test_tabular, train_tabular

# Flower ClientApp
app = ClientApp()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_cfg(msg: Message, context: Context) -> dict:
    """Merge train_config sent by server with local run_config fallbacks.

    Server values take precedence; run_config provides defaults so the
    client still works when running standalone (e.g. flwr run . --local).
    """
    server_cfg = msg.content.get("config", {})
    rc = context.run_config

    def _get(key, default):
        return server_cfg.get(key, rc.get(key, default))

    return {
        "plugin_name":      str(_get("plugin-name",       "credit_risk")),
        "lr":               float(_get("lr",               rc.get("learning-rate", 0.001))),
        "epochs":           int(_get("local-epochs",       2)),
        "batch_size":       int(_get("batch-size",         64)),
        "non_iid":          bool(_get("non-iid",           False)),
        "use_dp":           bool(_get("use-dp",            False)),
        "noise_multiplier": float(_get("noise-multiplier", 1.1)),
        "max_grad_norm":    float(_get("max-grad-norm",    1.0)),
        "use_synthetic":    bool(_get("use-synthetic",     False)),
        "fedprox_mu":       float(_get("fedprox-mu",       0.0)),
    }


# ---------------------------------------------------------------------------
# Train handler
# ---------------------------------------------------------------------------

@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the domain-plugin model on this organization's local data."""

    # 1. Parse all experiment flags from server ConfigRecord
    cfg = _read_cfg(msg, context)

    plugin_name    = cfg["plugin_name"]
    lr             = cfg["lr"]
    epochs         = cfg["epochs"]
    batch_size     = cfg["batch_size"]
    non_iid        = cfg["non_iid"]
    use_dp         = cfg["use_dp"]
    noise_mult     = cfg["noise_multiplier"]
    max_grad_norm  = cfg["max_grad_norm"]
    use_synthetic  = cfg["use_synthetic"]
    fedprox_mu     = cfg["fedprox_mu"]

    # 2. Load enterprise plugin & restore global model weights
    plugin = load_plugin(plugin_name)
    model = plugin.build_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Load local data partition
    #    load_partition handles IID/Non-IID split AND synthetic augmentation (E5)
    partition_id   = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    trainloader, _ = plugin.load_partition(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        non_iid=non_iid,
        use_synthetic=use_synthetic,    # CTGAN augmentation happens inside plugin (E5)
    )

    # 4. Train — DPWrapper + FedProx are no-ops unless the flags are set
    train_loss, epsilon = train_tabular(
        net=model,
        trainloader=trainloader,
        epochs=epochs,
        lr=lr,
        device=device,
        use_dp=use_dp,
        noise_multiplier=noise_mult,
        max_grad_norm=max_grad_norm,
        fedprox_mu=fedprox_mu,
    )

    # 5. Send updated weights + metrics back to ServerApp
    model_record = ArrayRecord(model.state_dict())
    metrics = MetricRecord({
        "train_loss":   float(train_loss),
        "epsilon":      float(epsilon) if epsilon != float("inf") else -1.0,
        "num-examples": float(len(trainloader.dataset)),
    })
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)


# ---------------------------------------------------------------------------
# Evaluate handler
# ---------------------------------------------------------------------------

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global model on this organization's local validation split."""

    rc = context.run_config
    plugin_name    = str(rc.get("plugin-name",  "credit_risk"))
    batch_size     = int(rc.get("batch-size",    64))
    non_iid        = bool(rc.get("non-iid",      False))

    # Load plugin and restore global model weights
    plugin = load_plugin(plugin_name)
    model = plugin.build_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load validation split (never synthetic-augmented — honest evaluation)
    partition_id   = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    _, valloader = plugin.load_partition(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        non_iid=non_iid,
        use_synthetic=False,            # val split is always real data
    )

    # Evaluate and return loss, accuracy, F1
    eval_metrics = test_tabular(model, valloader, device)

    metrics = MetricRecord({
        "eval_loss":    float(eval_metrics["loss"]),
        "eval_acc":     float(eval_metrics["accuracy"]),
        "eval_f1":      float(eval_metrics["f1"]),
        "num-examples": float(len(valloader.dataset)),
    })
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)
