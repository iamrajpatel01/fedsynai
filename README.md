# FedSynAI: Enterprise Privacy-Preserving Federated Learning & Synthetic Data Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flower 1.36+](https://img.shields.io/badge/Flower-1.36+-FF6F00.svg)](https://flower.ai/)
[![Opacus DP](https://img.shields.io/badge/Privacy-Opacus%20DP-success.svg)](https://opacus.ai/)
[![SDV CTGAN](https://img.shields.io/badge/Synthetic-SDV%20CTGAN-purple.svg)](https://sdv.dev/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**FedSynAI** is an enterprise-grade platform uniting **Federated Learning (Flower Grid Architecture)**, **Differential Privacy (Opacus)**, **Generative Synthetic Data Augmentation (SDV / CTGAN)**, and **Membership Inference Attack (MIA) Auditing** for privacy-critical tabular domains like credit risk and healthcare.

---

## 🌟 Key Features

- **Domain Plugin Architecture**: Pluggable models and datasets implementing the `FedSynAIPlugin` interface. Built-in `credit_risk` tabular dataset with balanced classes and Dirichlet label skew for Non-IID simulation.
- **Modern Flower Grid Architecture**: Full support for Flower 1.36+ `ServerApp` and `ClientApp` communicating via typed `ArrayRecord`, `ConfigRecord`, and `MetricRecord`.
- **Differential Privacy with Opacus**: Client-level DP training using `PrivacyEngine` with per-sample gradient clipping and calibrated Gaussian noise, tracking $(\varepsilon, \delta)$ privacy expenditure.
- **Generative Synthetic Augmentation (CTGAN / SDV)**: Client-side local data augmentation to mitigate data scarcity and Non-IID performance degradation.
- **Automated MIA Security Audit**: Server-side confidence-score Membership Inference Attack auditor evaluating post-training privacy leakage (ROC-AUC).
- **Interactive Streamlit Dashboard**: Dark glassmorphic comparison dashboard visualizing learning curves, accuracy, F1, and MIA privacy scores across all experiments.

For in-depth mathematical formulations, threat models, and software design patterns, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🔬 Benchmark Results (E1 – E5)

Below are the actual benchmark metrics logged in `results/` across all 5 configurations on the 4-node simulated federation:

| ID | Experiment Name | Non-IID Skew | DP Budget ($\varepsilon$) | Synthetic Augmentation | Final Accuracy | Weighted F1 | Final Loss | MIA Privacy (AUC) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | **Centralised Baseline** | ❌ (IID) | ❌ None | ❌ None | **92.63%** | **0.9254** | `0.1996` | `0.4909` |
| **E2** | **Standard FL (FedAvg)** | ❌ (IID) | ❌ None | ❌ None | **82.50%** | **0.8019** | `0.4266` | `0.4938` |
| **E3** | **Non-IID FL (FedProx)** | ✅ ($\alpha=0.5, \mu=0.01$) | ❌ None | ❌ None | **91.13%** | **0.9087** | `0.2083` | `0.4921` |
| **E4** | **Differential Privacy FL** | ❌ (IID) | ✅ $\varepsilon=2.58, \delta=10^{-5}$ | ❌ None | **68.75%** | **0.5602** | `0.6546` | `0.4937` |
| **E5** | **FedSynAI (Full Pipeline)** | ✅ ($\mu=0.01$) | ✅ $\varepsilon=1.62, \delta=10^{-5}$ | ✅ CTGAN ($1\times$) | **68.75%** | **0.5602** | `0.6708` | `0.4958` |

> **Privacy Note**: An MIA ROC-AUC score of $\approx 0.50$ signifies that an adversary attempting to infer membership in the training set cannot distinguish members from non-members better than a random coin toss.

---

## 🚀 Quickstart

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/iamrajpatel01/fedsynai.git
cd fedsynai

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows PowerShell
# source .venv/bin/activate # Linux / macOS

# Install dependencies via requirements.txt or pip install -e .
pip install -r requirements.txt
pip install -e .
```

### 2. Configure 4 Simulated SuperNodes

```powershell
flwr federation simulation-config @none/default local --num-supernodes 4
```

### 3. Windows PowerShell Encoding Setup (Mandatory on Windows)

```powershell
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### 4. Run Experiments

```powershell
# E1: Centralised Baseline (standalone PyTorch upper-bound)
python run_e1_centralized.py

# E2: Standard Federated Learning (FedAvg, IID)
flwr run . --stream

# E3: Non-IID FL with FedProx (alpha=0.5 label skew, mu=0.01)
flwr run . --run-config "non-iid=true fedprox-mu=0.01" --stream

# E4: Differential Privacy FL (Opacus PrivacyEngine, noise=1.1, clip=1.0)
flwr run . --run-config "use-dp=true noise-multiplier=1.1 max-grad-norm=1.0" --stream

# E5: Full FedSynAI Pipeline (Non-IID + DP + Synthetic CTGAN Augmentation)
flwr run . --run-config "non-iid=true use-dp=true use-synthetic=true fedprox-mu=0.01" --stream
```

### 5. Launch the Comparison Dashboard

```powershell
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) (or displayed port) to view live KPI cards, interactive learning curves, and comparative charts.

---

## 📂 Project Structure

```
fedsynai/
├── ARCHITECTURE.md                # System design, math formulations & threat model
├── LICENSE                        # Apache 2.0 open-source license
├── README.md                      # Platform guide & quickstart
├── pyproject.toml                 # Flower app config & dependencies
├── requirements.txt               # Pinned dependencies specification
├── run_e1_centralized.py          # E1 Centralised Baseline runner
├── final_model_credit_risk.pt     # Saved final global model weights
├── dashboard/
│   └── app.py                     # Streamlit comparison dashboard
├── data/
│   └── cs-training.csv            # Cached credit risk tabular dataset
├── pytorchexample/
│   ├── plugins/
│   │   ├── __init__.py            # Plugin package & dynamic loader
│   │   ├── interfaces.py          # FedSynAIPlugin abstract base class
│   │   └── credit_risk.py         # Credit Risk MLP plugin with Non-IID splitting
│   ├── client_app.py              # Flower ClientApp (DP + FedProx + Synth)
│   ├── server_app.py              # Flower ServerApp (FedAvg + MIA Security Audit)
│   ├── task.py                    # Tabular train_tabular & test_tabular engines
│   ├── privacy_engine.py          # Opacus DPWrapper module
│   ├── synth_engine.py            # SDV / CTGAN synthetic augmentation wrapper
│   └── security_audit.py          # Membership Inference Attack auditor
└── results/                       # Auto-generated metrics_*.json logs
    ├── metrics_e1_centralized.json
    ├── metrics_credit_risk_dpFalse_synthFalse_noniidFalse.json      # E2
    ├── metrics_credit_risk_dpFalse_synthFalse_noniidTrue.json       # E3
    ├── metrics_credit_risk_dpTrue_synthFalse_noniidFalse.json       # E4
    └── metrics_credit_risk_dpTrue_synthTrue_noniidTrue.json        # E5
```

---

## 🛡️ License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
