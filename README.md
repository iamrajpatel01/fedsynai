# FedSynAI: Enterprise Privacy-Preserving Federated Learning & Synthetic Data Platform

**FedSynAI** is an enterprise-grade platform combining **Federated Learning (Flower 1.36+ Grid Architecture)**, **Differential Privacy (Opacus)**, **Generative Synthetic Data Augmentation (SDV / CTGAN)**, and **Membership Inference Attack (MIA) Auditing** for privacy-critical tabular domains like credit risk and healthcare.

---

## 🌟 Key Features

- **Domain Plugin Architecture**: Pluggable models and datasets implementing the `FedSynAIPlugin` interface. Built-in `credit_risk` tabular dataset with balanced classes and Dirichlet label skew for Non-IID simulation.
- **Modern Flower Grid Architecture**: Full support for Flower 1.36+ `ServerApp` and `ClientApp` communicating via typed `ArrayRecord`, `ConfigRecord`, and `MetricRecord`.
- **Differential Privacy with Opacus**: Client-level DP training using `PrivacyEngine` with per-sample gradient clipping and calibrated Gaussian noise, tracking $(\varepsilon, \delta)$ privacy budget.
- **Generative Synthetic Augmentation (CTGAN / SDV)**: Client-side local data augmentation to mitigate data scarcity and Non-IID performance degradation.
- **Automated MIA Security Audit**: Server-side confidence-score Membership Inference Attack auditor evaluating post-training privacy leakage (ROC-AUC).
- **Interactive Streamlit Dashboard**: Dark glassmorphic comparison dashboard visualizing learning curves, accuracy, F1, and MIA privacy scores across all experiments.

---

## 🔬 Experiment Matrix (E1 – E5)

| ID | Experiment | Non-IID Skew | Differential Privacy | Synthetic Augmentation | Objective |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **E1** | Centralised Baseline | ❌ | ❌ | ❌ | Upper-bound performance ceiling |
| **E2** | Standard FL (FedAvg) | ❌ | ❌ | ❌ | Baseline collaborative learning |
| **E3** | Non-IID FL (FedProx) | ✅ ($\alpha=0.5, \mu=0.01$) | ❌ | ❌ | Heterogeneous data robustness |
| **E4** | Differential Privacy FL | ❌ | ✅ ($\varepsilon=2.58, \delta=10^{-5}$) | ❌ | Provable privacy protection |
| **E5** | FedSynAI (Full Pipeline) | ✅ ($\mu=0.01$) | ✅ ($\varepsilon=1.62$) | ✅ CTGAN ($1\times$) | Privacy + Synthetic Augmentation |

---

## 🚀 Quickstart

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/iamrajpatel01/fedsynai.git
cd fedsynai

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux / macOS

# Install dependencies
pip install -e .
```

### 2. Configure 4 Simulated SuperNodes

```powershell
flwr federation simulation-config @none/default local --num-supernodes 4
```

### 3. Run Experiments

```powershell
# E1: Centralised Baseline (no Flower required)
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

### 4. Launch the Comparison Dashboard

```powershell
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) (or displayed port) to view live charts and metrics.

---

## 📂 Project Structure

```
fedsynai/
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
├── results/                       # Auto-generated metrics_*.json logs
├── pyproject.toml                 # Flower app config & dependencies
├── run_e1_centralized.py          # E1 Centralised Baseline runner
└── README.md
```

---

## 🛡️ License

This project is licensed under the Apache 2.0 License.
