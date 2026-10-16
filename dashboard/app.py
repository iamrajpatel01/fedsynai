"""FedSynAI Dashboard — Streamlit app for comparing E1–E5 experiments.

Reads all results/metrics_*.json files produced by server_app.py and
run_e1_centralized.py, then renders:
  • Side-by-side accuracy / F1 bar chart across experiments
  • Per-round loss & accuracy line charts for each FL experiment
  • MIA AUC privacy leakage comparison table
  • Full config table for each experiment

Usage:
    pip install streamlit plotly
    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import streamlit as st

# ---- Page config -------------------------------------------------------------
st.set_page_config(
    page_title="FedSynAI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS --------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 4px;
    }
    .experiment-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: #21262d;
        border: 1px solid #30363d;
        color: #c9d1d9;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e6edf3;
        border-left: 3px solid #58a6ff;
        padding-left: 10px;
        margin: 24px 0 12px 0;
    }
    .privacy-good  { color: #3fb950; font-weight: 600; }
    .privacy-bad   { color: #f85149; font-weight: 600; }
    .privacy-mid   { color: #d29922; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---- Data loading ------------------------------------------------------------
POSSIBLE_DIRS = [
    Path(__file__).resolve().parent.parent / "results",
    Path(__file__).resolve().parent.parent.parent / "results",
    Path("results").resolve(),
]

EXPERIMENT_LABELS = {
    "e1_centralized":                              "E1 - Centralised Baseline",
    "credit_risk_dpFalse_synthFalse_noniidFalse":  "E2 - Standard FL (FedAvg, IID)",
    "credit_risk_dpFalse_synthFalse_noniidTrue":   "E3 - Non-IID FL (FedProx)",
    "credit_risk_dpTrue_synthFalse_noniidFalse":   "E4 - DP-FL",
    "credit_risk_dpTrue_synthTrue_noniidTrue":      "E5 - FedSynAI (Full Pipeline)",
}

EXPERIMENT_COLORS = {
    "E1 - Centralised Baseline":         "#8b949e",
    "E2 - Standard FL (FedAvg, IID)":    "#58a6ff",
    "E3 - Non-IID FL (FedProx)":         "#d29922",
    "E4 - DP-FL":                        "#f0883e",
    "E5 - FedSynAI (Full Pipeline)":     "#3fb950",
}


@st.cache_data(ttl=10)
def load_all_results() -> dict[str, dict]:
    """Load every metrics JSON from project or workspace results directories."""
    results = {}
    found_paths: set[str] = set()
    for d in POSSIBLE_DIRS:
        if not d.exists():
            continue
        for path in sorted(d.glob("metrics_*.json")):
            if path.name in found_paths:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                exp_key = data.get("experiment", path.stem.replace("metrics_", ""))
                label = EXPERIMENT_LABELS.get(exp_key, exp_key)
                results[label] = data
                found_paths.add(path.name)
            except Exception:
                pass
    return results



# ---- Header ------------------------------------------------------------------
st.markdown("""
<div style="padding: 16px 0 8px 0;">
  <h1 style="color:#e6edf3; font-size:2rem; margin:0;">🛡️ FedSynAI Dashboard</h1>
  <p style="color:#8b949e; margin:4px 0 0 0; font-size:0.95rem;">
    Enterprise Privacy-Preserving Federated Learning — Experiment Comparison (E1–E5)
  </p>
</div>
<hr style="border-color:#30363d; margin:12px 0 24px 0;">
""", unsafe_allow_html=True)

# ---- Load data ---------------------------------------------------------------
results = load_all_results()

if not results:
    st.warning(
        "⚠️ No results found in `results/` directory.\n\n"
        "Run at least one experiment first:\n"
        "```\npython run_e1_centralized.py\n```\n"
        "or\n"
        "```\nflwr run .\n```"
    )
    st.stop()

# ---- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    selected_exps = st.multiselect(
        "Show experiments",
        options=list(results.keys()),
        default=list(results.keys()),
    )
    st.markdown("---")
    st.markdown("### 📋 Quick Commands")
    st.code("python run_e1_centralized.py", language="bash")
    st.code("flwr run .", language="bash")
    st.code('flwr run . --run-config "use-dp=true"', language="bash")
    st.code('flwr run . --run-config "non-iid=true fedprox-mu=0.01"', language="bash")
    st.code('flwr run . --run-config "non-iid=true use-dp=true use-synthetic=true"', language="bash")
    st.markdown("---")
    if st.button("🔄 Refresh Results"):
        st.cache_data.clear()
        st.rerun()

filtered = {k: v for k, v in results.items() if k in selected_exps}

if not filtered:
    st.info("Select at least one experiment in the sidebar.")
    st.stop()


# ---- Helper: extract final-round metrics ------------------------------------
def final_metrics(data: dict) -> dict:
    rounds = data.get("rounds", [])
    if rounds:
        last = rounds[-1]
        return {
            "accuracy": last.get("accuracy", 0.0),
            "f1":       last.get("f1", 0.0),
            "loss":     last.get("loss", 0.0),
        }
    m = data.get("metrics", {})
    return {
        "accuracy": m.get("accuracy", 0.0),
        "f1":       m.get("f1", 0.0),
        "loss":     m.get("test_loss", m.get("loss", 0.0)),
    }


# ============================================================
# SECTION 1: Top-line KPI cards
# ============================================================
st.markdown('<div class="section-header">📊 Final Round Performance</div>', unsafe_allow_html=True)

cols = st.columns(len(filtered))
for col, (label, data) in zip(cols, filtered.items()):
    fm = final_metrics(data)
    mia = data.get("mia_auc", -1.0)
    mia_str = f"{mia:.3f}" if mia >= 0 else "N/A"
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:0.75rem;color:#8b949e;margin-bottom:8px;">{label}</div>
          <div class="metric-value">{fm['accuracy']:.1%}</div>
          <div class="metric-label">Accuracy</div>
          <div style="margin-top:10px;font-size:0.85rem;color:#8b949e;">
            F1 {fm['f1']:.3f} &nbsp;|&nbsp; MIA {mia_str}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# SECTION 2: Bar chart — Accuracy & F1 comparison
# ============================================================
try:
    import plotly.graph_objects as go

    st.markdown('<div class="section-header">📈 Accuracy & F1 Comparison</div>', unsafe_allow_html=True)

    labels_plot = list(filtered.keys())
    accs  = [final_metrics(filtered[l])["accuracy"] for l in labels_plot]
    f1s   = [final_metrics(filtered[l])["f1"]       for l in labels_plot]
    colors = [EXPERIMENT_COLORS.get(l, "#58a6ff") for l in labels_plot]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Accuracy", x=labels_plot, y=accs,
        marker_color=colors, opacity=0.85,
        text=[f"{v:.1%}" for v in accs], textposition="outside",
    ))
    fig_bar.add_trace(go.Bar(
        name="Weighted F1", x=labels_plot, y=f1s,
        marker_color=colors, opacity=0.45,
        text=[f"{v:.3f}" for v in f1s], textposition="outside",
    ))
    fig_bar.update_layout(
        barmode="group",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#c9d1d9", font_family="Inter",
        legend=dict(bgcolor="#21262d", bordercolor="#30363d", borderwidth=1),
        yaxis=dict(tickformat=".0%", gridcolor="#21262d", range=[0, 1.12]),
        xaxis=dict(tickangle=-15),
        margin=dict(t=20, b=20),
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Per-round line charts ----------------------------------------------
    has_rounds = {l: d for l, d in filtered.items() if len(d.get("rounds", [])) > 1}
    if has_rounds:
        st.markdown('<div class="section-header">📉 Per-Round Learning Curves</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        fig_loss = go.Figure()
        fig_acc  = go.Figure()
        for label, data in has_rounds.items():
            rounds_data = data["rounds"]
            xs    = [r["round"] for r in rounds_data]
            losses = [r.get("loss", 0) for r in rounds_data]
            accs_r = [r.get("accuracy", 0) for r in rounds_data]
            color  = EXPERIMENT_COLORS.get(label, "#58a6ff")
            fig_loss.add_trace(go.Scatter(x=xs, y=losses, mode="lines+markers",
                                          name=label, line=dict(color=color, width=2)))
            fig_acc.add_trace(go.Scatter(x=xs, y=accs_r, mode="lines+markers",
                                         name=label, line=dict(color=color, width=2)))

        for fig, title, yformat in [
            (fig_loss, "Global Loss per Round", ".4f"),
            (fig_acc,  "Global Accuracy per Round", ".1%"),
        ]:
            fig.update_layout(
                title=dict(text=title, font=dict(size=14, color="#e6edf3")),
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#c9d1d9", font_family="Inter",
                legend=dict(bgcolor="#21262d", bordercolor="#30363d", borderwidth=1,
                            font=dict(size=10)),
                xaxis=dict(title="Round", gridcolor="#21262d"),
                yaxis=dict(tickformat=yformat, gridcolor="#21262d"),
                margin=dict(t=40, b=20), height=340,
            )

        with c1:
            st.plotly_chart(fig_loss, use_container_width=True)
        with c2:
            st.plotly_chart(fig_acc,  use_container_width=True)

    # ---- MIA Privacy AUC chart ----------------------------------------------
    mia_data = {l: d.get("mia_auc", -1) for l, d in filtered.items() if d.get("mia_auc", -1) >= 0}
    if mia_data:
        st.markdown('<div class="section-header">🔒 Privacy Leakage — MIA AUC</div>', unsafe_allow_html=True)
        st.caption("AUC = 0.50 → perfect privacy (random guess) &nbsp;|&nbsp; AUC = 1.00 → complete leakage")

        mia_labels = list(mia_data.keys())
        mia_vals   = list(mia_data.values())
        mia_colors = []
        for v in mia_vals:
            if v < 0.60:   mia_colors.append("#3fb950")   # green  = private
            elif v < 0.75: mia_colors.append("#d29922")   # yellow = moderate
            else:          mia_colors.append("#f85149")   # red    = leaky

        fig_mia = go.Figure(go.Bar(
            x=mia_labels, y=mia_vals,
            marker_color=mia_colors,
            text=[f"{v:.3f}" for v in mia_vals], textposition="outside",
        ))
        fig_mia.add_hline(y=0.5, line_dash="dash", line_color="#3fb950",
                           annotation_text="0.50 = perfect privacy",
                           annotation_font_color="#3fb950")
        fig_mia.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", font_family="Inter",
            yaxis=dict(range=[0.4, 1.05], gridcolor="#21262d"),
            xaxis=dict(tickangle=-15),
            margin=dict(t=20, b=20), height=340,
            showlegend=False,
        )
        st.plotly_chart(fig_mia, use_container_width=True)

except ImportError:
    st.warning("Install plotly for interactive charts: `pip install plotly`")


# ============================================================
# SECTION 3: Results table
# ============================================================
st.markdown('<div class="section-header">📋 Full Results Table</div>', unsafe_allow_html=True)

table_rows = []
for label, data in filtered.items():
    fm   = final_metrics(data)
    mia  = data.get("mia_auc", -1.0)
    cfg  = data.get("config", {})
    table_rows.append({
        "Experiment":    label,
        "Accuracy":      f"{fm['accuracy']:.4f}",
        "F1 (weighted)": f"{fm['f1']:.4f}",
        "Final Loss":    f"{fm['loss']:.4f}",
        "MIA AUC":       f"{mia:.4f}" if mia >= 0 else "—",
        "DP":            "✅" if cfg.get("use_dp") else "❌",
        "Non-IID":       "✅" if cfg.get("non_iid") else "❌",
        "Synthetic":     "✅" if cfg.get("use_synthetic") else "❌",
        "Rounds":        str(len(data.get("rounds", []))),
    })

try:
    import pandas as pd
    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
except ImportError:
    for row in table_rows:
        st.write(row)


# ---- Footer ------------------------------------------------------------------
st.markdown("""
<hr style="border-color:#30363d;margin-top:40px;">
<p style="color:#484f58;font-size:0.78rem;text-align:center;">
  FedSynAI · Privacy-Preserving Federated Learning Platform ·
  Results auto-refresh every 10 s when new JSON files appear in <code>results/</code>
</p>
""", unsafe_allow_html=True)
