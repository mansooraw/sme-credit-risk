"""Model insights: performance, drivers, tuning and behaviour over time."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

import experience as X  # noqa: E402
import utils as U  # noqa: E402
from src import config  # noqa: E402

X.page_start("insights")
U.page_header("Model insights", "What the model relies on, how it compares with the alternatives, how well its "
              "probabilities match observed default rates and how it holds up over time.", eyebrow="Evaluation")
meta = U.load_metadata()


def find_figure(*words: str) -> Path | None:
    if not config.FIGURES_DIR.exists():
        return None
    for path in sorted(config.FIGURES_DIR.glob("*.png")):
        stem = path.stem.lower()
        if not stem.startswith("app_") and all(w in stem for w in words):
            return path
    return None


tab_perf, tab_drivers, tab_tuning, tab_time = st.tabs(["Performance", "Drivers", "Tuning", "Over time"])

with tab_perf:
    metrics = meta.get("metrics_test") or {}
    names = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "recall": "Recall", "precision": "Precision", "f1": "F1",
             "brier": "Brier score", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy"}
    accents = [U.COLOURS["navy"], U.COLOURS["teal"], U.COLOURS["teal"], U.COLOURS["amber"]]
    helps = {"roc_auc": "Area under the ROC curve on the test split.", "pr_auc": "Area under the precision-recall curve on the test split.",
             "recall": "Defaults with PD at or above the threshold, divided by all defaults.",
             "precision": "Defaults among the loans flagged at or above the threshold.",
             "f1": "Harmonic mean of precision and recall at the threshold.",
             "brier": "Mean squared gap between the predicted PD and the outcome (0 or 1); lower is better.",
             "accuracy": "Share of test loans classified correctly at the threshold.",
             "balanced_accuracy": "Average of recall on defaults and recall on repaid loans."}
    items = [{"label": names[k], "value": U.fmt_metric(metrics[k]), "accent": accents[i % 4],
              "help": helps[k] + " From the model metadata."}
             for i, k in enumerate(k for k in names if k in metrics)]
    if items:
        U.tiles(items, cols=4, min_width=200)
        cv = meta.get("metrics_cv") or {}
        cv_mean, cv_std = U.find_value(cv, ["roc_auc_mean", "mean", "roc_auc"]), U.find_value(cv, ["roc_auc_std", "std"])
        cap = "Test-set metrics at the business threshold, from the model metadata."
        if cv_mean is not None:
            cap += f" Cross-validated ROC-AUC on the training data: {U.fmt_metric(cv_mean)}"
            cap += f" (standard deviation {U.fmt_metric(cv_std)})." if cv_std is not None else "."
        st.caption(cap)

    portfolio, source = U.scored_portfolio(U.model_version_key())
    pp, yy = portfolio["p"].to_numpy(), portfolio["Default"].to_numpy().astype(int)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        frac_pos, mean_pred = calibration_curve(yy, pp, n_bins=10, strategy="quantile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration",
                                 line=dict(color=U.COLOURS["grey"], dash="dot", width=1.5)))
        fig.add_trace(go.Scatter(x=mean_pred * 100, y=frac_pos * 100, mode="lines+markers", name="Model",
                                 line=dict(color=U.COLOURS["navy"], width=3),
                                 marker=dict(size=9, color=U.COLOURS["teal"], line=dict(color="#FFFFFF", width=2)),
                                 hovertemplate="Predicted %{x:.1f}%<br>Observed %{y:.1f}%<extra></extra>"))
        fig.update_layout(title="Calibration by decile of PD", xaxis=dict(title="Mean predicted PD (%)", range=[0, 100]),
                          yaxis=dict(title="Observed default rate (%)", range=[0, 100]))
        U.chart(fig, height=380, key="calibration")
    with c2:
        fpr, tpr, _ = roc_curve(yy, pp)
        idx = np.unique(np.linspace(0, len(fpr) - 1, 400).astype(int))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random ranking",
                                 line=dict(color=U.COLOURS["grey"], dash="dot", width=1.5)))
        fig.add_trace(go.Scatter(x=fpr[idx], y=tpr[idx], mode="lines", name=f"Model, AUC {roc_auc_score(yy, pp):.3f}",
                                 line=dict(color=U.COLOURS["teal"], width=3.5), fill="tozeroy",
                                 fillcolor=U.hex_rgba(U.COLOURS["teal"], 0.08)))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        U.chart(fig, height=380, key="roc")
    st.caption(f"Computed in the app from the {source}.")
    figs = [f for f in (find_figure("fig05", "calibration"), find_figure("roc")) if f is not None]
    if figs:
        with st.expander("Evaluation figures from the notebooks"):
            for f in figs:
                st.image(str(f), caption=f.name, use_container_width=True)

    br = meta.get("business_results")
    if isinstance(br, dict) and br:
        U.section_title("Business results on the held-out portfolio", "coins")
        policies = {k: v for k, v in br.items() if isinstance(v, dict)}
        if policies:
            frame = pd.DataFrame(policies).T
            frame = frame.apply(lambda col: col.map(lambda v: U.format_business_value(col.name, v)))
            frame.index = [U.prettify_key(i) for i in frame.index]
            frame.columns = [U.prettify_key(c) for c in frame.columns]
            st.dataframe(frame, use_container_width=True)
        scalars = [(k, v) for k, v in br.items() if not isinstance(v, (dict, list))]
        if scalars:
            st.dataframe(pd.DataFrame([(U.prettify_key(k), U.format_business_value(k, v)) for k, v in scalars],
                                      columns=["Measure", "Value"]), hide_index=True, use_container_width=True,
                         height=35 * (len(scalars) + 1) + 3)
            st.caption("From the model metadata. Rates are shares of the held-out loans; values in USD use the cost assumptions.")

with tab_drivers:
    g = U.global_importance(15)
    if g is not None:
        g = g.iloc[::-1]
        fig = go.Figure()
        for _, r in g.iterrows():
            fig.add_shape(type="line", x0=0, x1=r["importance"], y0=r["label"], y1=r["label"],
                          line=dict(color=U.hex_rgba(U.COLOURS["navy"], 0.35), width=3))
        fig.add_trace(go.Scatter(x=g["importance"], y=g["label"], mode="markers+text",
                                 marker=dict(size=14, color=U.COLOURS["teal"], line=dict(color="#FFFFFF", width=2.5)),
                                 text=[f"{v:.2f}" for v in g["importance"]], textposition="middle right",
                                 textfont=dict(family=U.MONO_STACK, size=11, color=U.MUTED),
                                 hovertemplate="%{y}<br>Mean |SHAP| %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Global importance: mean absolute SHAP value", showlegend=False,
                          xaxis=dict(title="Mean absolute SHAP value (log-odds), one-hot levels summed per input",
                                     range=[0, float(g["importance"].max()) * 1.18]),
                          yaxis=dict(showgrid=False), margin=dict(l=8, r=8, t=44, b=44))
        U.chart(fig, height=36 * len(g) + 110, key="importance")
        st.caption("A longer stem means the input moves the score more on average, in either direction. "
                   "Source: shap_global.csv stored with the model.")
    else:
        st.info("Global importance is not available for the loaded model.")

with tab_tuning:
    table = U.records_table(meta.get("comparison_table"))
    U.section_title("Model comparison", "grid")
    if table is not None:
        st.dataframe(U.pretty_table(table), hide_index=True, use_container_width=True)
        st.caption("All models were trained and tested on the same split; source: model metadata.")
        label_col = table.columns[0]
        auc_col = next((c for c in ("test_roc_auc", "roc_auc") if c in table.columns), None)
        if auc_col and pd.api.types.is_numeric_dtype(table[auc_col]):
            srt = table.sort_values(auc_col)
            fig = go.Figure(go.Bar(x=srt[auc_col], y=srt[label_col].astype(str), orientation="h",
                                   marker=dict(color=[U.COLOURS["teal"] if i == len(srt) - 1 else U.hex_rgba(U.COLOURS["navy"], 0.55)
                                                      for i in range(len(srt))]),
                                   text=[f"{v:.3f}" for v in srt[auc_col]], textposition="outside",
                                   textfont=dict(family=U.MONO_STACK, size=11)))
            fig.update_layout(title="Test ROC-AUC by model", barcornerradius=6, xaxis=dict(range=[0, 1.08], title="ROC-AUC"))
            U.chart(fig, height=60 * len(srt) + 100, key="comparison")
    else:
        st.info("No comparison table in the metadata.")
    U.section_title("Tuning: before and after", "sliders")
    table = U.records_table(meta.get("before_after_tuning"))
    if table is not None:
        st.dataframe(U.pretty_table(table), hide_index=True, use_container_width=True)
        st.caption("Default parameters against the tuned parameters on the same test set; source: model metadata.")
    else:
        st.info("No tuning table in the metadata.")

with tab_time:
    tv = meta.get("temporal_validation")
    if isinstance(tv, dict) and tv:
        rows = U.flatten(tv)
        num = [(k, v) for k, v in rows if isinstance(v, (int, float)) and not isinstance(v, bool)]
        preferred = ["roc_auc", "recall", "pr_auc", "precision", "test_default_rate", "mean_predicted_pd"]
        num = sorted(num, key=lambda kv: preferred.index(kv[0]) if kv[0] in preferred else len(preferred))
        if num:
            U.tiles([{"label": U.prettify_key(k), "value": U.fmt_metric(v) if abs(v) <= 1.5 else f"{v:,.0f}",
                      "accent": U.COLOURS["navy"]} for k, v in num[:4]], cols=min(4, len(num[:4])))
        st.dataframe(pd.DataFrame([(U.prettify_key(k), str(v)) for k, v in rows], columns=["Item", "Value"]),
                     hide_index=True, use_container_width=True)
        st.caption("The model is retrained on older approvals and tested on later ones to show how performance "
                   "changes over time. Recent years are affected by right-censoring: loans approved shortly before "
                   "the data ends have had little time to default.")
    else:
        st.info("No temporal validation results in the metadata.")
