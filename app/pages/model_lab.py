"""Model lab: the final and comparison models side by side on ROC, precision-recall and calibration, plus tuning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_curve  # noqa: E402

import experience as X  # noqa: E402
import utils as U  # noqa: E402

X.page_start("lab")
U.page_header(
    "Model lab",
    "The final model against the comparison model on the same held-out loans. Move the threshold, or click a point on "
    "the ROC curve, and the marker moves along both curves.",
    eyebrow="Model comparison",
)
meta = U.load_metadata()
slot = st.empty()
slot.markdown(U.skeleton(300, "Scoring the held-out loans with both models"), unsafe_allow_html=True)
scores, source = U.model_lab_scores(U.model_version_key())
slot.empty()
y = scores["Default"].to_numpy()
final_name = str(meta.get("model_name", "Final model"))
comp_name = "Logistic regression" if U.load_comparison_model() is not None else None
MODELS = [("p", final_name, U.COLOURS["teal"])]
if "p_comp" in scores:
    MODELS.append(("p_comp", f"Comparison: {comp_name}", U.COLOURS["navy"]))
MAX_POINTS = 500


@st.cache_data(show_spinner=False)
def curves(model_key: str, col: str) -> dict:
    p = scores[col].to_numpy()
    fpr, tpr, thr_roc = roc_curve(y, p)
    prec, rec, thr_pr = precision_recall_curve(y, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, MAX_POINTS).astype(int))
    keep_pr = np.unique(np.linspace(0, len(thr_pr) - 1, MAX_POINTS).astype(int))
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {"fpr": fpr[keep], "tpr": tpr[keep], "thr": np.clip(thr_roc[keep], 0, 1), "auc": float(auc(fpr, tpr)),
            "prec": prec[keep_pr], "rec": rec[keep_pr], "thr_pr": thr_pr[keep_pr], "pr_auc": float(auc(rec, prec)),
            "cal_x": mean, "cal_y": frac, "brier": float(np.mean((p - y) ** 2))}


def point_at(col: str, t: float) -> dict:
    p = scores[col].to_numpy()
    flag = p >= t  # flagged as risky (not approved)
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    return {"tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1), "precision": tp / max(tp + fp, 1),
            "approval": float((~flag).mean())}


cv = {k: curves(U.model_version_key(), k) for k, *_ in MODELS}
t = U.threshold_slider("Threshold (probability of default)", key="thr_lab")
pts = {k: point_at(k, t) for k, *_ in MODELS}

U.tiles([{"label": name.split(":")[-1].strip()[:26], "value": f"{cv[k]['auc']:.3f}", "accent": colour,
          "sub": f"ROC-AUC; PR-AUC {cv[k]['pr_auc']:.3f}; recall {pts[k]['tpr']:.1%} at {t:.3f}",
          "help": "ROC-AUC and PR-AUC computed in the app on the held-out loans; recall is the share of defaults with PD at or "
                  "above the selected threshold."} for k, name, colour in MODELS], cols=len(MODELS))

c1, c2 = st.columns(2, gap="large")
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=U.COLOURS["grey"], dash="dot", width=1.3),
                             hoverinfo="skip"))
    for k, name, colour in MODELS:
        c = cv[k]
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines+markers", name=f"{name.split(':')[-1].strip()} ({c['auc']:.3f})",
                                 line=dict(color=colour, width=3), marker=dict(size=4, opacity=0.0),
                                 customdata=c["thr"],
                                 hovertemplate="Threshold %{customdata:.3f}<br>False positive rate %{x:.1%}"
                                               "<br>True positive rate %{y:.1%}<br>Click to use this threshold<extra>" + name.split(":")[-1].strip() + "</extra>"))
    for k, name, colour in MODELS:
        fig.add_trace(go.Scatter(x=[pts[k]["fpr"]], y=[pts[k]["tpr"]], mode="markers", showlegend=False,
                                 marker=dict(size=15, color=colour, symbol="circle", line=dict(color="#FFFFFF", width=3)),
                                 hovertemplate=f"Threshold {t:.3f}<br>FPR %{{x:.1%}}<br>TPR %{{y:.1%}}<extra></extra>"))
    fig.update_layout(title="ROC curve", xaxis=dict(title="False positive rate (good loans not approved)", range=[0, 1]),
                      yaxis=dict(title="True positive rate (defaults stopped)", range=[0, 1.02]), clickmode="event+select",
                      legend=dict(y=1.04), margin=dict(l=8, r=8, t=90, b=40), transition=dict(duration=450, easing="cubic-in-out"))
    event = U.chart(fig, height=430, key=f"lab_roc_{st.session_state.get('lab_v', 0)}", on_select="rerun")
    sel = list(event.selection.points) if event is not None and hasattr(event, "selection") else []
    sig = repr(sel)
    if sel and st.session_state.get("_lab_seen") != sig:
        st.session_state["_lab_seen"] = sig
        pt = sel[0]
        curve_no = int(pt.get("curve_number", 1)) - 1
        idx = int(pt.get("point_index", pt.get("point_number", 0)))
        if 0 <= curve_no < len(MODELS):
            thr = cv[MODELS[curve_no][0]]["thr"]
            if 0 <= idx < len(thr):
                U.set_shared_threshold(float(np.clip(round(thr[idx] / 0.005) * 0.005, 0.01, 0.99)))
                st.rerun()
with c2:
    fig = go.Figure()
    base = float(y.mean())
    fig.add_hline(y=base, line_dash="dot", line_color=U.COLOURS["grey"], annotation_text=f"default rate {base:.1%}",
                  annotation_position="bottom left", annotation_font=dict(size=11, color=U.MUTED))
    for k, name, colour in MODELS:
        c = cv[k]
        fig.add_trace(go.Scatter(x=c["rec"], y=c["prec"], mode="lines", name=f"{name.split(':')[-1].strip()} ({c['pr_auc']:.3f})",
                                 line=dict(color=colour, width=3), customdata=np.append(c["thr_pr"], np.nan)[: len(c["rec"])],
                                 hovertemplate="Threshold %{customdata:.3f}<br>Recall %{x:.1%}<br>Precision %{y:.1%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[pts[k]["tpr"]], y=[pts[k]["precision"]], mode="markers", showlegend=False,
                                 marker=dict(size=15, color=colour, line=dict(color="#FFFFFF", width=3)),
                                 hovertemplate=f"Threshold {t:.3f}<br>Recall %{{x:.1%}}<br>Precision %{{y:.1%}}"
                                               f"<br>Approval rate {pts[k]['approval']:.1%}<extra></extra>"))
    fig.update_layout(title="Precision-recall curve", xaxis=dict(title="Recall on defaults", range=[0, 1]),
                      yaxis=dict(title="Precision", range=[0, 1.02]), legend=dict(y=1.04),
                      margin=dict(l=8, r=8, t=90, b=40), transition=dict(duration=450, easing="cubic-in-out"))
    U.chart(fig, height=430, key="lab_pr")

c1, c2 = st.columns(2, gap="large")
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration",
                             line=dict(color=U.COLOURS["grey"], dash="dot", width=1.3), hoverinfo="skip"))
    for k, name, colour in MODELS:
        c = cv[k]
        fig.add_trace(go.Scatter(x=c["cal_x"] * 100, y=c["cal_y"] * 100, mode="lines+markers",
                                 name=f"{name.split(':')[-1].strip()} (Brier {c['brier']:.3f})",
                                 line=dict(color=colour, width=3), marker=dict(size=9, line=dict(color="#FFFFFF", width=2)),
                                 hovertemplate="Mean predicted PD %{x:.1f}%<br>Observed default rate %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="Calibration by decile of PD", xaxis=dict(title="Mean predicted PD (%)", range=[0, 100]),
                      yaxis=dict(title="Observed default rate (%)", range=[0, 100]), legend=dict(y=1.02),
                      margin=dict(l=8, r=8, t=120, b=40))
    U.chart(fig, height=420, key="lab_cal")
with c2:
    tuning = U.records_table(meta.get("before_after_tuning"))
    if tuning is not None and {"model", "test_roc_auc"} <= set(tuning.columns):
        stage = next((c for c in ("version", "stage") if c in tuning.columns), None)
        fig = go.Figure()
        palette = [U.COLOURS["teal"], U.COLOURS["navy"], U.COLOURS["amber"], U.COLOURS["red"], U.COLOURS["grey"]]
        for i, (m, g) in enumerate(tuning.groupby("model", sort=False)):
            g = g.reset_index(drop=True)
            labels = [str(s).capitalize() for s in (g[stage] if stage else range(len(g)))]
            fig.add_trace(go.Scatter(x=labels, y=g["test_roc_auc"], mode="lines+markers+text", name=m,
                                     line=dict(color=palette[i % 5], width=3), marker=dict(size=12, line=dict(color="#FFFFFF", width=2)),
                                     text=[f"{v:.3f}" for v in g["test_roc_auc"]], textposition=["middle left", "middle right"][: len(g)],
                                     textfont=dict(family=U.MONO_STACK, size=11, color=palette[i % 5]),
                                     customdata=g.get("test_pr_auc", pd.Series([np.nan] * len(g))),
                                     hovertemplate=f"<b>{m}</b> %{{x}}<br>Test ROC-AUC %{{y:.4f}}<br>Test PR-AUC %{{customdata:.4f}}<extra></extra>"))
        fig.update_layout(title="Tuning: default against tuned (test ROC-AUC)",
                          xaxis=dict(title="", showgrid=False, range=[-0.45, tuning.groupby("model").size().max() - 0.55]),
                          yaxis=dict(title="Test ROC-AUC"), legend=dict(y=1.04), margin=dict(l=8, r=8, t=90, b=40))
        U.chart(fig, height=420, key="lab_slope")
    else:
        st.info("No before and after tuning table in the metadata.")

# Why LightGBM: reasons computed from the metadata
U.section_title("Why LightGBM", "check")
table = U.records_table(meta.get("comparison_table"))
br = meta.get("business_results") or {}
tech, biz = [], []
stated = meta.get("model_choice") or meta.get("why_lightgbm") or meta.get("selection_rationale")
if table is not None and "test_roc_auc" in table.columns:
    tb = table.set_index(table.columns[0])
    lgbm = next((i for i in tb.index if "lightgbm" in str(i).lower()), None)
    if lgbm is not None:
        others = tb.drop(index=lgbm)
        rival = others["test_roc_auc"].idxmax()
        rank = int((tb["test_roc_auc"] > tb.loc[lgbm, "test_roc_auc"]).sum()) + 1
        tech.append(f"Test ROC-AUC {tb.loc[lgbm, 'test_roc_auc']:.3f}, rank {rank} of {len(tb)} models compared; the closest "
                    f"alternative, {rival}, scored {others.loc[rival, 'test_roc_auc']:.3f}.")
        if "test_brier" in tb.columns and tb.loc[lgbm, "test_brier"] < tb["test_brier"].drop(lgbm).min() - 0.0005:
            tech.append(f"Brier score {tb.loc[lgbm, 'test_brier']:.3f} against {tb['test_brier'].drop(lgbm).min():.3f} for the best "
                        "alternative, so its probabilities can be priced with the cost matrix.")
        if "cv_roc_auc_std" in tb.columns:
            tech.append(f"Stable across folds: cross-validated ROC-AUC standard deviation {tb.loc[lgbm, 'cv_roc_auc_std']:.4f}.")
        if "full_fit_seconds" in tb.columns and pd.notna(tb.loc[lgbm, "full_fit_seconds"]):
            slower = others[others["full_fit_seconds"] > tb.loc[lgbm, "full_fit_seconds"]]
            tech.append(f"Fits the full training split in {tb.loc[lgbm, 'full_fit_seconds']:.1f} seconds, faster than "
                        f"{len(slower)} of the {len(others)} other models, which keeps retraining cheap.")
path = U.artefact_path("model")
if path is not None:
    tech.append(f"The deployed file is {path.stat().st_size / 1e6:.1f} MB, well inside the 100 MB limit, and explains each "
                "score with exact TreeSHAP values.")
nv_m, nv_c, nv_a = (U.find_value(br, [k]) for k in ("net_value_model", "net_value_comparison_model", "net_value_approve_all"))
lr_m, lr_c = U.find_value(br, ["loss_reduction_pct"]), U.find_value(br, ["loss_reduction_pct_comparison_model"])
if nv_m is not None and nv_c is not None:
    biz.append(f"Net value on the held-out portfolio {U.fmt_usd(float(nv_m))} against {U.fmt_usd(float(nv_c))} for the "
               f"comparison model, a gain of {U.fmt_usd(float(nv_m) - float(nv_c))}.")
if nv_a is not None and nv_m is not None:
    biz.append(f"{U.fmt_usd(float(nv_m) - float(nv_a))} more than approving every loan ({U.fmt_usd(float(nv_a))}).")
if lr_m is not None:
    biz.append(f"Avoids {U.fmt_metric(lr_m, pct=True)} of charge-off loss"
               + (f" against {U.fmt_metric(lr_c, pct=True)} for the comparison model." if lr_c is not None else "."))
appr = U.find_value(br, ["approval_rate"])
if appr is not None:
    biz.append(f"Approves {U.fmt_metric(appr, pct=True)} of applications, above the "
               f"{U.cost_assumptions()['MIN_APPROVAL_RATE']:.0%} business minimum.")
if isinstance(stated, str):
    tech.insert(0, stated)
elif isinstance(stated, list):
    tech = [str(s) for s in stated] + tech
k1, k2 = st.columns(2, gap="large")
k1.markdown(f'<div class="cl-card"><h4>{U.icon("chart", 18)} Technical reasons</h4><ul class="cl-reading">'
            + "".join(f"<li>{s}</li>" for s in tech) + "</ul></div>", unsafe_allow_html=True)
k2.markdown(f'<div class="cl-card"><h4>{U.icon("coins", 18, U.COLOURS["amber"])} Business reasons</h4><ul class="cl-reading">'
            + "".join(f"<li>{s}</li>" for s in biz) + "</ul></div>", unsafe_allow_html=True)
st.caption(f"Curves computed in the app on the {source}; the comparison table, tuning and business results come from the "
           "model metadata.")
