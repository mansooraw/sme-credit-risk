"""Model lab: the figure strip, the curves of the final and the comparison model on the same held-out loans on
the left (ROC, precision-recall or calibration; click the ROC curve to move the threshold) with its pill row, and
on the right the shared threshold, the two models at that threshold in the dark panel and the reasons for the
choice in rows that open sheets."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_curve  # noqa: E402

import ui  # noqa: E402
import utils as U  # noqa: E402

ui.page("Model lab", "Both models on the same held-out loans; click the ROC curve to move the threshold.", eyebrow="Model",
        heading="Two models, one test.")
meta = U.load_metadata()
scores, source = U.model_lab_scores(U.model_version_key())
y = scores["Default"].to_numpy()
final_name = str(meta.get("model_name", "Final model"))
comp_name = "Logistic regression" if U.load_comparison_model() is not None else None
MODELS = [("p", final_name, ui.CHART["accent"])]
if "p_comp" in scores:
    MODELS.append(("p_comp", comp_name or "Comparison model", ui.CHART["ink"]))
MAX_POINTS = 500
HEIGHT = 400
STEP = 0.005


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
t = float(np.clip(round(U.shared_threshold() / STEP) * STEP, 0.01, 0.99))  # the slider below shows the same value
pts = {k: point_at(k, t) for k, *_ in MODELS}
has_comp = "p_comp" in cv
chosen = st.session_state.get("lab_pick", "final")


def marker(k, x, yv, colour, extra=""):
    return go.Scatter(x=[x], y=[yv], mode="markers", showlegend=False,
                      marker=dict(size=11, color=colour, line=dict(color="#ffffff", width=2)),
                      hovertemplate=f"Threshold {t:.3f}<br>%{{x:.1%}}, %{{y:.1%}}{extra}<extra></extra>")


# ---------------------------------------------------------------------------
# The middle pictures
# ---------------------------------------------------------------------------

def picture_roc():
    focus = "p_comp" if chosen == "comparison" and has_comp else "p"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=ui.CHART["hairline"], dash="dash", width=1.5),
                             hoverinfo="skip"))
    for k, name, colour in MODELS:
        c = cv[k]
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines+markers", name=f"{name} ({c['auc']:.3f})",
                                 line=dict(color=colour, width=3 if k == focus else 1.8), marker=dict(size=4, opacity=0.0),
                                 customdata=c["thr"],
                                 hovertemplate="Threshold %{customdata:.3f}<br>Repaid loans declined %{x:.1%}"
                                               "<br>Defaults caught %{y:.1%}<br>Click to use this threshold<extra>" + name + "</extra>"))
    for k, name, colour in MODELS:
        fig.add_trace(marker(k, pts[k]["fpr"], pts[k]["tpr"], colour))
    ui.chart_layout(fig, "Defaults caught against repaid loans declined")
    fig.update_layout(xaxis=dict(title="Share of repaid loans declined (false positive rate)", tickformat=".0%", range=[0, 1]),
                      yaxis=dict(title="Share of defaults caught (recall)", tickformat=".0%", range=[0, 1.02]),
                      clickmode="event+select")
    with st.container(key="picture_fixed"):
        event = U.chart(fig, height=HEIGHT, key=f"lab_roc_{st.session_state.get('lab_v', 0)}", on_select="rerun")
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
                U.set_shared_threshold(float(np.clip(round(thr[idx] / STEP) * STEP, 0.01, 0.99)))
                st.rerun()


def picture_pr():
    fig = go.Figure()
    base = float(y.mean())
    fig.add_hline(y=base, line_dash="dot", line_color=ui.CHART["muted"], line_width=1.5, annotation_text=f"default rate {base:.1%}",
                  annotation_position="bottom left", annotation_font=dict(size=12, color=ui.CHART["muted"]))
    for k, name, colour in MODELS:
        c = cv[k]
        fig.add_trace(go.Scatter(x=c["rec"], y=c["prec"], mode="lines", name=f"{name} ({c['pr_auc']:.3f})",
                                 line=dict(color=colour, width=2.5), customdata=np.append(c["thr_pr"], np.nan)[: len(c["rec"])],
                                 hovertemplate="Threshold %{customdata:.3f}<br>Recall %{x:.1%}<br>Precision %{y:.1%}<extra></extra>"))
        fig.add_trace(marker(k, pts[k]["tpr"], pts[k]["precision"], colour, f"<br>Approval rate {pts[k]['approval']:.1%}"))
    ui.chart_layout(fig, "Precision against recall")
    fig.update_layout(xaxis=dict(title="Share of defaults caught (recall)", tickformat=".0%", range=[0, 1]),
                      yaxis=dict(title="Share of flagged loans that default (precision)", tickformat=".0%", range=[0, 1.02]))
    ui.picture(fig, height=HEIGHT)


def picture_calibration():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color=ui.CHART["hairline"], dash="dash", width=1.5), hoverinfo="skip"))
    for k, name, colour in MODELS:
        c = cv[k]
        fig.add_trace(go.Scatter(x=c["cal_x"], y=c["cal_y"], mode="lines+markers", name=f"{name} (Brier {c['brier']:.3f})",
                                 line=dict(color=colour, width=2.5), marker=dict(size=7, line=dict(color="#ffffff", width=1.5)),
                                 hovertemplate="Mean predicted PD %{x:.1%}<br>Observed default rate %{y:.1%}<extra></extra>"))
    ui.chart_layout(fig, "Calibration by PD decile")
    fig.update_layout(xaxis=dict(title="Mean predicted probability of default", tickformat=".0%", range=[0, 1]),
                      yaxis=dict(title="Observed default rate", tickformat=".0%", range=[0, 1]))
    ui.picture(fig, height=HEIGHT)


PICTURES = {"final": picture_roc, "comparison": picture_roc, "recall": picture_pr, "precision": picture_calibration}

# ---------------------------------------------------------------------------
# Why LightGBM: reasons computed from the metadata, shown in the sheet
# ---------------------------------------------------------------------------

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
    biz.append(f"Net value on the held-out portfolio {ui.money(float(nv_m))} against {ui.money(float(nv_c))} for the "
               f"comparison model, a gain of {ui.money(float(nv_m) - float(nv_c))}.")
if nv_a is not None and nv_m is not None:
    biz.append(f"{ui.money(float(nv_m) - float(nv_a))} more than approving every loan ({ui.money(float(nv_a))}).")
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
tuning = U.records_table(meta.get("before_after_tuning"))
tuned = None
if tuning is not None and {"model", "test_roc_auc"} <= set(tuning.columns):
    g = tuning[tuning["model"].astype(str).str.lower().str.contains("lightgbm")]
    if len(g) >= 2:
        tuned = f"Tuning moved the LightGBM test ROC-AUC from {g['test_roc_auc'].iloc[0]:.3f} to {g['test_roc_auc'].iloc[-1]:.3f}."

# ---------------------------------------------------------------------------
# The page: figure strip, curves left, threshold and results right, data strip
# ---------------------------------------------------------------------------

fin, comp = cv["p"], cv.get("p_comp")
ui.kpis([
    ("Final model AUC", f"{fin['auc']:.3f}", f"PR-AUC {fin['pr_auc']:.3f}"),
    ("Comparison AUC", "n/a" if comp is None else f"{comp['auc']:.3f}", "Not loaded" if comp is None else f"PR-AUC {comp['pr_auc']:.3f}"),
    ("Recall", ui.pct(pts["p"]["tpr"]), f"At threshold {t:.3f}"),
    ("Precision", ui.pct(pts["p"]["precision"]), f"At threshold {t:.3f}"),
])


def trained(key: str) -> str:
    p = U.artefact_path(key)
    if key == "model" and meta.get("trained_on"):
        return ui.day(str(meta["trained_on"]))
    return ui.day(dt.date.fromtimestamp(p.stat().st_mtime)) if p is not None else "Not recorded"


def short_name(name: str) -> str:
    return name if len(name) < 28 else name.split("(")[0].strip()


with st.container(key="two"):
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        PICTURES[chosen]()
        with st.container(key="pick_buttons"):
            cols = st.columns(4)
            for col, key, label in zip(cols, ("final", "comparison", "recall", "precision"),
                                       ("Final ROC", "Comparison ROC", "Precision-recall", "Calibration")):
                if col.button(label, key=f"btn_figures_{key}", type="primary" if key == chosen else "secondary",
                              use_container_width=True, disabled=key == "comparison" and not has_comp):
                    st.session_state["lab_pick"] = key
                    st.rerun()

    with right:
        with st.container(key="lab_controls"):
            U.html(ui.card_html("Threshold", "Loans at or above it are flagged; or click the ROC curve.", ""))
            t = U.threshold_slider("Threshold (probability of default)", key="thr_lab")

        ui.card("At this threshold", f"Threshold {t:.3f}, both models on the same loans.", ui.rows([
            {"a": short_name(name), "a_small": f"Approves {pts[k]['approval']:.1%}",
             "b": f"{pts[k]['tpr']:.1%} caught", "b_small": f"Precision {pts[k]['precision']:.1%}"} for k, name, _ in MODELS]),
            kind="panel")

        with st.container(key="sheet_rows"):
            U.html(ui.card_html("Why LightGBM", "Open a row for the reasons and the method.", ""))
            if ui.row_button({"a": "Technical reasons", "a_small": "Accuracy, calibration, speed"}, key="lab_tech"):
                st.session_state["lab_sheet"] = "tech"
            if ui.row_button({"a": "Business reasons", "a_small": "Net value and loss avoided"}, key="lab_biz"):
                st.session_state["lab_sheet"] = "biz"
            if ui.row_button({"a": "How it is worked out", "a_small": "Curves, threshold and data"}, key="lab_method"):
                st.session_state["lab_sheet"] = "method"

strip = [ui.strip_title("Models"), ui.strip_value(f"{short_name(final_name)} trained", trained("model"))]
if has_comp:
    strip.append(ui.strip_value(f"{comp_name or 'Comparison model'} saved", trained("comparison")))
strip.append(f'<span class="l">Curves on the {source}</span>')
ui.strip(strip)


# ---------------------------------------------------------------------------
# The sheet
# ---------------------------------------------------------------------------

@st.dialog("How it is measured")
def sheet(which: str) -> None:
    if which == "tech":
        U.html(ui.eyebrow("Why LightGBM: technical reasons"))
        lines = tech or ["Not available in the metadata."]
    elif which == "biz":
        U.html(ui.eyebrow("Why LightGBM: business reasons"))
        lines = biz or ["Not available in the metadata."]
    else:
        U.html(ui.eyebrow("How it is worked out"))
        lines = [f"Curves computed in the app on the {source}; the marker shows the threshold on each curve.",
                 "Loans with a probability of default at or above the threshold are flagged; recall is the share of "
                 "defaults flagged and precision the share of flagged loans that default.",
                 "Calibration groups the loans into ten equal-sized PD bands and compares the mean predicted PD with the "
                 "observed default rate.",
                 "Comparison table, tuning and business results come from the model metadata."]
        if tuned:
            lines.insert(2, tuned)
    for line in lines:
        U.html(ui.note(line, "accent"))


which = st.session_state.pop("lab_sheet", None)
if which:
    sheet(which)
