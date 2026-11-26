"""Model insights: the figure strip of test metrics, the picture on the left (global importance, a dependence
plot, errors by sector, temporal validation) with its pill row, and on the right the dependence input, the top
drivers in the dark panel and the rows that open the weakest segments, the tuning table and the method."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
import utils as U  # noqa: E402

ui.page("Model insights", "What the model relies on, how each input moves the score and where it errs.", eyebrow="Model",
        heading="What the model relies on.")
meta = U.load_metadata()
HEIGHT = 400
SHAP_ROWS = 1500
ROWS = 3          # rows per right-hand list that fit at 1138 x 690
SEGMENT_MIN = 200  # sectors with fewer held-out loans are left out of the error analysis
DEP_FEATURES = ["TermYears", "Portion", "LogDisbursement", "NoEmp", "LoanPerEmployee", "DisbursedToApprovedRatio",
                "RetainedJob", "CreateJob", "JobsPerDollar"]

metrics = meta.get("metrics_test") or {}
tv = meta.get("temporal_validation") if isinstance(meta.get("temporal_validation"), dict) else {}
t = U.threshold()


def value(v, pct=False):
    return "n/a" if v is None else (ui.pct(float(v)) if pct else f"{float(v):.3f}")


@st.cache_data(show_spinner="Computing SHAP values on a sample of held-out loans")
def shap_sample(model_key: str, n: int = SHAP_ROWS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP values (log-odds, summed to input features) for a fixed sample of the committed test sample."""
    df = pd.read_csv(U.TEST_SAMPLE_PATH)
    df = U._clean_text(df.sample(min(n, len(df)), random_state=7).reset_index(drop=True))
    model = U.load_model()
    pre, _ = U.split_pipeline(model)
    X_ = df[U.deployment_features()]
    Xt = pre.transform(X_) if pre is not None else X_.to_numpy()
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = U.warm_explainer().shap_values(Xt)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    out_names = U.output_feature_names(model)[: values.shape[1]]
    sv = pd.DataFrame(values[:, : len(out_names)], columns=out_names)
    sv = sv.T.groupby([U.source_feature(c) for c in sv.columns]).sum().T
    return df, sv


@st.cache_data(show_spinner=False)
def errors(model_key: str, threshold: float) -> tuple[pd.DataFrame, str, float]:
    portfolio, source = U.scored_portfolio(model_key)
    pf = portfolio.assign(flag=portfolio["p"] >= threshold, y=portfolio["Default"].astype(int))
    pf["missed"] = (~pf["flag"]) & (pf["y"] == 1)
    pf["declined_good"] = pf["flag"] & (pf["y"] == 0)
    err = pf.groupby("Sector").agg(loans=("y", "size"), default_rate=("y", "mean"), mean_pd=("p", "mean"),
                                   missed=("missed", "sum"), defaults=("y", "sum"),
                                   declined_good=("declined_good", "sum")).reset_index()
    err["missed_share"] = err["missed"] / err["defaults"].clip(lower=1)
    err["declined_share"] = err["declined_good"] / (err["loans"] - err["defaults"]).clip(lower=1)
    overall_missed = float(pf["missed"].sum() / max(int(pf["y"].sum()), 1))
    return err[err["loans"] >= SEGMENT_MIN].sort_values("missed_share", ascending=False), source, overall_missed


def short_sector(s: str) -> str:
    s = str(s)
    if len(s) >= 28:
        s = s.split(",")[0]
    if len(s) >= 28:
        s = s.split(" and ")[0]
    return s


key = U.model_version_key()
importance = U.global_importance(12)
err, source, overall_missed = errors(key, t)

# ---------------------------------------------------------------------------
# The middle pictures
# ---------------------------------------------------------------------------


def picture_importance():
    if importance is None:
        U.html(ui.empty("Global importance is not available for the loaded model: run notebook 05 to write shap_global.csv."))
        return
    g = importance.iloc[::-1]
    fig = go.Figure(go.Bar(x=g["importance"], y=[U.short_label(s) for s in g["source"]], orientation="h",
                           marker_color=ui.CHART["accent"], text=[f"{v:.2f}" for v in g["importance"]], textposition="outside",
                           cliponaxis=False, textfont=dict(size=12, color=ui.CHART["muted"]), customdata=g["label"],
                           hovertemplate="%{customdata}<br>Mean |SHAP| %{x:.3f}<extra></extra>"))
    ui.chart_layout(fig, "What the model relies on most")
    fig.update_layout(showlegend=False, bargap=0.3, yaxis=dict(showgrid=False),
                      xaxis=dict(title="Mean absolute SHAP value (log-odds)", range=[0, float(g["importance"].max()) * 1.15]))
    ui.picture(fig, height=HEIGHT)


def picture_dependence(feat):
    sample, sv = shap_sample(key)
    x = sample[feat].to_numpy(dtype=float)
    log_x = feat == "LogDisbursement"
    if log_x:
        x = np.expm1(x)
    yv = sample["Default"].to_numpy().astype(int)
    fig = go.Figure()
    fig.add_hline(y=0, line_color=ui.CHART["hairline"], line_width=1)
    for outcome, colour, label in [(0, ui.CHART["accent"], "Repaid"), (1, ui.CHART["bad"], "Defaulted")]:
        m = yv == outcome
        fig.add_trace(go.Scattergl(x=x[m], y=sv.loc[m, feat], mode="markers", name=label,
                                   marker=dict(size=5, color=colour, opacity=0.45),
                                   hovertemplate=f"{U.short_label(feat)} %{{x:,.2f}}<br>SHAP %{{y:.2f}}<extra>{label}</extra>"))
    ui.chart_layout(fig, f"How {U.short_label(feat).lower()} moves the score")
    fig.update_layout(xaxis=dict(title=U.short_label(feat) + (" (USD, log scale)" if log_x else ""), type="log" if log_x else None),
                      yaxis=dict(title="SHAP value (log-odds); above 0 raises the PD"))
    ui.picture(fig, height=HEIGHT)


def picture_errors():
    e = err.sort_values("loans", ascending=False).head(12).sort_values("missed_share")
    labels = [short_sector(s) for s in e["Sector"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=e["missed_share"], y=labels, orientation="h", name="Defaults approved", marker_color=ui.CHART["bad"],
                         customdata=e[["loans", "default_rate"]].to_numpy(),
                         hovertemplate="%{y}<br>Defaults approved %{x:.1%}<br>%{customdata[0]:,.0f} loans, default rate "
                                       "%{customdata[1]:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(x=e["declined_share"], y=labels, orientation="h", name="Repaid loans not approved",
                         marker_color=ui.CHART["accent_light"],
                         hovertemplate="%{y}<br>Repaid loans not approved %{x:.1%}<extra></extra>"))
    ui.chart_layout(fig, f"Errors by sector at threshold {t:.0%}")
    fig.update_layout(barmode="group", bargap=0.25, yaxis=dict(showgrid=False),
                      xaxis=dict(title="Share of the sector's defaults or repaid loans", tickformat=".0%"))
    ui.picture(fig, height=HEIGHT)


def picture_temporal():
    keys = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("recall", "Recall"), ("precision", "Precision")]
    keys = [(k, label) for k, label in keys if k in tv]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[label for _, label in keys], y=[metrics.get(k) for k, _ in keys], name="Random split",
                         marker_color=ui.CHART["accent_light"], text=[value(metrics.get(k)) for k, _ in keys],
                         textposition="outside", textfont=dict(size=12, color=ui.CHART["muted"]),
                         hovertemplate="%{x}<br>Random split %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=[label for _, label in keys], y=[tv.get(k) for k, _ in keys],
                         name=f"Temporal split (tested on {str(tv.get('test', 'later years')).replace('ApprovalFY', 'approvals')})",
                         marker_color=ui.CHART["accent"], text=[value(tv.get(k)) for k, _ in keys],
                         textposition="outside", textfont=dict(size=12, color=ui.CHART["muted"]),
                         hovertemplate="%{x}<br>Temporal split %{y:.3f}<extra></extra>"))
    ui.chart_layout(fig, "Trained on older loans, tested on later ones")
    fig.update_layout(barmode="group", bargap=0.3, xaxis=dict(showgrid=False), yaxis=dict(title="Score on the test loans", range=[0, 1.1]),
                      legend=dict(y=1.0, yanchor="bottom"))
    ui.picture(fig, height=HEIGHT)


# ---------------------------------------------------------------------------
# The page: figure strip, picture left, input and top drivers right, data strip
# ---------------------------------------------------------------------------

drop = tv.get("roc_auc_drop_vs_random_split")
ui.kpis([
    ("ROC-AUC", value(metrics.get("roc_auc")), "Test split"),
    ("PR-AUC", value(metrics.get("pr_auc")), "Test split"),
    ("Recall", value(metrics.get("recall"), pct=True), f"At threshold {t:.0%}"),
    ("Temporal AUC", value(tv.get("roc_auc")), "Not recorded" if drop is None else f"Drop {float(drop):.3f}"),
])
chosen = st.session_state.get("insights_pick", "roc")
if chosen not in ("roc", "pr", "recall", "temporal"):
    chosen = "roc"
feat = st.session_state.get("ins_dep_feature", DEP_FEATURES[0])

with st.container(key="two"):
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        if chosen == "roc":
            picture_importance()
        elif chosen == "pr":
            try:
                sample, sv = shap_sample(key)
                ok = feat in sample.columns and feat in sv.columns
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                picture_dependence(feat)
            else:
                U.html(ui.empty("SHAP dependence is not available for the loaded model: run notebook 05 to write the model files."))
        elif chosen == "recall":
            picture_errors()
        elif tv:
            picture_temporal()
        else:
            U.html(ui.empty("No temporal validation results in the metadata: run notebook 05 to add them."))
        with st.container(key="pick_buttons"):
            cols = st.columns(4)
            for col, k, label in zip(cols, ("roc", "pr", "recall", "temporal"),
                                     ("Importance", "Dependence", "Errors by sector", "Over time")):
                if col.button(label, key=f"btn_figures_{k}", type="primary" if k == chosen else "secondary",
                              use_container_width=True):
                    st.session_state["insights_pick"] = k
                    st.rerun()

    with right:
        with st.container(key="insights_controls"):
            U.html(ui.card_html("Input", "The input the dependence picture plots.", ""))
            if st.session_state.get("ins_dep_feature") not in DEP_FEATURES:
                st.session_state["ins_dep_feature"] = DEP_FEATURES[0]

            def show_dependence():
                st.session_state["insights_pick"] = "pr"

            st.selectbox("Input", DEP_FEATURES, key="ins_dep_feature", format_func=U.short_label, on_change=show_dependence)

        if importance is not None:
            ui.card("Top drivers", "Mean absolute SHAP value, log-odds.", ui.rows([
                {"a": U.short_label(r["source"]), "b": f"{r['importance']:.2f}"} for _, r in importance.head(ROWS).iterrows()]),
                kind="panel")
        else:
            ui.card("Top drivers", None, ui.empty("Global importance is not available: run notebook 05 to write shap_global.csv."),
                    kind="panel")

        tuning = U.records_table(meta.get("before_after_tuning"))
        has_tuning = tuning is not None and {"model", "test_roc_auc"} <= set(tuning.columns)
        with st.container(key="sheet_rows"):
            U.html(ui.card_html("Where it errs", "Open a row for the segments, the tuning or the method.", ""))
            worst = err.head(1)
            if len(worst):
                w = worst.iloc[0]
                if ui.row_button({"a": "Weakest segments", "a_small": short_sector(w["Sector"]),
                                  "badge": "High" if w["missed_share"] > overall_missed * 1.5 else "Medium",
                                  "b_small": ui.pct(w["missed_share"])}, key="ins_segments"):
                    st.session_state["insights_sheet"] = "segments"
            if has_tuning and ui.row_button({"a": "See all", "a_small": f"Tuning, {len(tuning)} model versions"}, key="ins_tuning"):
                st.session_state["insights_sheet"] = "tuning"
            if ui.row_button({"a": "How it is worked out", "a_small": "SHAP, errors and temporal split"}, key="ins_how"):
                st.session_state["insights_sheet"] = "how"

strip = [ui.strip_title("Tuning")]
if has_tuning:
    g = tuning[tuning["model"].astype(str).str.lower().str.contains("lightgbm")]
    if len(g) >= 2:
        strip += [ui.strip_value("ROC-AUC before", f"{g['test_roc_auc'].iloc[0]:.3f}"),
                  ui.strip_value("after tuning", f"{g['test_roc_auc'].iloc[-1]:.3f}")]
strip.append(f'<span class="l">Errors on the {source}</span>')
ui.strip(strip)


# ---------------------------------------------------------------------------
# The sheets: weakest segments, the full tuning table, the method
# ---------------------------------------------------------------------------

@st.dialog("Weakest segments", width="large")
def sheet_segments() -> None:
    U.html(ui.note(f"Share of each sector's defaults the model approves at threshold {t:.0%}; all sectors "
                   f"{overall_missed:.1%}. Sectors with at least {SEGMENT_MIN} held-out loans.", "accent"))
    U.html(ui.rows([
        {"a": short_sector(r["Sector"]), "a_small": f"{int(r['loans']):,} loans, default rate {r['default_rate']:.1%}",
         "badge": "High" if r["missed_share"] > overall_missed * 1.5 else ("Medium" if r["missed_share"] > overall_missed else "Low"),
         "b_small": ui.pct(r["missed_share"])} for _, r in err.head(8).iterrows()]))


@st.dialog("Tuning", width="large")
def sheet_tuning() -> None:
    table = U.records_table(meta.get("before_after_tuning"))
    U.html(ui.note("Each model before and after tuning, scored on the held-out test split. From the model metadata.", "accent"))
    st.dataframe(table, hide_index=True, use_container_width=True, key="ins_tuning_table")


@st.dialog("How it is worked out")
def sheet_how() -> None:
    for line in [
        "ROC-AUC, PR-AUC and recall are read from the model metadata (held-out test split).",
        "Importance is the mean absolute SHAP value of each input (shap_global.csv); one-hot levels are summed.",
        f"Dependence plots TreeExplainer SHAP values for {SHAP_ROWS:,} loans of the committed test sample.",
        f"Errors by sector are worked out in the app on the {source} at the business threshold.",
        f"The temporal split retrains on {tv.get('train', 'older approvals')} and tests on {tv.get('test', 'later approvals')}; "
        "recent years are right-censored, as loans approved shortly before the data ends had little time to default.",
    ]:
        U.html(ui.note(line, "accent"))


which = st.session_state.pop("insights_sheet", None)
if which == "segments":
    sheet_segments()
elif which == "tuning":
    sheet_tuning()
elif which == "how":
    sheet_how()
