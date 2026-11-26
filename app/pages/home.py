"""Overview: the board. Four figures on the left choose what the middle picture shows; the right column
holds the example applications (each opens a sheet), the cost assumptions and the data note."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import experience as X  # noqa: E402
import ui  # noqa: E402
import utils as U  # noqa: E402

meta = U.load_metadata()
t, d = U.threshold(), U.decline_threshold()
c = U.cost_assumptions()
metrics = meta.get("metrics_test", {}) or {}
policy = U.model_policy_results(meta)
approval = U.find_value(policy, ["approval_rate"])
loss_red = U.find_value(meta.get("business_results", {}), ["loss_reduction_pct", "loss_reduction", "loss_reduction_percent"])
n_test = meta.get("n_test")
schema = U.load_schema()
mix = X.decision_mix()
model_key = U.model_version_key()


def share(v):
    """Metadata shares arrive either as fractions or as percentages."""
    if v is None:
        return None
    v = float(v)
    return v / 100 if abs(v) > 1.5 else v


roc = metrics.get("roc_auc")
recall = share(metrics.get("recall"))
approval = share(approval)
loss_red = share(loss_red)

ui.page("Overview", "A clear view of lending performance and decision policy.", eyebrow="Overview", heading="Credit risk, in focus.")


# ---------------------------------------------------------------------------
# Curves for the middle picture (computed once per model version)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def curves(key: str) -> dict:
    from sklearn.metrics import roc_curve
    portfolio, source = U.scored_portfolio(key)
    p = portfolio["p"].to_numpy()
    y = portfolio["Default"].to_numpy().astype(int)
    disb = portfolio["DisbursementGross"].to_numpy()
    portion = portfolio["Portion"].to_numpy()
    fpr, tpr, thr = roc_curve(y, p)
    keep = np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int)
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    rows = []
    for g in grid:
        s = U.policy_summary(p, y, disb, portion, float(g))
        rows.append({"threshold": g, "approval_rate": s["approval_rate"], "recall": s["recall"],
                     "net_value": s["realised_net_value"], "loss_reduction": s["loss_reduction"],
                     "default_rate_approved": s["realised_default_rate_approved"]})
    at_t = U.policy_summary(p, y, disb, portion, float(t))
    return {"roc": pd.DataFrame({"fpr": fpr[keep], "tpr": tpr[keep]}), "grid": pd.DataFrame(rows),
            "at_t": {k: v for k, v in at_t.items() if k != "confusion"}, "source": source, "n": int(len(p))}


C = curves(model_key)
grid = C["grid"]


def mark_threshold(fig, ymax=None):
    fig.add_vline(x=t, line_color=ui.CHART["ink"], line_dash="dot", line_width=1.5)
    fig.add_annotation(x=t, y=1.0, yref="paper", text=f"model threshold {t:.0%}", showarrow=False, xanchor="left",
                       xshift=6, yshift=-6, font=dict(size=12, color=ui.CHART["muted"]))


def picture_roc():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.CHART["hairline"], width=1.5, dash="dash"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=C["roc"]["fpr"], y=C["roc"]["tpr"], mode="lines", name="Final model",
                             line=dict(color=ui.CHART["accent"], width=2.5),
                             hovertemplate="Repaid loans declined %{x:.1%}<br>Defaults caught %{y:.1%}<extra></extra>"))
    a = C["at_t"]
    fig.add_trace(go.Scatter(x=[_fpr_at_t()], y=[a["recall"]], mode="markers+text",
                             marker=dict(size=11, color=ui.CHART["ink"], line=dict(color="#ffffff", width=2)),
                             text=[f"threshold {t:.0%}"], textposition="bottom right", textfont=dict(size=12, color=ui.CHART["muted"]),
                             hovertemplate=f"At the model threshold<br>Defaults caught {a['recall']:.1%}<extra></extra>", showlegend=False))
    ui.chart_layout(fig, "Defaults caught against repaid loans declined")
    fig.update_layout(xaxis=dict(title="Share of repaid loans declined", tickformat=".0%", range=[0, 1]),
                      yaxis=dict(title="Share of defaults caught", tickformat=".0%", range=[0, 1.02]), showlegend=False)
    return fig


def _fpr_at_t() -> float:
    portfolio, _ = U.scored_portfolio(model_key)
    p = portfolio["p"].to_numpy()
    y = portfolio["Default"].to_numpy().astype(int)
    repaid = y == 0
    return float(((p >= t) & repaid).sum() / max(repaid.sum(), 1))


def picture_curve(ycol: str, title: str, ylabel: str, fmt: str, floor: float | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid["threshold"], y=grid[ycol], mode="lines", line=dict(color=ui.CHART["accent"], width=2.5),
                             hovertemplate="Threshold %{x:.2f}<br>" + ylabel + " %{y:" + fmt + "}<extra></extra>", showlegend=False))
    if floor is not None:
        fig.add_hline(y=floor, line_color=ui.CHART["warning"], line_dash="dot", line_width=1.5,
                      annotation_text=f"business minimum {floor:.0%}", annotation_position="bottom right",
                      annotation_font=dict(size=12, color=ui.CHART["warning"]))
    sel = grid.iloc[(grid["threshold"] - t).abs().argsort()[:1]]
    fig.add_trace(go.Scatter(x=sel["threshold"], y=sel[ycol], mode="markers", marker=dict(size=11, color=ui.CHART["ink"], line=dict(color="#ffffff", width=2)),
                             hoverinfo="skip", showlegend=False))
    mark_threshold(fig)
    ui.chart_layout(fig, title)
    fig.update_layout(xaxis=dict(title="Approval threshold (probability of default)", range=[0, 1]),
                      yaxis=dict(title=ylabel, tickformat=fmt))
    return fig


def picture_value():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid["threshold"], y=grid["net_value"] / 1e6, mode="lines", name="Net value, USD m",
                             line=dict(color=ui.CHART["accent"], width=2.5),
                             hovertemplate="Threshold %{x:.2f}<br>Net value USD %{y:,.1f} m<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid["threshold"], y=grid["loss_reduction"] * 100, mode="lines", name="Charge-off loss avoided, %",
                             yaxis="y2", line=dict(color=ui.CHART["accent_light"], width=2, dash="dot"),
                             hovertemplate="Threshold %{x:.2f}<br>Loss avoided %{y:.0f}%<extra></extra>"))
    mark_threshold(fig)
    ui.chart_layout(fig, "Net value and loss avoided by threshold")
    fig.update_layout(xaxis=dict(title="Approval threshold (probability of default)", range=[0, 1]),
                      yaxis=dict(title="Realised net value (USD m)"),
                      yaxis2=dict(title="Loss avoided (%)", overlaying="y", side="right", range=[0, 100], showgrid=False,
                                  tickfont=dict(color=ui.CHART["muted"], size=12), title_font=dict(color=ui.CHART["muted"], size=12)),
                      margin=dict(r=56))
    return fig


PICTURES = {
    "roc": picture_roc,
    "recall": lambda: picture_curve("recall", "Defaults caught by threshold", "Share of defaults caught", ".0%"),
    "approval": lambda: picture_curve("approval_rate", "Approval rate by threshold", "Approval rate", ".0%", floor=c["MIN_APPROVAL_RATE"]),
    "loss": picture_value,
}

# ---------------------------------------------------------------------------
# The page: figure strip, picture left, policy and examples right, cost strip
# ---------------------------------------------------------------------------

chosen = st.session_state.get("home_pick", "roc")
ui.kpis([
    ("ROC-AUC", "n/a" if roc is None else f"{float(roc):.3f}", None),
    ("Defaults caught", "n/a" if recall is None else f"{recall:.1%}", None),
    ("Approval rate", "n/a" if approval is None else f"{approval:.1%}", None),
    ("Loss avoided", "n/a" if loss_red is None else f"{loss_red:.1%}", None),
])

with st.container(key="two"):
    left, right = st.columns([1.18, 1], gap="medium")
    with left:
        with st.container(key="picture_fixed"):
            fig = PICTURES[chosen]()
            U.chart(fig, height=344, key="home_picture")
        with st.container(key="pick_buttons"):
            c1, c2, c3, c4 = st.columns(4)
            for col, key, label in zip((c1, c2, c3, c4), ("roc", "recall", "approval", "loss"),
                                       ("ROC curve", "Defaults caught", "Approval rate", "Net value")):
                if col.button(label, key=f"btn_figures_{key}", type="primary" if key == chosen else "secondary",
                              use_container_width=True):
                    st.session_state["home_pick"] = key
                    st.rerun()
    with right:
        ui.card("Decision policy", None, ui.rows([
            {"a": "Approved", "a_small": f"PD below {t:.0%}", "b": f"{mix['approve']:.1%}"},
            {"a": "Referred", "a_small": f"PD {t:.0%} to {d:.0%}", "b": f"{mix['refer']:.1%}"},
            {"a": "Declined", "a_small": f"PD {d:.0%} and above", "b": f"{mix['decline']:.1%}"},
        ]) + ui.stack([("good", mix["approve"], f"{mix['approve']:.1%}"), ("warning", mix["refer"], f"{mix['refer']:.1%}"),
                      ("bad", mix["decline"], f"{mix['decline']:.1%}")]), kind="panel")
        try:
            ex = U.score_examples(model_key)
        except Exception:  # noqa: BLE001
            ex = None
        with st.container(key="sheet_rows"):
            U.html(ui.card_html("Example applications", "Open one to see its score and what drives it.", ""))
            if ex is None:
                U.html(ui.empty("The example applications are not available: run notebook 05 to write the model files."))
            else:
                for key, r in ex.iterrows():
                    dec = {"Approve": "Approved", "Refer": "Referred", "Decline": "Declined"}[r["decision"]]
                    pressed = ui.row_button({"a": U.EXAMPLE_LABELS.get(key, key), "a_small": f"{ui.money(r['DisbursementGross'])} · {ui.months(r['Term'])}",
                                             "badge": dec, "b_small": f"PD {r['p']:.1%}"}, key=f"ex_{key}")
                    if pressed:
                        st.session_state["home_sheet"] = key

n_train, rate = schema.get("n_training_loans"), schema.get("training_default_rate")
ui.strip([
    ui.strip_title("Cost assumptions"),
    f'<span class="ui-eyebrow" style="margin:0">If repaid</span><span class="v">+{c["NET_INTEREST_MARGIN"]:.0%}</span><span class="l">of disbursed</span>',
    f'<span class="ui-eyebrow" style="margin:0">If it defaults</span><span class="v">-{c["LOSS_GIVEN_DEFAULT"]:.0%}</span><span class="l">of unguaranteed</span>',
    f'<span class="l">SBA 7(a) loans · 1962 to 2014' + (f" · trained on {n_train:,} loans" if n_train else "") + "</span>",
])


# ---------------------------------------------------------------------------
# The sheet: one example application, scored and explained
# ---------------------------------------------------------------------------

@st.dialog("Application")
def sheet(key: str) -> None:
    ex = U.score_examples(model_key)
    r = ex.loc[key]
    raw = dict(U.EXAMPLES[key])
    rows = pd.DataFrame([{**raw, "Recession": int(raw["Recession"])}], index=[key])
    feats, p = U.score_form_rows(rows)
    p = float(p[0])
    decision = U.decisions(np.array([p]))[0]
    dec = {"Approve": "Approved", "Refer": "Referred", "Decline": "Declined"}[decision]
    gain, loss = U.value_components(np.array([raw["DisbursementGross"]]), np.array([feats["Portion"].iloc[0]]))
    ev = (1 - p) * float(gain[0]) - p * float(loss[0])
    U.html(ui.pair([("Probability of default", f"{p:.1%}", U.band_for(p)["name"]), ("Expected net value", ui.money(ev), "to the bank")]))
    U.html(f'<div style="margin:8px 0 12px">{ui.badge(dec)}</div>')
    try:
        contrib, _ = U.local_contributions(U.load_model(), feats.iloc[[0]])
        drivers = U.explanation_sentences(contrib, 3)
    except Exception:  # noqa: BLE001
        drivers = []
    if drivers:
        U.html(ui.eyebrow("What drives the score") + "".join(ui.note(s, "accent") for s in drivers))
    facts = [("Sector", raw["Sector"]), ("State", U.state_name(raw["State"])), ("Disbursed", ui.money(raw["DisbursementGross"])),
             ("Term", ui.months(raw["Term"])), ("Guaranteed", f"{feats['Portion'].iloc[0]:.0%}"), ("Employees", f"{raw['NoEmp']}"),
             ("Business", "New" if raw["NewExist"] == "New" else "Existing"), ("Lender", str(raw["Bank"]).title())]
    U.html(ui.eyebrow("Application") + ui.rows([{"a": k, "b": v} for k, v in facts]))
    if st.button("Open in Score a loan", type="primary", use_container_width=True, key="sheet_open"):
        st.session_state["example"] = key
        st.session_state["auto_score"] = True
        st.session_state.pop("home_sheet", None)
        U.switch_page("pages/score_loan.py")


if st.session_state.get("home_sheet"):
    key = st.session_state.pop("home_sheet")
    sheet(key)
