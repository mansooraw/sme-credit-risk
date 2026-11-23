"""Portfolio and threshold: how the approval threshold changes risk and return on held-out loans."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import analytics as A  # noqa: E402
import experience as X  # noqa: E402
import utils as U  # noqa: E402

X.page_start("portfolio")
U.page_header(
    "Portfolio and threshold",
    "Move the approval threshold and see what would have happened to the held-out test loans: how many are "
    "approved, how many of those default, where the risk sits and what the bank earns or loses.",
    eyebrow="Risk appetite",
)

portfolio, source = U.scored_portfolio(U.model_version_key())
p = portfolio["p"].to_numpy()
y = portfolio["Default"].to_numpy().astype(int)
disb = portfolio["DisbursementGross"].to_numpy()
portion = portfolio["Portion"].to_numpy()
rec_t = U.threshold()
min_rate = U.cost_assumptions()["MIN_APPROVAL_RATE"]
st.caption(f"Data: {source}. Probabilities computed with the loaded model; outcomes are the recorded loan status.")


@st.cache_data(show_spinner=False)
def cached_curve(model_key: str) -> pd.DataFrame:
    return U.threshold_curve(p, y, disb, portion)


@st.cache_data(show_spinner=False)
def cached_objectives(model_key: str) -> dict:
    gain, loss = U.value_components(disb, portion)
    return A.objective_thresholds(p, y, gain, loss, min_rate)


objectives = cached_objectives(U.model_version_key())
OBJ = [("max_value", "Max net value", "Highest realised net value on the held-out loans, with no approval floor."),
       ("approval_floor", f"Best with {min_rate:.0%} approved", "Highest realised net value while approving at least "
        f"{min_rate:.0%} of loans: the rule used to set the model threshold."),
       ("recall_target", "Stop 90% of defaults", "The highest threshold (most approvals) that still stops at least 90% "
        "of the loans that defaulted.")]
with st.container(key="objectives"):
    U.html('<div class="cl-tile-label" style="margin-bottom:6px">Recommended by objective</div>')
    oc = st.columns([1, 1, 1, 1])
    current = U.shared_threshold()
    for col, (k, label, tip) in zip(oc, OBJ):
        on = abs(objectives[k] - current) < 1e-9
        if col.button(f"{label}: {objectives[k]:.3f}", key=f"obj_{k}", type="primary" if on else "secondary",
                      use_container_width=True, help=tip):
            U.set_shared_threshold(objectives[k])
            st.rerun()
    if oc[3].button(f"Model threshold: {rec_t:.3f}", key="obj_model", use_container_width=True,
                    type="primary" if abs(rec_t - current) < 1e-9 else "secondary",
                    help="The threshold stored in the model metadata."):
        U.set_shared_threshold(rec_t)
        st.rerun()

t = U.threshold_slider(key="thr_portfolio")
s = U.policy_summary(p, y, disb, portion, t)
base_default = float(y.mean())


def delta(text: str, good: bool) -> str:
    return f"<span class='{'cl-delta-up' if good else 'cl-delta-down'}'>{text}</span> vs approve-all"


nv_gain = s["realised_net_value"] - s["net_value_approve_all"]
dr_change = (s["realised_default_rate_approved"] - base_default) * 100
U.tiles([
    {"label": "Approval rate", "value": f"{s['approval_rate']:.1%}", "help": "Held-out loans with PD below the threshold, divided by all held-out loans.",
     "sub": delta(f"{(s['approval_rate'] - 1) * 100:+.1f} pts", s["approval_rate"] >= min_rate),
     "accent": U.COLOURS["teal"] if s["approval_rate"] >= min_rate else U.COLOURS["red"]},
    {"label": "Default rate, approved", "value": f"{s['realised_default_rate_approved']:.1%}",
     "sub": delta(f"{dr_change:+.1f} pts", dr_change <= 0), "accent": U.COLOURS["amber"],
     "help": f"Approved loans that were charged off, divided by approved loans. The average PD of approved loans is {s['expected_default_rate_approved']:.1%}."},
    {"label": "Realised net value", "value": U.fmt_usd(s["realised_net_value"]),
     "sub": delta(("+" if nv_gain >= 0 else "") + U.fmt_usd(nv_gain), nv_gain >= 0), "accent": U.COLOURS["navy"],
     "help": "Sum over approved loans of the margin if repaid, minus the unguaranteed loss and workout cost if charged off."},
    {"label": "Loss avoided", "value": U.fmt_usd(s["loss_avoided"]),
     "sub": f"{s['loss_reduction']:.1%} of charge-off loss", "accent": U.COLOURS["teal"],
     "help": "Charge-off loss if every loan were approved, minus the charge-off loss of the loans approved here."},
    {"label": "Recall on defaults", "value": f"{s['recall']:.1%}", "sub": f"Precision {s['precision']:.1%}",
     "accent": U.COLOURS["red"], "help": "Defaulted loans not approved, divided by all defaulted loans."},
], cols=5, min_width=150, compact=True)
if s["approval_rate"] < min_rate:
    st.warning(f"At this threshold the approval rate is below the business minimum of {min_rate:.0%}.")

c1, c2 = st.columns([1.35, 1.1], gap="large")
with c1:
    curve = cached_curve(U.model_version_key())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["realised_net_value"] / 1e6, name="Realised net value (USD m)",
                             line=dict(color=U.COLOURS["navy"], width=3.5), fill="tozeroy",
                             fillcolor=U.hex_rgba(U.COLOURS["navy"], 0.06),
                             hovertemplate="Threshold %{x:.2f}<br>Net value USD %{y:,.0f} m<extra></extra>"))
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["approval_rate"] * 100, name="Approval rate (%)", yaxis="y2",
                             line=dict(color=U.COLOURS["teal"], width=2.5, dash="dot"),
                             hovertemplate="Threshold %{x:.2f}<br>Approval %{y:.1f}%<extra></extra>"))
    fig.add_vline(x=rec_t, line_color=U.COLOURS["amber"], line_width=3,
                  annotation_text=f"recommended {rec_t:.3f}", annotation_position="top right",
                  annotation_font=dict(size=11, color="#7A5A12"))
    sel = curve.iloc[(curve["threshold"] - t).abs().argsort()[:1]]
    fig.add_trace(go.Scatter(x=[t, t], y=[min(0, float(curve["realised_net_value"].min() / 1e6)), float(curve["realised_net_value"].max() / 1e6) * 1.05],
                             mode="lines", line=dict(color=U.COLOURS["red"], dash="dash", width=2), name="Selected",
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[t], y=[s["realised_net_value"] / 1e6], mode="markers+text", name="Selected threshold",
                             marker=dict(size=15, color=U.COLOURS["red"], line=dict(color="#FFFFFF", width=3)),
                             text=[f"selected {t:.3f}"], textposition="bottom right", textfont=dict(color=U.COLOURS["red"], size=11),
                             hovertemplate=f"Selected {t:.3f}<br>Net value {U.fmt_usd(s['realised_net_value'])}"
                                           f"<br>Approval {s['approval_rate']:.1%}<extra></extra>", showlegend=False))
    fig.add_hline(y=min_rate * 100, yref="y2", line_color=U.COLOURS["grey"], line_dash="dot", line_width=1)
    ymin = float(curve["realised_net_value"].min() / 1e6)
    fig.update_layout(
        title="Net value and approval rate by threshold",
        xaxis=dict(title="Approval threshold"),
        yaxis=dict(title="Realised net value (USD m)", range=[min(0, ymin * 0.9), None]),
        yaxis2=dict(title="Approval rate (%)", overlaying="y", side="right", range=[0, 100], showgrid=False,
                    tickfont=dict(color=U.MUTED, size=12)),
        legend=dict(y=1.08), margin=dict(l=8, r=56, t=70, b=40),
        transition=dict(duration=600, easing="cubic-in-out"),
    )
    U.chart(fig, height=400, key="curve")
with c2:
    cm = s["confusion"]
    total = sum(cm.values()) or 1
    U.html(
        '<div class="cl-card"><h4>Confusion matrix at the selected threshold</h4>'
        '<div class="cl-cm" style="margin-top:12px">'
        '<div></div><div class="hd top">Approved</div><div class="hd top">Not approved</div>'
        f'<div class="hd">Repaid</div><div class="cell good"><div class="n">{cm["tn"]:,}</div><div class="t">{cm["tn"] / total:.1%} correctly approved</div></div>'
        f'<div class="cell warn"><div class="n">{cm["fp"]:,}</div><div class="t">{cm["fp"] / total:.1%} margin forgone</div></div>'
        f'<div class="hd">Defaulted</div><div class="cell bad"><div class="n">{cm["fn"]:,}</div><div class="t">{cm["fn"] / total:.1%} missed defaults</div></div>'
        f'<div class="cell good"><div class="n">{cm["tp"]:,}</div><div class="t">{cm["tp"] / total:.1%} defaults stopped</div></div>'
        "</div>"
        f'<p style="margin-top:14px">Rows are the recorded outcome, columns the model decision. Missed defaults are the costly error: '
        f"each loses part of the unguaranteed principal.</p></div>"
    )

approved = p < t
U.filter_chips(key="portfolio_clear")
flt = U.filters()
geo = pd.DataFrame({"State": portfolio["State"], "y": y, "approved": approved})
geo = geo[geo["approved"]].groupby("State").agg(rate=("y", "mean"), loans=("y", "size")).reset_index()
geo = geo[geo["State"].str.len() == 2]
c1, c2 = st.columns([1.25, 1], gap="large")
with c1:
    fig = go.Figure(go.Choropleth(
        locations=geo["State"], z=geo["rate"] * 100, locationmode="USA-states",
        colorscale=[[0, "#E3F4F1"], [0.35, U.COLOURS["teal"]], [0.7, U.COLOURS["amber"]], [1, U.COLOURS["red"]]],
        marker_line_color="#FFFFFF", marker_line_width=0.8,
        colorbar=dict(title=dict(text="%", font=dict(size=11)), thickness=10, len=0.7, tickfont=dict(size=11),
                      x=0.98, xanchor="right", ticksuffix=""),
        customdata=geo["loans"], hovertemplate="%{location}<br>Default rate %{z:.1f}%<br>%{customdata:,} approved loans<extra></extra>",
    ))
    fig.update_layout(title="Realised default rate of approved loans by state",
                      geo=dict(scope="usa", bgcolor="rgba(0,0,0,0)", lakecolor="#FFFFFF", showlakes=False),
                      margin=dict(l=0, r=10, t=44, b=0))
    fig.update_layout(clickmode="event+select")
    event = U.chart(fig, height=380, key=f"map_{U.filter_version()}", on_select="rerun")
    if U.apply_selection(event, "portfolio_map", "state", lookup=lambda pt: pt.get("location")):
        st.rerun()
with c2:
    _, el = U.expected_value(p, disb, portion)
    in_state = (portfolio["State"] == flt["state"]).to_numpy() if flt.get("state") else np.ones(len(p), bool)
    sec = pd.DataFrame({"Sector": portfolio["Sector"], "el": el, "p": p})[approved & in_state]
    sec = sec.groupby("Sector").agg(loss=("el", "sum"), pd_mean=("p", "mean"), loans=("p", "size")).reset_index()
    sec = sec[sec["loss"] > 0]
    from plotly.colors import sample_colorscale
    scale = [[0, "#CDEBE6"], [0.5, U.COLOURS["amber"]], [1, U.COLOURS["red"]]]
    span = float(sec["pd_mean"].max() - sec["pd_mean"].min()) or 1.0
    colours = sample_colorscale(scale, list((sec["pd_mean"] - sec["pd_mean"].min()) / span))
    root = "All sectors"
    fig = go.Figure(go.Treemap(
        labels=list(sec["Sector"]) + [root], parents=[root] * len(sec) + [""],
        values=list(sec["loss"]) + [float(sum(sec["loss"])) * 1.000001], branchvalues="total",
        marker=dict(colors=colours + ["#FFFFFF"], line=dict(color="#FFFFFF", width=2)),
        customdata=np.vstack([np.stack([sec["pd_mean"] * 100, sec["loans"]], axis=1),
                              [[float(np.average(sec["pd_mean"], weights=sec["loans"])) * 100, float(sec["loans"].sum())]]]),
        texttemplate="<b>%{label}</b><br>USD %{value:,.3s}", textfont=dict(size=12, color=U.INK),
        pathbar=dict(visible=False), tiling=dict(pad=2),
        hovertemplate="%{label}<br>Expected loss USD %{value:,.0f}<br>Mean PD %{customdata[0]:.1f}%<br>%{customdata[1]:,.0f} loans<extra></extra>",
    ))
    where = f" in {U.state_name(flt['state'])}" if flt.get("state") else ""
    fig.update_layout(title=f"Expected loss of approved loans by sector{where}", margin=dict(l=4, r=4, t=44, b=4),
                      uniformtext=dict(minsize=10, mode="hide"))
    U.chart(fig, height=380, key="treemap")
st.caption("Click a state on the map to see its sectors in the treemap. Box size is the expected loss of approved loans in "
           "each sector; colour is the mean PD. The threshold is shared with the Stress test, Monte Carlo and Model lab pages.")
