"""Stress test: apply a macro scenario to the scored test sample and see which segments break first."""

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

X.page_start("stress")
U.page_header(
    "Stress test",
    "Put the held-out loans through a downturn: switch the recession indicator on for every loan, raise the loss "
    "given default, squeeze the margin or move the guarantee. The model rescores each loan and the cost matrix "
    "prices the result.",
    eyebrow="Scenario analysis",
)

c = U.cost_assumptions()
base_lgd, base_margin, workout = c["LOSS_GIVEN_DEFAULT"], c["NET_INTEREST_MARGIN"], c["WORKOUT_COST_RATE"]
sample = U.sample_portfolio(U.model_version_key())
cols = U.deployment_features()


@st.cache_data(show_spinner=False)
def stressed_pd(model_key: str, recession: bool, shift_pts: int) -> np.ndarray:
    """PD of every sample loan with the recession flag and the guarantee shift applied."""
    X_ = sample[cols].copy()
    if recession:
        X_["Recession"] = 1
    if shift_pts:
        X_["Portion"] = (X_["Portion"] + shift_pts / 100).clip(0.05, 1.0)
    return U.load_model().predict_proba(X_)[:, 1]


with st.container(border=True):
    s1, s2 = st.columns(2, gap="large")
    recession = s1.toggle("Recession on every loan", value=True, key="st_recession",
                          help="Sets the 2007 to 2009 recession indicator to 1 for all loans and rescores them.")
    with s2:
        t = U.threshold_slider("Approval threshold", key="st_thr")
    s1, s2, s3 = st.columns(3, gap="large")
    lgd = s1.slider("Loss given default", 0.30, 0.80, float(base_lgd), 0.05, format="%.2f", key="st_lgd",
                    help=f"Share of the unguaranteed principal lost on default; baseline {base_lgd:.0%}.")
    margin = s2.slider("Net interest margin", 0.03, 0.08, float(base_margin), 0.005, format="%.3f", key="st_margin",
                       help=f"Earned on a repaid loan, share of the disbursed amount; baseline {base_margin:.0%}.")
    shift = s3.slider("Guarantee share shift (points)", -10, 10, 0, 1, key="st_shift",
                      help="Moves every loan's guaranteed share up or down, within 5% to 100%.")

slot = st.empty()
slot.markdown(U.skeleton(140, "Rescoring 20,000 loans under the scenario"), unsafe_allow_html=True)

disb = sample["DisbursementGross"].to_numpy()
portion0 = sample["Portion"].to_numpy()
portion1 = np.clip(portion0 + shift / 100, 0.05, 1.0)
p0 = sample["p"].to_numpy()
p1 = stressed_pd(U.model_version_key(), bool(recession), int(shift))
g0, l0 = A.gain_loss(disb, portion0, base_margin, base_lgd, workout)
g1, l1 = A.gain_loss(disb, portion1, margin, lgd, workout)
app0, app1 = p0 < t, p1 < t
ev0, ev1 = A.expected_net(p0, g0, l0), A.expected_net(p1, g1, l1)
slot.empty()


def kpis(app, p, ev, loss):
    return {"approval": float(app.mean()), "pd": float(p[app].mean()) if app.any() else 0.0,
            "value": float(ev[app].sum()), "el": float((p * loss)[app].sum())}


k0, k1 = kpis(app0, p0, ev0, l0), kpis(app1, p1, ev1, l1)


def dsub(now, before, fmt, good_up):
    diff = now - before
    good = diff >= 0 if good_up else diff <= 0
    return f"<span class='{'cl-delta-up' if good else 'cl-delta-down'}'>{fmt(diff)}</span> vs baseline"


U.tiles([
    {"label": "Approval rate", "value": f"{k1['approval']:.1%}", "accent": U.COLOURS["teal"],
     "sub": dsub(k1["approval"], k0["approval"], lambda v: f"{v * 100:+.1f} pts", True),
     "count": {"to": round(k1["approval"] * 100, 1), "decimals": 1, "suffix": "%"},
     "help": "Share of sample loans whose stressed PD is below the threshold."},
    {"label": "Mean PD, approved", "value": f"{k1['pd']:.1%}", "accent": U.COLOURS["amber"],
     "sub": dsub(k1["pd"], k0["pd"], lambda v: f"{v * 100:+.2f} pts", False),
     "count": {"to": round(k1["pd"] * 100, 2), "decimals": 2, "suffix": "%"},
     "help": "Average stressed probability of default of the approved loans."},
    {"label": "Expected net value", "value": U.fmt_usd(k1["value"]), "accent": U.COLOURS["navy"],
     "sub": dsub(k1["value"], k0["value"], lambda v: ("+" if v >= 0 else "") + U.fmt_usd(v), True),
     "count": {"to": round(k1["value"] / 1e6, 1), "decimals": 1, "prefix": "USD ", "suffix": " m"},
     "help": "Sum over approved loans of (1 - PD) x margin x disbursed minus PD x (LGD x unguaranteed share + workout) x disbursed."},
    {"label": "Expected loss", "value": U.fmt_usd(k1["el"]), "accent": U.COLOURS["red"],
     "sub": dsub(k1["el"], k0["el"], lambda v: ("+" if v >= 0 else "") + U.fmt_usd(v), False),
     "count": {"to": round(k1["el"] / 1e6, 1), "decimals": 1, "prefix": "USD ", "suffix": " m"},
     "help": "Sum over approved loans of PD x loss if charged off, with the scenario's LGD and guarantee."},
])

c1, c2 = st.columns([1.2, 1], gap="large")
with c1:
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)

    def curve(p, ev):
        order = np.argsort(p)
        cum = np.concatenate([[0], np.cumsum(ev[order])])
        return cum[np.searchsorted(p[order], grid, side="left")]

    v0, v1 = curve(p0, ev0), curve(p1, ev1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=v0 / 1e6, name="Baseline", line=dict(color=U.COLOURS["navy"], width=3),
                             hovertemplate="Threshold %{x:.2f}<br>Baseline USD %{y:,.1f} m<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=v1 / 1e6, name="Stressed", line=dict(color=U.COLOURS["red"], width=3),
                             fill="tonexty", fillcolor=U.hex_rgba(U.COLOURS["red"], 0.10),
                             hovertemplate="Threshold %{x:.2f}<br>Stressed USD %{y:,.1f} m<extra></extra>"))
    fig.add_vline(x=t, line_dash="dot", line_color=U.INK, annotation_text=f"threshold {t:.3f}",
                  annotation_font=dict(size=11, color=U.INK))
    best1 = float(grid[int(np.argmax(v1))])
    fig.add_trace(go.Scatter(x=[best1], y=[v1.max() / 1e6], mode="markers", name="Best under stress",
                             marker=dict(size=12, color=U.COLOURS["amber"], line=dict(color="#FFFFFF", width=2)),
                             hovertemplate=f"Best threshold under stress {best1:.2f}<br>USD %{{y:,.1f}} m<extra></extra>"))
    fig.update_layout(title="Expected net value by threshold", xaxis_title="Approval threshold",
                      yaxis_title="Expected net value (USD m)", transition=dict(duration=500, easing="cubic-in-out"))
    U.chart(fig, height=380, key="st_curve")
with c2:
    seg = pd.DataFrame({"Sector": sample["Sector"], "el0": (p0 * l0) * app0, "el1": (p1 * l1) * app1})
    seg = seg.groupby("Sector")[["el0", "el1"]].sum()
    seg = seg[(seg["el0"] + seg["el1"]) > 0].sort_values("el1").tail(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=seg.index, x=seg["el0"] / 1e3, orientation="h", name="Baseline", marker_color=U.hex_rgba(U.COLOURS["navy"], 0.55),
                         hovertemplate="<b>%{y}</b><br>Baseline expected loss USD %{x:,.0f} k<extra></extra>"))
    fig.add_trace(go.Bar(y=seg.index, x=seg["el1"] / 1e3, orientation="h", name="Stressed", marker_color=U.COLOURS["red"],
                         customdata=np.where(seg["el0"] > 0, seg["el1"] / seg["el0"] - 1, 0),
                         hovertemplate="<b>%{y}</b><br>Stressed expected loss USD %{x:,.0f} k<br>%{customdata:+.0%} on baseline<extra></extra>"))
    fig.update_layout(title="Expected loss of approved loans by sector", barmode="group", barcornerradius=4,
                      xaxis=dict(title="USD thousand", nticks=4), yaxis=dict(automargin=True, tickfont=dict(size=11)), bargap=0.25,
                      margin=dict(l=8, r=8, t=78, b=40))
    U.chart(fig, height=380, key="st_sectors")

# Which segments break first: break-even LGD by sector under the scenario PD and margin
U.section_title("Which segments break first", "shield")
rows = []
for sector, idx in sample.groupby("Sector").groups.items():
    m = np.zeros(len(sample), bool)
    m[sample.index.get_indexer(idx)] = True
    m &= app1
    if m.sum() < 30:
        continue
    be = A.break_even_lgd(p1[m], disb[m], portion1[m], margin, workout)
    rows.append({"Sector": sector, "Approved loans": int(m.sum()), "Mean stressed PD": float(p1[m].mean()),
                 "Expected net value (USD)": round(float(ev1[m].sum())), "Break-even LGD": be})
brk = pd.DataFrame(rows).sort_values("Break-even LGD")
losing = brk[brk["Expected net value (USD)"] < 0]
first = brk.head(3)
names = ", ".join(f"{r['Sector']} ({r['Break-even LGD']:.0%})" for _, r in first.iterrows())
summary = (f"Under this scenario, {len(losing)} of {len(brk)} sectors with at least 30 approved sample loans lose money "
           f"in expectation at a {t:.1%} threshold. ")
if len(first) and first.iloc[0]["Break-even LGD"] <= 1:
    summary += (f"As loss given default rises, the first to break are {names}: above those LGD levels their expected "
                "losses exceed the margin they earn.")
else:
    summary += (f"Even losing all of the unguaranteed principal would not wipe out the margin of any sector, because the "
                f"approved loans have low PDs; the thinnest buffers are in {names} (break-even LGD above 100% means the "
                "sector stays profitable at any LGD). Lower the guarantee or raise the threshold to find the breaking point.")
if len(losing):
    summary += f" {losing.iloc[0]['Sector']} is already loss-making at an LGD of {lgd:.0%}."
U.html(f'<div class="cl-insight">{U.icon("info", 18)}<span>{summary}</span></div>')
st.dataframe(brk, hide_index=True, use_container_width=True, column_config={
    "Mean stressed PD": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=float(max(brk["Mean stressed PD"].max(), 0.01))),
    "Expected net value (USD)": st.column_config.NumberColumn(format="localized"),
    "Break-even LGD": st.column_config.NumberColumn(format="percent", help="LGD at which the sector's expected net value is zero."),
})
st.caption(f"Data: the committed {len(sample):,}-loan stratified sample of the held-out test split. Baseline: recorded "
           f"recession flags and guarantees, LGD {base_lgd:.0%}, margin {base_margin:.0%}, workout {workout:.0%}. Both "
           "sides use expected values from the model's PD, because realised outcomes cannot change under a hypothetical scenario.")
