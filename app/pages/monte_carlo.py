"""Monte Carlo: the distribution of portfolio net value when each approved loan defaults at random with its PD."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import analytics as A  # noqa: E402
import experience as X  # noqa: E402
import utils as U  # noqa: E402

RUNS = 1000
SEED = 42
GRID = np.round(np.arange(0.05, 0.6001, 0.025), 3)

X.page_start("montecarlo")
U.page_header(
    "Monte Carlo",
    f"A point estimate hides the spread. Here every approved loan defaults at random with probability equal to its PD, "
    f"{RUNS:,} times over, to show the range of portfolio net value the bank should plan for at the chosen threshold.",
    eyebrow="Simulation",
)

sample = U.sample_portfolio(U.model_version_key())
p = sample["p"].to_numpy()
y = sample["Default"].to_numpy().astype(int)
gain, loss = U.value_components(sample["DisbursementGross"], sample["Portion"])
rec_t = U.threshold()


@st.cache_data(show_spinner=False)
def simulate(model_key: str, thresholds: tuple) -> np.ndarray:
    return A.monte_carlo(p, gain, loss, np.asarray(thresholds), runs=RUNS, seed=SEED)


t = U.threshold_slider(key="thr_mc")
slot = st.empty()
slot.markdown(U.skeleton(360, f"Simulating {RUNS:,} portfolio years"), unsafe_allow_html=True)
fan = simulate(U.model_version_key(), tuple(GRID))
cur = simulate(U.model_version_key(), (round(t, 4), round(rec_t, 4)))
slot.empty()
sim, sim_rec = cur[0], cur[1]
s = A.distribution_summary(sim)
approved = p < t
realised = float(np.where(y == 1, -loss, gain)[approved].sum())

U.tiles([
    {"label": "Median net value", "value": U.fmt_usd(s["p50"]), "accent": U.COLOURS["navy"],
     "sub": f"Mean {U.fmt_usd(s['mean'])}", "help": "The 50th percentile of simulated net value across the runs."},
    {"label": "5th percentile", "value": U.fmt_usd(s["p5"]), "accent": U.COLOURS["red"],
     "sub": "One year in twenty is worse", "help": "Net value exceeded in 95% of the simulated runs."},
    {"label": "95th percentile", "value": U.fmt_usd(s["p95"]), "accent": U.COLOURS["teal"],
     "sub": f"Spread {U.fmt_usd(s['p95'] - s['p5'])}", "help": "Net value exceeded in only 5% of the simulated runs."},
    {"label": "Chance of a loss-making year", "value": f"{s['prob_loss']:.1%}", "accent": U.COLOURS["amber"],
     "sub": f"{int(round(s['prob_loss'] * RUNS)):,} of {RUNS:,} runs below zero",
     "help": "Share of simulated runs in which the approved loans lose money in total."},
], cols=4)

c1, c2 = st.columns([1.25, 1], gap="large")
with c1:
    lo = float(min(sim.min(), sim_rec.min(), realised))
    hi = float(max(sim.max(), sim_rec.max(), realised))
    size = (hi - lo) / 45 or 1.0
    fig = go.Figure()
    if abs(t - rec_t) > 1e-9:
        fig.add_trace(go.Histogram(x=sim_rec / 1e6, name=f"Model threshold {rec_t:.3f}", xbins=dict(start=lo / 1e6, end=hi / 1e6, size=size / 1e6),
                                   marker=dict(color=U.hex_rgba(U.COLOURS["grey"], 0.45)),
                                   hovertemplate="USD %{x:,.2f} m<br>%{y} runs<extra>model threshold</extra>"))
    fig.add_trace(go.Histogram(x=sim / 1e6, name=f"Threshold {t:.3f}", xbins=dict(start=lo / 1e6, end=hi / 1e6, size=size / 1e6),
                               marker=dict(color=U.hex_rgba(U.COLOURS["teal"], 0.85), line=dict(color="#FFFFFF", width=0.5)),
                               hovertemplate="USD %{x:,.2f} m<br>%{y} runs<extra>selected</extra>"))
    for q, lab, col in [("p5", "P5", U.COLOURS["red"]), ("p50", "P50", U.COLOURS["navy"]), ("p95", "P95", U.COLOURS["teal"])]:
        fig.add_vline(x=s[q] / 1e6, line_color=col, line_width=2, line_dash="dash" if q != "p50" else "solid",
                      annotation_text=f"{lab} {U.fmt_usd(s[q])}", annotation_position="top",
                      annotation_font=dict(size=11, color=col))
    fig.add_vline(x=realised / 1e6, line_color=U.COLOURS["amber"], line_width=3,
                  annotation_text="realised", annotation_position="bottom right", annotation_font=dict(size=11, color="#7A5A12"))
    if lo < 0 < hi:
        fig.add_vrect(x0=lo / 1e6, x1=0, fillcolor=U.COLOURS["red"], opacity=0.06, line_width=0)
    fig.update_layout(title=f"Net value over {RUNS:,} simulated runs", barmode="overlay", bargap=0.04,
                      xaxis_title="Portfolio net value (USD m)", yaxis_title="Runs", margin=dict(l=8, r=8, t=90, b=40),
                      transition=dict(duration=500, easing="cubic-in-out"))
    U.chart(fig, height=420, key="mc_hist")
with c2:
    q5, q50, q95 = (np.percentile(fan, q, axis=1) / 1e6 for q in (5, 50, 95))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=GRID, y=q95, line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=GRID, y=q5, fill="tonexty", fillcolor=U.hex_rgba(U.COLOURS["teal"], 0.22), line=dict(width=0),
                             name="5th to 95th percentile", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=GRID, y=q50, mode="lines+markers", name="Median", line=dict(color=U.COLOURS["navy"], width=3),
                             marker=dict(size=6), customdata=np.stack([q5, q95, (fan < 0).mean(axis=1) * 100], axis=1),
                             hovertemplate="Threshold %{x:.3f}<br>Median USD %{y:,.1f} m<br>P5 USD %{customdata[0]:,.1f} m"
                                           "<br>P95 USD %{customdata[1]:,.1f} m<br>Loss-making %{customdata[2]:.1f}% of runs<extra></extra>"))
    fig.add_trace(go.Scatter(x=[t], y=[s["p50"] / 1e6], mode="markers", name="Selected",
                             marker=dict(size=15, color=U.COLOURS["red"], line=dict(color="#FFFFFF", width=3)),
                             hovertemplate=f"Selected {t:.3f}<br>Median {U.fmt_usd(s['p50'])}<extra></extra>"))
    fig.update_layout(title="How the distribution shifts with the threshold", xaxis_title="Approval threshold",
                      yaxis_title="Net value (USD m)", legend=dict(y=1.04), margin=dict(l=8, r=8, t=90, b=40),
                      transition=dict(duration=500, easing="cubic-in-out"))
    U.chart(fig, height=420, key="mc_fan")

best = float(GRID[int(np.argmax(np.percentile(fan, 5, axis=1)))])
pos = (sim < realised).mean()
U.html(f'<div class="cl-insight">{U.icon("info", 18)}<span>At a threshold of {t:.3f}, {approved.sum():,} of {len(p):,} sample loans '
       f'are approved. In 90% of runs net value lies between {U.fmt_usd(s["p5"])} and {U.fmt_usd(s["p95"])}; the value that '
       f'actually happened on these loans, {U.fmt_usd(realised)}, sits at the {pos:.0%} point of the simulated range. The '
       f'threshold with the best bad year (highest 5th percentile) on the grid is {best:.3f}.</span></div>')
st.caption(f"Data: the committed {len(p):,}-loan stratified sample of the held-out test split, scored with the loaded model. "
           f"Each run draws every loan's default independently (Bernoulli with its PD; seed {SEED}, the same draws for every "
           "threshold). Real defaults are correlated through the economy, so the true spread is wider: see the Stress test page.")
