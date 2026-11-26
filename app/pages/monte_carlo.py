"""Monte Carlo. Each approved loan defaults at random with its PD. The title block and the figure strip; the
picture on the left (the distribution with its percentile lines, the fan chart by threshold, the realised
value) with its pill row; on the right the run card, the dark panel with the percentiles and the row that
opens the method sheet; the model threshold along the bottom."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import analytics as A  # noqa: E402
import ui  # noqa: E402
import utils as U  # noqa: E402

RUN_OPTIONS = [500, 1000, 2000]
GRID = np.round(np.arange(0.05, 0.6001, 0.025), 3)

model_key = U.model_version_key()
sample = U.sample_portfolio(model_key)
p = sample["p"].to_numpy()
y = sample["Default"].to_numpy().astype(int)
gain, loss = U.value_components(sample["DisbursementGross"], sample["Portion"])
rec_t = U.threshold()

ui.page("Monte Carlo", "The spread of net value when every approved loan defaults at random with its PD.",
        eyebrow="Portfolio", heading="One portfolio, a thousand futures.")


@st.cache_data(show_spinner=False)
def simulate(model_key: str, thresholds: tuple, runs: int, seed: int) -> np.ndarray:
    return A.monte_carlo(p, gain, loss, np.asarray(thresholds), runs=runs, seed=seed)


# ---------------------------------------------------------------------------
# The page: figure strip, picture left, run and percentiles right, model threshold strip
# ---------------------------------------------------------------------------

chosen = st.session_state.get("mc_pick", "p50")
if chosen not in ("p50", "p5", "loss", "realised"):
    chosen = "p50"
PICKS = [("p50", "Median"), ("p5", "5th percentile"), ("loss", "Fan by threshold"), ("realised", "Realised value")]

kpi_slot = st.container()

with st.container(key="two"):
    left, right = st.columns([1.2, 1], gap="medium")

    # The right column is filled first: its controls set the runs the figures and the picture use.
    with right:
        with st.container(key="mc_controls"):
            U.html(ui.card_html("Run", "Set the runs, the seed and the threshold.", ""))
            r1, r2 = st.columns(2)
            RUNS = int(r1.selectbox("Runs", RUN_OPTIONS, index=1, key="mc_runs", help="Number of simulated portfolio years."))
            SEED = int(r2.number_input("Seed", min_value=0, max_value=9999, value=42, step=1, key="mc_seed",
                                       help="Random seed; the same draws are used for every threshold."))
            t = U.threshold_slider("Approval threshold", key="thr_mc")

        fan = simulate(model_key, tuple(GRID), RUNS, SEED)
        cur = simulate(model_key, (round(t, 4), round(rec_t, 4)), RUNS, SEED)
        sim, sim_rec = cur[0], cur[1]
        s = A.distribution_summary(sim)
        s_rec = A.distribution_summary(sim_rec)
        approved = p < t
        realised = float(np.where(y == 1, -loss, gain)[approved].sum())
        pos = float((sim < realised).mean())
        n_loss = int(round(s["prob_loss"] * RUNS))

        pr = np.percentile(sim, [5, 25, 50, 75, 95])
        ui.card("Percentiles", f"Chance of a loss-making year at {t:.3f}: {s['prob_loss']:.1%}.", ui.rows(
            [{"a": lab, "b": ui.money(v)} for lab, v in zip(["P5", "P25", "P50", "P75", "P95"], pr)]
            + [{"a": "Mean", "b": ui.money(s["mean"])}]), kind="panel")

        with st.container(key="sheet_rows"):
            U.html(ui.card_html("How it is worked out", None, ""))
            if ui.row_button({"a": "Method and data", "b": f"Threshold {rec_t:.3f}"}, key="mc_method"):
                st.session_state["mc_sheet"] = True

    with left:
        if chosen == "loss":
            q5, q50, q95 = (np.percentile(fan, q, axis=1) / 1e6 for q in (5, 50, 95))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=GRID, y=q95, mode="lines", name="P95", line=dict(color=ui.CHART["accent_light"], width=1.5),
                                     hovertemplate="Threshold %{x:.3f}<br>P95 USD %{y:,.1f} m<extra></extra>"))
            fig.add_trace(go.Scatter(x=GRID, y=q5, mode="lines", name="P5", line=dict(color=ui.CHART["bad"], width=1.5),
                                     fill="tonexty", fillcolor=ui.CHART["accent_soft"],
                                     hovertemplate="Threshold %{x:.3f}<br>P5 USD %{y:,.1f} m<extra></extra>"))
            fig.add_trace(go.Scatter(x=GRID, y=q50, mode="lines", name="Median", line=dict(color=ui.CHART["accent"], width=2.5),
                                     customdata=(fan < 0).mean(axis=1) * 100,
                                     hovertemplate="Threshold %{x:.3f}<br>Median USD %{y:,.1f} m"
                                                   "<br>Loss-making in %{customdata:.1f}% of runs<extra></extra>"))
            fig.add_trace(go.Scatter(x=[t], y=[s["p50"] / 1e6], mode="markers", showlegend=False,
                                     marker=dict(size=11, color=ui.CHART["ink"], line=dict(color="#ffffff", width=2)),
                                     hovertemplate=f"Selected {t:.3f}<br>Median {ui.money(s['p50'])}"
                                                   f"<br>Loss-making in {s['prob_loss']:.1%} of runs<extra></extra>"))
            fig.add_hline(y=0, line_color=ui.CHART["muted"], line_width=1)
            ui.chart_layout(fig, "Net value by threshold: P5, median and P95 of the runs")
            fig.update_layout(xaxis=dict(title="Approval threshold (probability of default)"),
                              yaxis=dict(title="Portfolio net value (USD m)"))
        else:
            lo = float(min(sim.min(), sim_rec.min(), realised))
            hi = float(max(sim.max(), sim_rec.max(), realised))
            size = (hi - lo) / 45 or 1.0
            fig = go.Figure()
            if abs(t - rec_t) > 1e-9:
                fig.add_trace(go.Histogram(x=sim_rec / 1e6, name=f"Model threshold {rec_t:.3f}",
                                           xbins=dict(start=lo / 1e6, end=hi / 1e6, size=size / 1e6),
                                           marker=dict(color=ui.CHART["hairline"]), opacity=0.9,
                                           hovertemplate="USD %{x:,.2f} m<br>%{y} runs<extra>model threshold</extra>"))
            fig.add_trace(go.Histogram(x=sim / 1e6, name=f"Threshold {t:.3f}",
                                       xbins=dict(start=lo / 1e6, end=hi / 1e6, size=size / 1e6),
                                       marker=dict(color=ui.CHART["accent_light"] if chosen == "realised" else ui.CHART["accent"],
                                                   line=dict(color="#ffffff", width=0.5)), opacity=0.85,
                                       hovertemplate="USD %{x:,.2f} m<br>%{y} runs<extra>selected</extra>"))
            lines = [("p5", "P5", s["p5"]), ("p50", "P50", s["p50"]), ("p95", "P95", s["p95"]), ("realised", "realised", realised)]
            for key, label, value in lines:
                on = key == chosen
                colour = ui.CHART["ink"] if on else ui.CHART["muted"]
                fig.add_vline(x=value / 1e6, line_color=colour, line_width=3 if on else 1.2,
                              line_dash="solid" if on else "dot")
                fig.add_annotation(x=value / 1e6, y=1.0, yref="paper", text=label, showarrow=False, xanchor="left",
                                   xshift=4, yshift=-4 if key != "realised" else -22,
                                   font=dict(size=13 if on else 12, color=colour))
            titles = {"p50": f"Net value over {RUNS:,} runs, with the median",
                      "p5": f"Net value over {RUNS:,} runs, with the 5th percentile",
                      "realised": f"Realised value against {RUNS:,} simulated runs"}
            ui.chart_layout(fig, titles[chosen])
            fig.update_layout(barmode="overlay", bargap=0.04, xaxis=dict(title="Portfolio net value (USD m)"),
                              yaxis=dict(title="Runs"))
        with st.container(key="picture_fixed"):
            U.chart(fig, height=400, key="mc_picture")

        with st.container(key="pick_buttons"):
            cols = st.columns(len(PICKS))
            for col, (key, label) in zip(cols, PICKS):
                if col.button(label, key=f"btn_figures_{key}", type="primary" if key == chosen else "secondary",
                              use_container_width=True):
                    st.session_state["mc_pick"] = key
                    st.rerun()

with kpi_slot:
    ui.kpis([
        ("P50 net value", ui.money(s["p50"]), f"Mean {ui.money(s['mean'])}"),
        ("P5 net value", ui.money(s["p5"]), "1 year in 20 is worse"),
        ("Chance of loss", ui.pct(s["prob_loss"]), f"{n_loss:,} of {RUNS:,} runs"),
        ("Realised value", ui.money(realised), f"At the {pos:.0%} point"),
    ])

ui.strip([
    ui.strip_title("At the model threshold"),
    ui.strip_value(f"P50 net value at {rec_t:.3f}", ui.money(s_rec["p50"])),
    ui.strip_value("P5 net value", ui.money(s_rec["p5"])),
    ui.strip_value("Chance of a loss-making year", ui.pct(s_rec["prob_loss"])),
    f'<span class="l">{len(p):,} sample loans · {RUNS:,} runs · seed {SEED}</span>',
])


# ---------------------------------------------------------------------------
# The sheet: method
# ---------------------------------------------------------------------------

@st.dialog("How it is worked out")
def sheet() -> None:
    best = float(GRID[int(np.argmax(np.percentile(fan, 5, axis=1)))])
    bullets = [
        f"Data: the committed {len(p):,}-loan stratified sample of the held-out test split, scored with the loaded "
        f"model; {approved.sum():,} of them are approved at {t:.3f}.",
        "Each run draws every approved loan's default independently (Bernoulli with its PD); the seed fixes the "
        "draws, and the same draws are used for every threshold.",
        "Net value per run: the margin on repaid loans minus the unguaranteed loss and workout cost on defaulted ones.",
        f"Realised is the value that actually happened on these loans at the selected threshold: "
        f"{ui.money(realised)}, at the {pos:.0%} point of the runs. The best P5 on the grid is at {best:.3f}.",
        "Real defaults are correlated through the economy, so the true spread is wider: see the Stress test page.",
    ]
    U.html(ui.eyebrow("Method") + "".join(ui.note(b, "accent") for b in bullets))


if st.session_state.get("mc_sheet"):
    st.session_state.pop("mc_sheet")
    sheet()
