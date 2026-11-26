"""Stress test. A downturn scenario is applied to the scored test sample. The title block and the figure
strip; the picture on the left (net value by threshold, loss by sector, the break-even buffers) with its pill
row; on the right the scenario card, the dark panel with the segments that break first and the row that opens
the method sheet; the base case along the bottom."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import analytics as A  # noqa: E402
import ui  # noqa: E402
import utils as U  # noqa: E402

c = U.cost_assumptions()
base_lgd, base_margin, workout = c["LOSS_GIVEN_DEFAULT"], c["NET_INTEREST_MARGIN"], c["WORKOUT_COST_RATE"]
model_key = U.model_version_key()
sample = U.sample_portfolio(model_key)
cols = U.deployment_features()

ui.page("Stress test", "Which segments break first in a downturn: set the scenario on the right.",
        eyebrow="Portfolio", heading="Push the book. Watch it bend.")


@st.cache_data(show_spinner=False)
def stressed_pd(model_key: str, recession: bool, shift_pts: int) -> np.ndarray:
    """PD of every sample loan with the recession flag and the guarantee shift applied."""
    X_ = sample[cols].copy()
    if recession:
        X_["Recession"] = 1
    if shift_pts:
        X_["Portion"] = (X_["Portion"] + shift_pts / 100).clip(0.05, 1.0)
    return U.load_model().predict_proba(X_)[:, 1]


def short_sector(name: str) -> str:
    """A sector name short enough for one row: the words before the first comma, ' and ' or bracket."""
    out = str(name)
    for cut in [",", " (", " and "]:
        out = out.split(cut)[0]
    return out


def signed_money(x: float) -> str:
    return ("+" if x >= 0 else "") + ui.money(x)


def kpis(app, p, ev, loss):
    el = p * loss
    return {"approval": float(app.mean()), "pd": float(p[app].mean()) if app.any() else 0.0,
            "value": float(ev[app].sum()), "el": float(el[app].sum()), "avoided": float(el[~app].sum())}


GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def curve(p, ev):
    order = np.argsort(p)
    cum = np.concatenate([[0], np.cumsum(ev[order])])
    return cum[np.searchsorted(p[order], GRID, side="left")]


# ---------------------------------------------------------------------------
# The page: figure strip, picture left, scenario and result right, base case strip
# ---------------------------------------------------------------------------

chosen = st.session_state.get("stress_pick", "value")
if chosen == "approval":
    chosen = "loss"
if chosen not in ("value", "default", "loss"):
    chosen = "value"
PICKS = [("value", "Net value by threshold"), ("default", "Loss by sector"), ("loss", "Break-even LGD")]

kpi_slot = st.container()

with st.container(key="two"):
    left, right = st.columns([1.2, 1], gap="medium")

    # The right column is filled first: its controls set the scenario the figures and the picture use.
    with right:
        with st.container(key="stress_controls"):
            U.html(ui.card_html("Scenario", "Set the downturn, then read the figures.", ""))
            recession = st.toggle("Recession on every loan", value=True, key="st_recession",
                                  help="Sets the 2007 to 2009 recession indicator to 1 for all loans and rescores them.")
            a1, a2 = st.columns(2)
            with a1:
                shift = st.slider("Guarantee shift", -10, 10, 0, 1, key="st_shift",
                                  help="Moves every loan's guaranteed share up or down by this many points, within 5% to 100%.")
            with a2:
                lgd = st.slider("LGD", 0.30, 0.80, float(base_lgd), 0.05, format="%.2f", key="st_lgd",
                                help=f"Loss given default: share of the unguaranteed principal lost on default; "
                                     f"base case {base_lgd:.0%}.")
            b1, b2 = st.columns(2)
            with b1:
                margin = st.slider("Margin", 0.03, 0.08, float(base_margin), 0.005, format="%.3f", key="st_margin",
                                   help=f"Net interest margin earned on a repaid loan, share of the disbursed amount; "
                                        f"base case {base_margin:.0%}.")
            with b2:
                t = U.threshold_slider("Threshold", key="st_thr")

        disb = sample["DisbursementGross"].to_numpy()
        portion0 = sample["Portion"].to_numpy()
        portion1 = np.clip(portion0 + shift / 100, 0.05, 1.0)
        p0 = sample["p"].to_numpy()
        p1 = stressed_pd(model_key, bool(recession), int(shift))
        g0, l0 = A.gain_loss(disb, portion0, base_margin, base_lgd, workout)
        g1, l1 = A.gain_loss(disb, portion1, margin, lgd, workout)
        app0, app1 = p0 < t, p1 < t
        ev0, ev1 = A.expected_net(p0, g0, l0), A.expected_net(p1, g1, l1)
        k0, k1 = kpis(app0, p0, ev0, l0), kpis(app1, p1, ev1, l1)

        # Break-even LGD by sector (sectors with at least 30 approved loans under the scenario)
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
        brk = pd.DataFrame(rows, columns=["Sector", "Approved loans", "Mean stressed PD", "Expected net value (USD)",
                                          "Break-even LGD"]).sort_values("Break-even LGD")
        losing = brk[brk["Expected net value (USD)"] < 0]

        def status(r) -> str:
            if r["Expected net value (USD)"] < 0:
                return "Loss"
            return "Caution" if r["Break-even LGD"] < lgd + 0.10 else "Pass"

        if brk.empty:
            line = "Under this scenario no sector has 30 approved loans."
            body = ui.empty("Raise the threshold to see the segments.")
        else:
            line = f"Under this scenario {len(losing)} of {len(brk)} sectors lose money."
            body = ui.rows([{"a": short_sector(r["Sector"]), "a_small": f"Break-even LGD {r['Break-even LGD']:.0%}",
                             "badge": status(r)} for _, r in brk.head(5).iterrows()])
        ui.card("Which breaks first", line, body, kind="panel")

        with st.container(key="sheet_rows"):
            U.html(ui.card_html("How it is worked out", None, ""))
            if ui.row_button({"a": "Method and data", "b": "All segments"}, key="st_method"):
                st.session_state["stress_sheet"] = True

    with left:
        if chosen == "value":
            v0, v1 = curve(p0, ev0), curve(p1, ev1)
            best1 = float(GRID[int(np.argmax(v1))])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=GRID, y=v0 / 1e6, name="Base case", line=dict(color=ui.CHART["accent_light"], width=2),
                                     hovertemplate="Threshold %{x:.2f}<br>Base case USD %{y:,.1f} m<extra></extra>"))
            fig.add_trace(go.Scatter(x=GRID, y=v1 / 1e6, name="Scenario", line=dict(color=ui.CHART["accent"], width=2.5),
                                     hovertemplate="Threshold %{x:.2f}<br>Scenario USD %{y:,.1f} m<extra></extra>"))
            fig.add_trace(go.Scatter(x=[best1], y=[v1.max() / 1e6], mode="markers", name="Best under the scenario",
                                     marker=dict(size=11, color=ui.CHART["ink"], line=dict(color="#ffffff", width=2)),
                                     hovertemplate=f"Best threshold under the scenario {best1:.2f}<br>USD %{{y:,.1f}} m<extra></extra>"))
            fig.add_vline(x=t, line_color=ui.CHART["ink"], line_dash="dot", line_width=1.5)
            fig.add_annotation(x=t, y=1.0, yref="paper", text=f"threshold {t:.3f}", showarrow=False, xanchor="left",
                               xshift=6, yshift=-6, font=dict(size=12, color=ui.CHART["muted"]))
            ui.chart_layout(fig, "Expected net value by threshold")
            fig.update_layout(xaxis=dict(title="Approval threshold (probability of default)", range=[0, 1]),
                              yaxis=dict(title="Expected net value (USD m)"))
        elif chosen == "default":
            seg = pd.DataFrame({"Sector": sample["Sector"], "el0": (p0 * l0) * app0, "el1": (p1 * l1) * app1})
            seg = seg.groupby("Sector")[["el0", "el1"]].sum()
            seg = seg[(seg["el0"] + seg["el1"]) > 0].sort_values("el1").tail(10)
            labels = [short_sector(s) for s in seg.index]
            fig = go.Figure()
            fig.add_trace(go.Bar(y=labels, x=seg["el0"] / 1e3, orientation="h", name="Base case",
                                 marker_color=ui.CHART["accent_light"], customdata=list(seg.index),
                                 hovertemplate="<b>%{customdata}</b><br>Base case expected loss USD %{x:,.0f} k<extra></extra>"))
            fig.add_trace(go.Bar(y=labels, x=seg["el1"] / 1e3, orientation="h", name="Scenario", marker_color=ui.CHART["bad"],
                                 customdata=np.stack([np.array(seg.index, dtype=object),
                                                      np.where(seg["el0"] > 0, seg["el1"] / seg["el0"] - 1, 0)], axis=1),
                                 hovertemplate="<b>%{customdata[0]}</b><br>Scenario expected loss USD %{x:,.0f} k"
                                               "<br>%{customdata[1]:+.0%} on the base case<extra></extra>"))
            ui.chart_layout(fig, "Expected loss of approved loans, top 10 sectors")
            fig.update_layout(barmode="group", bargap=0.25, xaxis=dict(title="Expected loss (USD thousand)"),
                              yaxis=dict(automargin=True))
        elif brk.empty:
            fig = None
            ui.card("Which segments break first", None,
                    ui.empty("No sector has 30 approved loans here: raise the threshold to see the buffers."))
        else:
            show = brk.iloc[::-1]
            shown = show["Break-even LGD"].clip(upper=2.0)
            colours = [ui.tone_colour(status(r)) if status(r) != "Pass" else ui.CHART["accent"] for _, r in show.iterrows()]
            fig = go.Figure(go.Bar(
                y=[short_sector(s) for s in show["Sector"]], x=shown * 100, orientation="h", marker_color=colours,
                customdata=np.stack([show["Sector"].to_numpy(dtype=object), show["Break-even LGD"] * 100,
                                     show["Approved loans"], show["Expected net value (USD)"]], axis=1),
                hovertemplate="<b>%{customdata[0]}</b><br>Break-even LGD %{customdata[1]:.0f}%"
                              "<br>%{customdata[2]:,} approved loans<br>Expected net value USD %{customdata[3]:,.0f}"
                              "<extra></extra>", showlegend=False))
            fig.add_vline(x=lgd * 100, line_color=ui.CHART["ink"], line_dash="dot", line_width=1.5)
            fig.add_annotation(x=lgd * 100, y=1.0, yref="paper", text=f"scenario LGD {lgd:.0%}", showarrow=False,
                               xanchor="left", xshift=6, yshift=-6, font=dict(size=12, color=ui.CHART["muted"]))
            ui.chart_layout(fig, "Break-even LGD by sector: the shortest bars break first")
            fig.update_layout(xaxis=dict(title="Break-even loss given default (%), shown up to 200%", ticksuffix="%"),
                              yaxis=dict(automargin=True), bargap=0.3)
        if fig is not None:
            with st.container(key="picture_fixed"):
                U.chart(fig, height=400, key="stress_picture")

        with st.container(key="pick_buttons"):
            cols = st.columns(len(PICKS))
            for col, (key, label) in zip(cols, PICKS):
                if col.button(label, key=f"btn_figures_{key}", type="primary" if key == chosen else "secondary",
                              use_container_width=True):
                    st.session_state["stress_pick"] = key
                    st.rerun()

with kpi_slot:
    ui.kpis([
        ("Net value", ui.money(k1["value"]), f"{signed_money(k1['value'] - k0['value'])} vs base"),
        ("Default rate", ui.pct(k1["pd"]), f"{round((k1['pd'] - k0['pd']) * 100, 2) + 0.0:+.2f} pts vs base"),
        ("Loss avoided", ui.money(k1["avoided"]), f"{signed_money(k1['avoided'] - k0['avoided'])} vs base"),
        ("Approval rate", ui.pct(k1["approval"]), f"{round((k1['approval'] - k0['approval']) * 100, 1) + 0.0:+.1f} pts vs base"),
    ])

ui.strip([
    ui.strip_title("Base case"),
    ui.strip_value("net value", ui.money(k0["value"])),
    ui.strip_value("default rate", ui.pct(k0["pd"])),
    ui.strip_value("loss avoided", ui.money(k0["avoided"])),
    ui.strip_value("approval rate", ui.pct(k0["approval"])),
    f'<span class="l">LGD {base_lgd:.0%} · margin {base_margin:.0%} · {len(sample):,} sample loans</span>',
])


# ---------------------------------------------------------------------------
# The sheet: method and the full break-even table
# ---------------------------------------------------------------------------

@st.dialog("How it is worked out")
def sheet() -> None:
    bullets = [
        f"Data: the committed {len(sample):,}-loan stratified sample of the held-out test split.",
        f"Base case: recorded recession flags and guarantees, LGD {base_lgd:.0%}, margin {base_margin:.0%}, "
        f"workout {workout:.0%}.",
        "Recession: the 2007 to 2009 indicator is set to 1 for every loan and the model rescores it. The guarantee "
        "shift moves each guaranteed share, which also changes the rescored PD.",
        "LGD and margin change only the pricing: the loss if charged off and the margin if repaid.",
        "Both sides use expected values from the model PD, because realised outcomes cannot change under a "
        "hypothetical scenario.",
        "Break-even LGD: the LGD at which a sector's expected net value is zero (sectors with at least 30 approved "
        "loans). Above 100% the sector stays profitable at any LGD. Loss: the sector loses money now; caution: it "
        "breaks within 10 points of the scenario LGD.",
    ]
    U.html(ui.eyebrow("Method") + "".join(ui.note(b, "accent") for b in bullets))
    if not brk.empty:
        U.html(ui.eyebrow("All segments") + ui.rows([
            {"a": r["Sector"], "a_small": f"{r['Approved loans']:,} approved, {ui.money(r['Expected net value (USD)'])}",
             "b": f"{r['Break-even LGD']:.0%}", "b_small": status(r)} for _, r in brk.iterrows()]))


if st.session_state.get("stress_sheet"):
    st.session_state.pop("stress_sheet")
    sheet()
