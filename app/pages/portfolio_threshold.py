"""Portfolio and threshold. The title block and the figure strip; the picture on the left (the value and
approval curves, the default map by state, the loss treemap by sector) with its pill row of choices; on the
right the threshold card with the objectives, the dark panel with the confusion matrix, and the rows that
open the filter and the method sheet; the base case along the bottom."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.colors import sample_colorscale  # noqa: E402

import analytics as A  # noqa: E402
import ui  # noqa: E402
import utils as U  # noqa: E402

model_key = U.model_version_key()
portfolio, source = U.scored_portfolio(model_key)
p = portfolio["p"].to_numpy()
y = portfolio["Default"].to_numpy().astype(int)
disb = portfolio["DisbursementGross"].to_numpy()
portion = portfolio["Portion"].to_numpy()
rec_t = U.threshold()
min_rate = U.cost_assumptions()["MIN_APPROVAL_RATE"]


@st.cache_data(show_spinner=False)
def cached_curve(model_key: str) -> pd.DataFrame:
    return U.threshold_curve(p, y, disb, portion)


@st.cache_data(show_spinner=False)
def cached_objectives(model_key: str) -> dict:
    gain, loss = U.value_components(disb, portion)
    return A.objective_thresholds(p, y, gain, loss, min_rate)


objectives = cached_objectives(model_key)
curve = cached_curve(model_key)

ui.page("Portfolio and threshold", "Set the threshold, then choose a picture to see where risk and return move.",
        eyebrow="Portfolio", heading="Where risk and return meet.")


def signed_money(x: float) -> str:
    return ("+" if x >= 0 else "") + ui.money(x)


# ---------------------------------------------------------------------------
# The page: figure strip, picture left, threshold and result right, base case strip
# ---------------------------------------------------------------------------

chosen = st.session_state.get("portfolio_pick", "approval")
if chosen not in ("approval", "default", "value", "loss"):
    chosen = "approval"
OBJ = [("max_value", "Max net value", "Highest realised net value on the held-out loans, with no approval floor."),
       ("approval_floor", f"Best with {min_rate:.0%} approved", "Highest realised net value while approving at least "
        f"{min_rate:.0%} of loans: the rule used to set the model threshold."),
       ("recall_target", "Stop 90% of defaults", "The highest threshold (most approvals) that still stops at least 90% "
        "of the loans that defaulted.")]
PICKS = [("approval", "Approval and value"), ("default", "Default map"), ("value", "Net value"), ("loss", "Loss by sector")]

kpi_slot = st.container()

with st.container(key="two"):
    left, right = st.columns([1.2, 1], gap="medium")

    # The right column is filled first: its slider sets the threshold the figures and the picture use.
    with right:
        with st.container(key="portfolio_controls"):
            U.html(ui.card_html("Threshold", "Drag it, or pick an objective below.", ""))
            t = U.threshold_slider("Approval threshold", key="thr_portfolio")
            s = U.policy_summary(p, y, disb, portion, t)
            if s["approval_rate"] < min_rate:
                U.html(ui.note(f"Below the business minimum of {min_rate:.0%} approved.", "warning"))
            current = U.shared_threshold()
            targets = [(k, label, objectives[k], tip) for k, label, tip in OBJ]
            targets.append(("model", "Model threshold", rec_t, "The threshold stored in the model metadata."))
            for k, label, value, tip in targets:
                with st.container(key=f"row_obj_{k}"):
                    U.html(ui.rows([{"key": k, "a": label, "b": f"{value:.3f}"}],
                                   chosen=k if abs(value - current) < 1e-9 else None))
                    if st.button(f"Set {label}", key=f"obj_{k}", help=tip, use_container_width=True):
                        U.set_shared_threshold(value)
                        st.rerun()

        cm = s["confusion"]
        total = sum(cm.values()) or 1
        ui.card("Confusion matrix", f"Held-out loans at the threshold {t:.3f}.", ui.rows([
            {"a": "Correctly approved", "a_small": "Repaid and approved", "b": f"{cm['tn']:,} · {cm['tn'] / total:.1%}"},
            {"a": "Defaults stopped", "a_small": "Defaulted and declined", "b": f"{cm['tp']:,} · {cm['tp'] / total:.1%}"},
            {"a": "Missed defaults", "a_small": "Defaulted and approved", "b": f"{cm['fn']:,} · {cm['fn'] / total:.1%}"},
            {"a": "Margin forgone", "a_small": "Repaid and declined", "b": f"{cm['fp']:,} · {cm['fp'] / total:.1%}"},
        ]) + ui.pair([("Recall", ui.pct(s["recall"]), None), ("Precision", ui.pct(s["precision"]), None)]), kind="panel")

        flt = U.filters()
        names = {"state": U.state_name, "sector": str, "bank": lambda v: str(v).title(),
                 "pd_range": lambda v: f"PD {v[0]:.0%} to {v[1]:.0%}"}
        with st.container(key="sheet_rows"):
            U.html(ui.card_html("Filter and method", "Click a state on the map to filter.", ""))
            if not flt:
                U.html(ui.rows([{"a": "State", "b": "All states"}]))
            else:
                for k, v in list(flt.items()):
                    if ui.row_button({"a": k.replace("_", " ").capitalize(), "b": names.get(k, str)(v)}, key=f"flt_{k}"):
                        U.set_filter(k, None)
                        st.session_state["filter_version"] = U.filter_version() + 1
                        st.rerun()
                if ui.row_button({"a": "Clear", "b": "All loans"}, key="portfolio_clear"):
                    U.clear_filters()
                    st.rerun()
            if ui.row_button({"a": "Method and data", "b": "Objectives"}, key="pf_method"):
                st.session_state["portfolio_sheet"] = True

    approved = p < t

    with left:
        if chosen in ("approval", "value"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["realised_net_value"] / 1e6, name="Net value, USD m",
                                     line=dict(color=ui.CHART["accent"], width=2.5), showlegend=chosen == "approval",
                                     hovertemplate="Threshold %{x:.2f}<br>Net value USD %{y:,.1f} m<extra></extra>"))
            if chosen == "approval":
                fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["approval_rate"] * 100, name="Approval rate, %",
                                         yaxis="y2", line=dict(color=ui.CHART["accent_light"], width=2, dash="dot"),
                                         hovertemplate="Threshold %{x:.2f}<br>Approval %{y:.1f}%<extra></extra>"))
                fig.add_hline(y=min_rate * 100, yref="y2", line_color=ui.CHART["warning"], line_dash="dot", line_width=1.5,
                              annotation_text=f"business minimum {min_rate:.0%}", annotation_position="bottom right",
                              annotation_font=dict(size=12, color=ui.CHART["warning"]))
                title = "Net value and approval rate by threshold"
            else:
                fig.add_hline(y=s["net_value_approve_all"] / 1e6, line_color=ui.CHART["muted"], line_dash="dot",
                              line_width=1.5, annotation_text="approve every loan", annotation_position="bottom right",
                              annotation_font=dict(size=12, color=ui.CHART["muted"]))
                title = "Realised net value by threshold"
            fig.add_vline(x=rec_t, line_color=ui.CHART["ink"], line_dash="dot", line_width=1.5)
            fig.add_annotation(x=rec_t, y=1.0, yref="paper", text=f"model threshold {rec_t:.3f}", showarrow=False,
                               xanchor="left", xshift=6, yshift=-6, font=dict(size=12, color=ui.CHART["muted"]))
            fig.add_trace(go.Scatter(x=[t], y=[s["realised_net_value"] / 1e6], mode="markers", showlegend=False,
                                     marker=dict(size=11, color=ui.CHART["ink"], line=dict(color="#ffffff", width=2)),
                                     hovertemplate=f"Selected {t:.3f}<br>Net value {ui.money(s['realised_net_value'])}"
                                                   f"<br>Approval {s['approval_rate']:.1%}<extra></extra>"))
            ui.chart_layout(fig, title)
            ymin = float(min(curve["realised_net_value"].min(), s["net_value_approve_all"]) / 1e6)
            fig.update_layout(xaxis=dict(title="Approval threshold (probability of default)", range=[0, 1]),
                              yaxis=dict(title="Realised net value (USD m)", range=[min(0, ymin * 1.1), None]))
            if chosen == "approval":
                fig.update_layout(yaxis2=dict(title="Approval rate (%)", overlaying="y", side="right", range=[0, 100],
                                              showgrid=False, tickfont=dict(color=ui.CHART["muted"], size=12),
                                              title_font=dict(color=ui.CHART["muted"], size=12)), margin=dict(r=56))
            with st.container(key="picture_fixed"):
                U.chart(fig, height=400, key="portfolio_picture")
        elif chosen == "default":
            geo = pd.DataFrame({"State": portfolio["State"], "y": y, "approved": approved})
            geo = geo[geo["approved"]].groupby("State").agg(rate=("y", "mean"), loans=("y", "size")).reset_index()
            geo = geo[geo["State"].str.len() == 2]
            fig = go.Figure(go.Choropleth(
                locations=geo["State"], z=geo["rate"] * 100, locationmode="USA-states", colorscale=ui.SEQUENTIAL,
                marker_line_color="#ffffff", marker_line_width=0.8,
                colorbar=dict(title=dict(text="%", font=dict(size=12, color=ui.CHART["muted"])), thickness=10, len=0.7,
                              tickfont=dict(size=12, color=ui.CHART["muted"]), x=0.98, xanchor="right"),
                customdata=geo["loans"],
                hovertemplate="%{location}<br>Default rate %{z:.1f}%<br>%{customdata:,} approved loans<extra></extra>",
            ))
            ui.chart_layout(fig, "Default rate of approved loans by state")
            fig.update_layout(geo=dict(scope="usa", bgcolor="rgba(0,0,0,0)", showlakes=False),
                              margin=dict(l=0, r=8, t=56, b=0), clickmode="event+select")
            # a chart key that remounts the map when the filter is cleared
            with st.container(key="picture_fixed"):
                event = U.chart(fig, height=400, key=f"map_{U.filter_version()}", on_select="rerun")
            if U.apply_selection(event, "portfolio_map", "state", lookup=lambda pt: pt.get("location")):
                st.rerun()
        else:
            _, el = U.expected_value(p, disb, portion)
            in_state = (portfolio["State"] == flt["state"]).to_numpy() if flt.get("state") else np.ones(len(p), bool)
            sec = pd.DataFrame({"Sector": portfolio["Sector"], "el": el, "p": p})[approved & in_state]
            sec = sec.groupby("Sector").agg(loss=("el", "sum"), pd_mean=("p", "mean"), loans=("p", "size")).reset_index()
            sec = sec[sec["loss"] > 0]
            if sec.empty:
                ui.card("Expected loss by sector", None,
                        ui.empty("No approved loan carries an expected loss here: raise the threshold or clear the filter."))
            else:
                span = float(sec["pd_mean"].max() - sec["pd_mean"].min()) or 1.0
                level = list((sec["pd_mean"] - sec["pd_mean"].min()) / span)
                colours = sample_colorscale(ui.SEQUENTIAL, level)
                text_colours = ["#ffffff" if v > 0.5 else ui.CHART["ink"] for v in level]
                root = "All sectors"
                fig = go.Figure(go.Treemap(
                    labels=list(sec["Sector"]) + [root], parents=[root] * len(sec) + [""],
                    values=list(sec["loss"]) + [float(sum(sec["loss"])) * 1.000001], branchvalues="total",
                    marker=dict(colors=colours + ["#ffffff"], line=dict(color="#ffffff", width=2)),
                    customdata=np.vstack([np.stack([sec["pd_mean"] * 100, sec["loans"]], axis=1),
                                          [[float(np.average(sec["pd_mean"], weights=sec["loans"])) * 100,
                                            float(sec["loans"].sum())]]]),
                    texttemplate="<b>%{label}</b><br>USD %{value:,.3s}",
                    textfont=dict(size=12, color=text_colours + [ui.CHART["ink"]]),
                    pathbar=dict(visible=False), tiling=dict(pad=2),
                    hovertemplate="%{label}<br>Expected loss USD %{value:,.0f}<br>Mean PD %{customdata[0]:.1f}%"
                                  "<br>%{customdata[1]:,.0f} loans<extra></extra>",
                ))
                where = f", {U.state_name(flt['state'])}" if flt.get("state") else ""
                ui.chart_layout(fig, f"Expected loss of approved loans by sector{where}")
                fig.update_layout(margin=dict(l=4, r=4, t=56, b=4), uniformtext=dict(minsize=10, mode="hide"))
                with st.container(key="picture_fixed"):
                    U.chart(fig, height=400, key="portfolio_picture")

        with st.container(key="pick_buttons"):
            cols = st.columns(len(PICKS))
            for col, (key, label) in zip(cols, PICKS):
                if col.button(label, key=f"btn_figures_{key}", type="primary" if key == chosen else "secondary",
                              use_container_width=True):
                    st.session_state["portfolio_pick"] = key
                    st.rerun()

base_default = float(y.mean())
nv_gain = s["realised_net_value"] - s["net_value_approve_all"]
dr_change = (s["realised_default_rate_approved"] - base_default) * 100
with kpi_slot:
    ui.kpis([
        ("Approval rate", ui.pct(s["approval_rate"]), f"{(s['approval_rate'] - 1) * 100:+.1f} pts vs all"),
        ("Default rate", ui.pct(s["realised_default_rate_approved"]), f"{dr_change:+.1f} pts vs all"),
        ("Realised value", ui.money(s["realised_net_value"]), f"{signed_money(nv_gain)} vs all"),
        ("Loss avoided", ui.money(s["loss_avoided"]), f"-{s['loss_reduction']:.1%} vs all"),
    ])

ui.strip([
    ui.strip_title("Base case"),
    ui.strip_value("approve every loan", ui.money(s["net_value_approve_all"])),
    ui.strip_value("default rate, all loans", ui.pct(base_default)),
    ui.strip_value("model threshold", f"{rec_t:.3f}"),
    f'<span class="l">{len(p):,} held-out loans · realised outcomes</span>',
])


# ---------------------------------------------------------------------------
# The sheet: method and the objectives
# ---------------------------------------------------------------------------

@st.dialog("How it is worked out")
def sheet() -> None:
    bullets = [
        f"Data: the {len(p):,} held-out loans, scored with the loaded model; every figure uses the recorded outcomes.",
        "A loan is approved when its probability of default is below the threshold; the threshold is shared with "
        "the Stress test and Monte Carlo pages.",
        "Realised net value: the margin on approved loans that were repaid, minus the unguaranteed loss and workout "
        "cost on approved loans that were charged off.",
        "Loss avoided: the charge-off loss if every loan were approved, minus the loss of the loans approved here.",
        f"The business minimum is {min_rate:.0%} of loans approved.",
    ]
    U.html(ui.eyebrow("Method") + "".join(ui.note(b, "accent") for b in bullets))
    U.html(ui.eyebrow("Objectives") + ui.rows([{"a": label, "a_small": tip, "b": f"{objectives[k]:.3f}"}
                                               for k, label, tip in OBJ]
                                              + [{"a": "Model threshold", "a_small": "Stored in the model metadata.",
                                                  "b": f"{rec_t:.3f}"}]))


if st.session_state.get("portfolio_sheet"):
    st.session_state.pop("portfolio_sheet")
    sheet()
