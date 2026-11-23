"""Bank league: how the lenders in the data compare on volume, default rate, size and guarantee."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import experience as X  # noqa: E402
import utils as U  # noqa: E402

X.page_start("banks")
U.page_header(
    "Bank league",
    "Every lender with enough loans in the data, ranked. Search the table, click a bubble in the chart or pick a bank "
    "to open its profile.",
    eyebrow="Lenders",
)

league = U.load_aggregate("bank_league.parquet")
years = U.load_aggregate("bank_years.parquet")
manifest = U.aggregate_manifest()
if league is None or years is None:
    st.error("The bank aggregates are missing. Run `python app/build_assets.py aggregates` and commit app/assets/aggregates.")
    st.stop()

league = league.copy()
league["rank_volume"] = league["loans"].rank(ascending=False, method="min").astype(int)
overall = float(league["defaults"].sum() / league["loans"].sum())

c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
query = c1.text_input("Search lenders", key="bank_search", placeholder="Type part of a bank name, for example WELLS")
lo, hi = int(league["loans"].min()), int(league["loans"].max())
min_loans = c2.select_slider("Minimum loans", options=sorted({lo, 100, 250, 500, 1000, 5000} & set(range(lo, hi + 1)) | {lo}),
                             value=lo if lo >= 100 else 100, key="bank_min")
view = league[league["loans"] >= min_loans]
if query:
    view = view[view["Bank"].str.contains(query.strip(), case=False, regex=False)]

selected = st.session_state.get("bank_selected")
if selected not in set(league["Bank"]):
    selected = league.iloc[0]["Bank"]

left, right = st.columns([1.45, 1], gap="large")
with left:
    sv = view.head(600)
    colour = [U.COLOURS["red"] if b == selected else U.hex_rgba(U.COLOURS["navy"], 0.55) for b in sv["Bank"]]
    fig = go.Figure(go.Scatter(
        x=sv["loans"], y=sv["default_rate"] * 100, mode="markers",
        marker=dict(size=np.clip(np.sqrt(sv["average_loan"]) / 22, 6, 46), color=sv["guaranteed_share"] * 100,
                    colorscale=[[0, U.COLOURS["amber"]], [0.5, U.COLOURS["teal"]], [1, U.COLOURS["navy"]]],
                    cmin=40, cmax=90, line=dict(color=["#0B1F3A" if b == selected else "#FFFFFF" for b in sv["Bank"]],
                                                    width=[3 if b == selected else 1 for b in sv["Bank"]]),
                    colorbar=dict(title=dict(text="Guaranteed %", font=dict(size=11)), thickness=10, len=0.7), opacity=0.85),
        customdata=np.stack([sv["Bank"].str.title(), sv["average_loan"], sv["guaranteed_share"] * 100,
                             sv["same_state_share"] * 100, sv["bank_state"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[4]})<br>%{x:,.0f} loans<br>Default rate %{y:.1f}%"
                      "<br>Average loan USD %{customdata[1]:,.0f}<br>Guaranteed %{customdata[2]:.0f}%"
                      "<br>Same-state borrowers %{customdata[3]:.0f}%<br>Click for the profile<extra></extra>"))
    fig.add_hline(y=overall * 100, line_dash="dot", line_color=U.COLOURS["grey"],
                  annotation_text=f"all lenders {overall:.1%}", annotation_position="top right",
                  annotation_font=dict(size=11, color=U.MUTED))
    fig.update_layout(title="Default rate against volume (bubble size: average loan)", clickmode="event+select",
                      xaxis=dict(title="Loans in the data (log scale)", type="log"), yaxis=dict(title="Default rate (%)"),
                      margin=dict(l=8, r=8, t=60, b=40))
    event = U.chart(fig, height=470, key=f"bank_scatter_{st.session_state.get('bank_v', 0)}", on_select="rerun")
    pts = list(event.selection.points) if event is not None and hasattr(event, "selection") else []
    sig = repr(pts)
    if pts and st.session_state.get("_bank_seen") != sig:
        st.session_state["_bank_seen"] = sig
        k = int(pts[0].get("point_index", pts[0].get("point_number", 0)))
        if 0 <= k < len(sv):
            st.session_state["bank_selected"] = sv.iloc[k]["Bank"]
            st.rerun()
with right:
    options = list(league["Bank"])
    st.session_state["bank_pick"] = selected
    st.selectbox("Bank profile", options, key="bank_pick", format_func=lambda b: b.title(),
                 on_change=lambda: st.session_state.update(bank_selected=st.session_state["bank_pick"]))
    b = league.set_index("Bank").loc[selected]
    yr = years[years["Bank"] == selected].sort_values("ApprovalYear")
    spark = ""
    if len(yr) >= 2:
        xs = yr["ApprovalYear"].to_numpy(dtype=float)
        ys = yr["loans"].to_numpy(dtype=float)
        W, H = 320, 64
        px = (xs - xs.min()) / max(xs.max() - xs.min(), 1) * (W - 8) + 4
        py = H - 6 - ys / max(ys.max(), 1) * (H - 14)
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y_:.1f}" for i, (x, y_) in enumerate(zip(px, py)))
        area = path + f" L{px[-1]:.1f},{H - 2} L{px[0]:.1f},{H - 2} Z"
        top = int(np.argmax(ys))
        spark = (f'<svg class="cl-spark" viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="img" '
                 f'aria-label="Approvals by year, peak {int(ys[top]):,} in {int(xs[top])}">'
                 f'<path d="{area}" fill="{U.hex_rgba(U.COLOURS["teal"], 0.18)}"/>'
                 f'<path d="{path}" fill="none" stroke="{U.COLOURS["teal"]}" stroke-width="2.2" vector-effect="non-scaling-stroke"/>'
                 f'<circle cx="{px[top]:.1f}" cy="{py[top]:.1f}" r="3.5" fill="{U.COLOURS["red"]}"/></svg>'
                 f'<div class="cl-mix-legend"><span>{int(xs.min())}</span><span>peak {int(ys[top]):,} loans in {int(xs[top])}</span>'
                 f'<span>{int(xs.max())}</span></div>')
    rel = b["default_rate"] / overall if overall else 1
    tone = "below" if rel < 0.9 else ("above" if rel > 1.1 else "close to")
    U.html(
        f'<div class="cl-card"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
        f'<h4 style="margin:0">{U.icon("bank", 18)} {selected.title()}</h4><span class="cl-rank">#{int(b["rank_volume"])}</span></div>'
        f'<p style="margin-top:4px">Based in {U.state_name(b["bank_state"])}, lending in {int(b["states_served"])} states '
        f'from {int(b["first_year"])} to {int(b["last_year"])}. Its default rate is {tone} the average of all lenders.</p>'
        '<div class="cl-profile">'
        f'<div><span>Loans</span><b>{int(b["loans"]):,}</b></div>'
        f'<div><span>Default rate</span><b>{b["default_rate"]:.1%}</b></div>'
        f'<div><span>Average loan</span><b>{U.fmt_usd(float(b["average_loan"]))}</b></div>'
        f'<div><span>Guaranteed share</span><b>{b["guaranteed_share"]:.0%}</b></div>'
        f'<div><span>Same-state borrowers</span><b>{b["same_state_share"]:.0%}</b></div>'
        f'<div><span>Average term</span><b>{b["average_term_years"]:.1f} years</b></div></div>'
        f'<div class="cl-tile-label" style="margin-top:14px">Approvals by year</div>{spark}</div>'
    )
    if st.button("Score a loan with this bank", key="bank_to_score", icon=":material/speed:", use_container_width=True):
        case = {**U.BORDERLINE, "Bank": selected, "BankState": b["bank_state"]}
        import analytics as A
        st.session_state.pop("_case_loaded", None)
        st.query_params.clear()
        st.query_params["case"] = A.encode_case(case)
        U.switch_page("pages/score_loan.py")

U.section_title("League table", "grid")
table = view[["rank_volume", "Bank", "bank_state", "loans", "default_rate", "average_loan", "guaranteed_share",
              "same_state_share"]].copy()
table["Bank"] = table["Bank"].str.title()
table["average_loan"] = table["average_loan"].round(0)
st.dataframe(table, hide_index=True, use_container_width=True, height=420, column_config={
    "rank_volume": st.column_config.NumberColumn("Rank", help="Rank by number of loans."),
    "bank_state": "State",
    "loans": st.column_config.NumberColumn("Loans", format="localized"),
    "default_rate": st.column_config.ProgressColumn("Default rate", format="percent", min_value=0,
                                                    max_value=float(max(view["default_rate"].max(), 0.01)) if len(view) else 1.0),
    "average_loan": st.column_config.NumberColumn("Average loan (USD)", format="localized"),
    "guaranteed_share": st.column_config.NumberColumn("Guaranteed share", format="percent"),
    "same_state_share": st.column_config.NumberColumn("Same-state share", format="percent"),
})
if len(view) >= 10:
    big = view.head(10)
    U.html(f'<div class="cl-insight">{U.icon("spark", 18)}<span>The ten largest lenders shown made {big["loans"].sum():,} loans '
           f'with a combined default rate of {big["defaults"].sum() / big["loans"].sum():.1%}, against {overall:.1%} for all '
           f'{len(league):,} lenders in the league. Click a column header to sort.</span></div>')
st.caption(f"Source: committed aggregate of all cleaned loans (app/assets/aggregates/bank_league.parquet); lenders with at least "
           f"{manifest.get('bank_min_loans', 'n/a')} loans. Default rates are historical and include loans of every age.")
