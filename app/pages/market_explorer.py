"""Market explorer: default rates across states, years, sectors and loan structures, with click-to-drill filters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import experience as X  # noqa: E402
import utils as U  # noqa: E402
from src import config  # noqa: E402

X.page_start("market")
U.page_header(
    "Market explorer",
    "Fifty years of guaranteed SME lending. Press play to watch default rates move across the states, then click a "
    "state on the map or a sector in the bar chart: every other chart filters to your selection.",
    eyebrow="Explore the data",
)

sy = U.load_aggregate("market_state_sector_year.parquet")
sa = U.load_aggregate("market_state_sector_age.parquet")
tp = U.load_aggregate("market_state_sector_term_portion.parquet")
manifest = U.aggregate_manifest()
if sy is None or sa is None or tp is None:
    st.error("The market aggregates are missing. Run `python app/build_assets.py aggregates` and commit app/assets/aggregates.")
    st.stop()

MIN_YEAR_LOANS = 1500   # years with fewer loans nationally are left out of the animation
MIN_CELL = 200          # smallest group quoted in an insight sentence


def insight(text: str) -> None:
    U.html(f'<div class="cl-insight">{U.icon("spark", 18)}<span>{text}</span></div>')


flt = U.filters()
states = sorted(s for s in sy["State"].unique() if len(str(s)) == 2 and s in U.US_STATES)
sectors = sorted(sy["Sector"].unique())

# Tap-friendly equivalents of the chart clicks
f1, f2 = st.columns(2)
st.session_state["mx_state"] = flt.get("state", "All")
st.session_state["mx_sector"] = flt.get("sector", "All")
f1.selectbox("State", ["All"] + states, key="mx_state", format_func=lambda v: "All states" if v == "All" else U.state_name(v),
             on_change=lambda: U.set_filter("state", st.session_state["mx_state"]))
f2.selectbox("Sector", ["All"] + sectors, key="mx_sector", format_func=lambda v: "All sectors" if v == "All" else v,
             on_change=lambda: U.set_filter("sector", st.session_state["mx_sector"]))
U.filter_chips(key="mx_clear")
flt = U.filters()
v = U.filter_version()


def by(df, state=True, sector=True):
    m = pd.Series(True, index=df.index)
    if state and flt.get("state"):
        m &= df["State"] == flt["state"]
    if sector and flt.get("sector"):
        m &= df["Sector"] == flt["sector"]
    return df[m]


# 1. Animated choropleth of default rate by state across approval years
U.section_title("Default rate by state, year by year", "map")
geo = by(sy, state=False).groupby(["ApprovalYear", "State"], as_index=False)[["loans", "defaults"]].sum()
geo = geo[geo["State"].isin(states)]
national = sy.groupby("ApprovalYear")[["loans", "defaults"]].sum()
years = [int(y) for y in national.index if national.loc[y, "loans"] >= MIN_YEAR_LOANS]
geo = geo[geo["ApprovalYear"].isin(years)]
geo["rate"] = geo["defaults"] / geo["loans"] * 100
zmax = float(np.nanpercentile(geo.loc[geo["loans"] >= 50, "rate"], 97)) if (geo["loans"] >= 50).any() else 50.0
scale = [[0, "#E3F4F1"], [0.35, U.COLOURS["teal"]], [0.7, U.COLOURS["amber"]], [1, U.COLOURS["red"]]]


def frame_trace(yr):
    g = geo[geo["ApprovalYear"] == yr].set_index("State").reindex(states)
    width = [3.5 if s == flt.get("state") else 0.7 for s in states]
    colour = [U.INK if s == flt.get("state") else "#FFFFFF" for s in states]
    return go.Choropleth(
        locations=states, z=g["rate"].to_numpy(), locationmode="USA-states", zmin=0, zmax=zmax, colorscale=scale,
        marker=dict(line=dict(color=colour, width=width)),
        customdata=np.stack([[U.state_name(s) for s in states], g["loans"].fillna(0).to_numpy(),
                             g["defaults"].fillna(0).to_numpy()], axis=1),
        colorbar=dict(title=dict(text="Default %", font=dict(size=11)), thickness=10, len=0.75, x=0.98, xanchor="right"),
        hovertemplate="<b>%{customdata[0]}</b> " + str(yr) + "<br>Default rate %{z:.1f}%<br>%{customdata[1]:,.0f} loans, "
                      "%{customdata[2]:,.0f} charged off<br>Click to filter<extra></extra>",
    )


nat_rate = (national["defaults"] / national["loans"]).loc[years] if years else None
start_year = int(nat_rate.idxmax()) if years else None  # open on the worst year, then play from the start
fig = go.Figure(data=[frame_trace(start_year)],
                frames=[go.Frame(data=[frame_trace(yr)], name=str(yr)) for yr in years])
fig.update_layout(
    geo=dict(scope="usa", bgcolor="rgba(0,0,0,0)", showlakes=False),
    margin=dict(l=0, r=0, t=10, b=0), clickmode="event+select",
    updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0.02, xanchor="left", yanchor="bottom", direction="left",
                      pad=dict(r=8, t=8), bgcolor=U.COLOURS["teal"], font=dict(color="#FFFFFF", size=12),
                      buttons=[dict(label="Play", method="animate",
                                    args=[None, dict(frame=dict(duration=650, redraw=True), transition=dict(duration=300),
                                                     fromcurrent=True, mode="immediate")]),
                               dict(label="Pause", method="animate",
                                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
    sliders=[dict(active=years.index(start_year) if years else 0, x=0.12, len=0.86, y=0.02, yanchor="bottom", pad=dict(t=8, b=4),
                  currentvalue=dict(prefix="Approval year ", font=dict(size=12, color=U.INK)),
                  font=dict(size=10, color=U.MUTED), tickcolor=U.MUTED,
                  steps=[dict(label=str(yr), method="animate",
                              args=[[str(yr)], dict(frame=dict(duration=0, redraw=True), mode="immediate")]) for yr in years])],
)
event = U.chart(fig, height=470, key=f"mx_map_{v}", on_select="rerun")
if U.apply_selection(event, "mx_map", "state", lookup=lambda pt: pt.get("location")):
    st.rerun()
if len(geo):
    nat = national.loc[years]
    peak = int((nat["defaults"] / nat["loans"]).idxmax())
    g_peak = geo[(geo["ApprovalYear"] == peak) & (geo["loans"] >= MIN_CELL)].sort_values("rate")
    sect = f" for {flt['sector']}" if flt.get("sector") else ""
    if len(g_peak):
        worst, best = g_peak.iloc[-1], g_peak.iloc[0]
        insight(f"Loans approved in {peak} defaulted most often nationally ({nat.loc[peak, 'defaults'] / nat.loc[peak, 'loans']:.1%}); "
                f"that year{sect}, {U.state_name(worst['State'])} reached {worst['rate']:.1f}% while "
                f"{U.state_name(best['State'])} stayed at {best['rate']:.1f}% (states with at least {MIN_CELL} loans).")

# 2. Sectors (click to filter) and 3. sunburst
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    sec = by(sy, sector=False).groupby("Sector", as_index=False)[["loans", "defaults", "disbursed"]].sum()
    sec["rate"] = sec["defaults"] / sec["loans"]
    sec = sec[sec["loans"] >= 50].sort_values("rate")
    colours = [U.COLOURS["red"] if s == flt.get("sector") else (U.hex_rgba(U.COLOURS["navy"], 0.75) if not flt.get("sector") else
               U.hex_rgba(U.COLOURS["navy"], 0.3)) for s in sec["Sector"]]
    fig = go.Figure(go.Bar(
        x=sec["rate"] * 100, y=sec["Sector"], orientation="h", marker=dict(color=colours),
        customdata=np.stack([sec["loans"], sec["defaults"], sec["disbursed"] / sec["loans"]], axis=1),
        text=[f"{r:.1%}" for r in sec["rate"]], textposition="outside", cliponaxis=False,
        textfont=dict(size=10, color=U.MUTED),
        hovertemplate="<b>%{y}</b><br>Default rate %{x:.1f}%<br>%{customdata[0]:,.0f} loans, %{customdata[1]:,.0f} charged off"
                      "<br>Average loan USD %{customdata[2]:,.0f}<br>Click to filter<extra></extra>"))
    where = f" in {U.state_name(flt['state'])}" if flt.get("state") else ""
    fig.update_layout(title=f"Default rate by sector{where}", barcornerradius=4, xaxis=dict(title="Default rate (%)"),
                      yaxis=dict(tickfont=dict(size=10.5)), margin=dict(l=8, r=40, t=60, b=40), clickmode="event+select")
    event = U.chart(fig, height=560, key=f"mx_sector_{v}", on_select="rerun")
    if U.apply_selection(event, "mx_sector", "sector", lookup=lambda pt: pt.get("y")):
        st.rerun()
    if len(sec) >= 2:
        insight(f"{sec.iloc[-1]['Sector']} has the highest default rate{where} ({sec.iloc[-1]['rate']:.1%}), "
                f"{sec.iloc[-1]['rate'] / max(sec.iloc[0]['rate'], 1e-9):.1f} times that of {sec.iloc[0]['Sector']} ({sec.iloc[0]['rate']:.1%}).")
with c2:
    sb = by(sa).groupby(["Sector", "NewExist"], as_index=False)[["loans", "defaults"]].sum()
    sb = sb[sb["loans"] > 0]
    ids, labels, parents, values, colours, hover = [], [], [], [], [], []
    for s_name, g in sb.groupby("Sector"):
        tot, dflt = g["loans"].sum(), g["defaults"].sum()
        ids.append(s_name); labels.append(s_name[:22]); parents.append(""); values.append(tot)
        colours.append(U.hex_rgba(U.COLOURS["navy"], 0.35 + 0.65 * min(dflt / tot / 0.4, 1)))
        hover.append(f"{s_name}<br>{tot:,.0f} loans, default rate {dflt / tot:.1%}")
        for _, r in g.iterrows():
            age = f"{s_name}/{r['NewExist']}"
            ids.append(age); labels.append(r["NewExist"]); parents.append(s_name); values.append(r["loans"])
            colours.append("#8FA9C8" if r["NewExist"] == "Existing" else ("#C9B37A" if r["NewExist"] == "New" else "#B8C2CF"))
            hover.append(f"{s_name}, {r['NewExist']} business<br>{r['loans']:,.0f} loans, default rate {r['defaults'] / r['loans']:.1%}")
            for lab, val, col in [("Repaid", r["loans"] - r["defaults"], U.COLOURS["teal"]), ("Charged off", r["defaults"], U.COLOURS["red"])]:
                if val > 0:
                    ids.append(f"{age}/{lab}"); labels.append(lab); parents.append(age); values.append(val); colours.append(col)
                    hover.append(f"{s_name}, {r['NewExist']}: {lab.lower()}<br>{val:,.0f} loans ({val / r['loans']:.1%})")
    fig = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
                                marker=dict(colors=colours, line=dict(color="#FFFFFF", width=1)), hovertext=hover,
                                hoverinfo="text", insidetextorientation="radial", maxdepth=3,
                                textfont=dict(size=10)))
    fig.update_layout(uniformtext=dict(minsize=9, mode="hide"))
    fig.update_layout(title="Sector, business age and outcome" + (f" in {U.state_name(flt['state'])}" if flt.get("state") else ""),
                      margin=dict(l=4, r=4, t=60, b=4))
    U.chart(fig, height=560, key="mx_sunburst")
    age = by(sa).groupby("NewExist")[["loans", "defaults"]].sum()
    if {"New", "Existing"} <= set(age.index):
        rn, re_ = age.loc["New", "defaults"] / age.loc["New", "loans"], age.loc["Existing", "defaults"] / age.loc["Existing", "loans"]
        scope = flt.get("sector", "these loans")
        insight(f"For {scope}{where}, new businesses default at {rn:.1%} against {re_:.1%} for established ones. "
                "Click a sector segment to zoom in and the centre to zoom out.")

# 4. Term by guarantee heatmap
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    term_bands = manifest.get("term_bands") or sorted(tp["TermBand"].unique())
    portion_bands = manifest.get("portion_bands") or sorted(tp["PortionBand"].unique())
    hm = by(tp).groupby(["TermBand", "PortionBand"], as_index=False)[["loans", "defaults"]].sum()
    hm["rate"] = hm["defaults"] / hm["loans"]
    rate = hm.pivot(index="TermBand", columns="PortionBand", values="rate").reindex(index=term_bands, columns=portion_bands)
    loans = hm.pivot(index="TermBand", columns="PortionBand", values="loans").reindex(index=term_bands, columns=portion_bands).fillna(0)
    z = np.where(loans.to_numpy() >= 20, rate.to_numpy() * 100, np.nan)
    text = np.where(np.isnan(z), "", np.vectorize(lambda x: f"{x:.0f}%")(np.nan_to_num(z)))
    fig = go.Figure(go.Heatmap(z=z, x=portion_bands, y=term_bands, colorscale=scale, zmin=0,
                               text=text, texttemplate="%{text}", textfont=dict(size=11),
                               customdata=loans.to_numpy(), xgap=2, ygap=2,
                               colorbar=dict(title=dict(text="%", font=dict(size=11)), thickness=10),
                               hovertemplate="Term %{y}<br>Guarantee %{x}<br>Default rate %{z:.1f}%<br>%{customdata:,.0f} loans<extra></extra>"))
    fig.update_layout(title="Default rate by term and guaranteed share", xaxis=dict(title="Guaranteed share", tickangle=-30),
                      yaxis=dict(title="Term", autorange="reversed"), margin=dict(l=8, r=8, t=60, b=60))
    U.chart(fig, height=430, key="mx_heatmap")
    cells = hm[hm["loans"] >= MIN_CELL].sort_values("rate")
    if len(cells) >= 2:
        hi_, lo_ = cells.iloc[-1], cells.iloc[0]
        insight(f"The riskiest structure is a {hi_['TermBand'].lower()} term with a {hi_['PortionBand']} guarantee "
                f"({hi_['rate']:.1%} defaulted, {hi_['loans']:,.0f} loans); {lo_['TermBand'].lower()} with "
                f"{lo_['PortionBand']} is the safest ({lo_['rate']:.1%}). Short terms carry the risk: long loans are usually secured on property.")
with c2:
    tl = by(sy).groupby("ApprovalYear", as_index=False)[["loans", "defaults", "disbursed"]].sum()
    tl = tl[tl["ApprovalYear"] >= (years[0] if years else tl["ApprovalYear"].min())]
    tl["rate"] = tl["defaults"] / tl["loans"] * 100
    fig = go.Figure()
    r0, r1 = pd.Timestamp(config.RECESSION_START), pd.Timestamp(config.RECESSION_END)
    fig.add_vrect(x0=r0.year + (r0.dayofyear - 1) / 365, x1=r1.year + r1.dayofyear / 365, fillcolor=U.COLOURS["red"],
                  opacity=0.12, line_width=0, annotation_text="Great Recession", annotation_position="top left",
                  annotation_font=dict(size=11, color=U.COLOURS["red"]))
    fig.add_trace(go.Bar(x=tl["ApprovalYear"], y=tl["loans"], name="Loans approved", marker_color=U.hex_rgba(U.COLOURS["navy"], 0.6),
                         customdata=tl["disbursed"], hovertemplate="%{x}<br>%{y:,.0f} loans<br>USD %{customdata:,.3s} disbursed<extra></extra>"))
    fig.add_trace(go.Scatter(x=tl["ApprovalYear"], y=tl["rate"], name="Default rate (%)", yaxis="y2", mode="lines+markers",
                             line=dict(color=U.COLOURS["red"], width=3), marker=dict(size=5),
                             hovertemplate="%{x}<br>Default rate %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="Loan volume and default rate by approval year", barcornerradius=3,
                      xaxis=dict(title="Approval year"), yaxis=dict(title="Loans"),
                      yaxis2=dict(title="Default rate (%)", overlaying="y", side="right", showgrid=False,
                                  tickfont=dict(color=U.MUTED, size=12), rangemode="tozero"),
                      legend=dict(y=1.06), margin=dict(l=8, r=50, t=80, b=40))
    U.chart(fig, height=430, key="mx_timeline")
    if len(tl):
        pk = tl.loc[tl["loans"].idxmax()]
        worst = tl[tl["loans"] >= MIN_CELL]
        worst = worst.loc[worst["rate"].idxmax()] if len(worst) else pk
        insight(f"Approvals peaked at {pk['loans']:,.0f} in {int(pk['ApprovalYear'])}. Loans approved in {int(worst['ApprovalYear'])} "
                f"defaulted most often ({worst['rate']:.1f}%): they were running when the recession hit. The latest years look "
                "safer partly because those loans had little time to default before the data ends.")

st.caption(f"Source: committed aggregates of all {manifest.get('n_loans', len(sy)):,} cleaned loans (app/assets/aggregates, built "
           f"{manifest.get('built_on', 'n/a')}). The map animates the years with at least {MIN_YEAR_LOANS:,} approvals nationally; "
           "states with fewer than 20 loans in a cell of the heatmap are left blank.")
