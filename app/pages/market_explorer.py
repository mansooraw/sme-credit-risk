"""Market explorer: the figure strip for the current filter, the picture on the left (the animated year-by-year
map, the default rate by year, the mean loan by sector, the default rate by guaranteed share) with its pill row,
and on the right the filter card, the top states in the dark panel and the rows that open sheets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
import utils as U  # noqa: E402
from src import config  # noqa: E402

ui.page("Market explorer", "Default rate by state, year by year: click a state or a bar to filter.", eyebrow="Explore",
        heading="Fifty states, thirty years, one map.")

sy = U.load_aggregate("market_state_sector_year.parquet")
tp = U.load_aggregate("market_state_sector_term_portion.parquet")
manifest = U.aggregate_manifest()
if sy is None or tp is None:
    U.html(ui.empty("The market aggregates are missing: run python app/build_assets.py aggregates and commit app/assets/aggregates."))
    st.stop()

MIN_YEAR_LOANS = 1500   # years with fewer loans nationally are left out of the map
MIN_BAR_LOANS = 50      # sectors and bands with fewer loans are left out of the bars
HEIGHT = 400
TOP_STATES = 3   # the rows that fit in the dark panel beside the picture

states = sorted(s for s in sy["State"].unique() if len(str(s)) == 2 and s in U.US_STATES)
sectors = sorted(sy["Sector"].unique())
flt = U.filters()
v = U.filter_version()


def by(df, state=True, sector=True):
    m = pd.Series(True, index=df.index)
    if state and flt.get("state"):
        m &= df["State"] == flt["state"]
    if sector and flt.get("sector"):
        m &= df["Sector"] == flt["sector"]
    return df[m]


def short_sector(s: str) -> str:
    """A sector name that fits a row label on one line."""
    s = str(s)
    if len(s) >= 28:
        s = s.split(",")[0]
    if len(s) >= 28:
        s = s.split(" and ")[0]
    return s


cur = by(sy)
n, dflt, disb, guar = (float(cur[c].sum()) for c in ("loans", "defaults", "disbursed", "guaranteed"))
rate = dflt / n if n else None
overall = float(sy["defaults"].sum() / sy["loans"].sum())
where = f" in {U.state_name(flt['state'])}" if flt.get("state") else ""
national = sy.groupby("ApprovalYear")[["loans", "defaults"]].sum()
years = [int(y) for y in national.index if national.loc[y, "loans"] >= MIN_YEAR_LOANS]
names = ([U.state_name(flt["state"])] if flt.get("state") else []) + ([flt["sector"]] if flt.get("sector") else [])


def clickable(fig, key):
    """The picture, clickable: the same part as ui.picture, with a key that remounts when the filter is cleared."""
    with st.container(key="picture_fixed"):
        return U.chart(fig, height=HEIGHT, key=key, on_select="rerun")


# ---------------------------------------------------------------------------
# The middle pictures
# ---------------------------------------------------------------------------

def picture_map():
    geo = by(sy, state=False).groupby(["ApprovalYear", "State"], as_index=False)[["loans", "defaults"]].sum()
    geo = geo[geo["State"].isin(states) & geo["ApprovalYear"].isin(years)]
    geo["rate"] = geo["defaults"] / geo["loans"] * 100
    big = geo["loans"] >= 50
    zmax = float(np.nanpercentile(geo.loc[big, "rate"], 97)) if big.any() else 50.0

    def frame_trace(yr):
        g = geo[geo["ApprovalYear"] == yr].set_index("State").reindex(states)
        width = [2.5 if s == flt.get("state") else 0.7 for s in states]
        colour = [ui.CHART["ink"] if s == flt.get("state") else "#ffffff" for s in states]
        return go.Choropleth(
            locations=states, z=g["rate"].to_numpy(), locationmode="USA-states", zmin=0, zmax=zmax,
            colorscale=ui.SEQUENTIAL, marker=dict(line=dict(color=colour, width=width)),
            customdata=np.stack([[U.state_name(s) for s in states], g["loans"].fillna(0).to_numpy(),
                                 g["defaults"].fillna(0).to_numpy()], axis=1),
            colorbar=dict(title=dict(text="Default %", font=dict(size=12, color=ui.CHART["muted"])), thickness=10, len=0.7,
                          x=0.98, xanchor="right", tickfont=dict(size=12, color=ui.CHART["muted"])),
            hovertemplate="<b>%{customdata[0]}</b> " + str(yr) + "<br>Default rate %{z:.1f}%<br>%{customdata[1]:,.0f} loans, "
                          "%{customdata[2]:,.0f} charged off<br>Click to filter<extra></extra>",
        )

    nat_rate = (national["defaults"] / national["loans"]).loc[years] if years else None
    start_year = int(nat_rate.idxmax()) if years else None  # opens on the worst year nationally
    still = dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0), mode="immediate")
    fig = go.Figure(data=[frame_trace(start_year)], frames=[go.Frame(data=[frame_trace(yr)], name=str(yr)) for yr in years])
    ui.chart_layout(fig, f"Default rate by state, year by year{where}")
    fig.update_layout(
        geo=dict(scope="usa", bgcolor="rgba(0,0,0,0)", showlakes=False),
        margin=dict(l=0, r=0, t=56, b=0), clickmode="event+select",
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0.0, xanchor="left", yanchor="bottom", direction="left",
                          pad=dict(r=6, t=6), bgcolor=ui.CHART["ink"], font=dict(color=ui.CHART["canvas"], size=12),
                          buttons=[dict(label="Run", method="animate",
                                        args=[None, dict(frame=dict(duration=650, redraw=True), transition=dict(duration=0),
                                                         fromcurrent=True, mode="immediate")]),
                                   dict(label="Stop", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                                                           mode="immediate")])])],
        sliders=[dict(active=years.index(start_year) if years else 0, x=0.14, len=0.84, y=0.0, yanchor="bottom",
                      pad=dict(t=6, b=2), currentvalue=dict(prefix="Approval year ", font=dict(size=12, color=ui.CHART["ink"])),
                      font=dict(size=12, color=ui.CHART["muted"]), tickcolor=ui.CHART["muted"],
                      steps=[dict(label=str(yr), method="animate", args=[[str(yr)], still]) for yr in years])],
    )
    event = clickable(fig, f"mx_map_{v}")
    if U.apply_selection(event, "mx_map", "state", lookup=lambda pt: pt.get("location")):
        st.rerun()


def picture_year():
    tl = cur.groupby("ApprovalYear", as_index=False)[["loans", "defaults"]].sum()
    tl = tl[tl["ApprovalYear"] >= (years[0] if years else tl["ApprovalYear"].min())]
    tl["rate"] = tl["defaults"] / tl["loans"]
    r0, r1 = pd.Timestamp(config.RECESSION_START), pd.Timestamp(config.RECESSION_END)
    fig = go.Figure()
    fig.add_vrect(x0=r0.year + (r0.dayofyear - 1) / 365, x1=r1.year + r1.dayofyear / 365, fillcolor=ui.CHART["accent_soft"],
                  opacity=1, line_width=0, layer="below", annotation_text="Great Recession", annotation_position="top left",
                  annotation_font=dict(size=12, color=ui.CHART["muted"]))
    fig.add_trace(go.Scatter(x=tl["ApprovalYear"], y=tl["rate"], mode="lines+markers", line=dict(color=ui.CHART["accent"], width=2.5),
                             marker=dict(size=5, color=ui.CHART["accent"]), customdata=tl["loans"], showlegend=False,
                             hovertemplate="%{x}<br>Default rate %{y:.1%}<br>%{customdata:,.0f} loans<extra></extra>"))
    fig.add_hline(y=overall, line_color=ui.CHART["ink"], line_dash="dot", line_width=1.5,
                  annotation_text=f"all loans {overall:.1%}", annotation_position="top right",
                  annotation_font=dict(size=12, color=ui.CHART["muted"]))
    ui.chart_layout(fig, f"Default rate by approval year{where}")
    fig.update_layout(xaxis=dict(title="Approval year"), yaxis=dict(title="Default rate", tickformat=".0%", rangemode="tozero"))
    ui.picture(fig, height=HEIGHT)


def picture_sector():
    sec = by(sy, sector=False).groupby("Sector", as_index=False)[["loans", "defaults", "disbursed"]].sum()
    sec = sec[sec["loans"] >= MIN_BAR_LOANS]
    sec["mean"] = sec["disbursed"] / sec["loans"]
    sec["rate"] = sec["defaults"] / sec["loans"]
    sec = sec.sort_values("mean")
    sel = flt.get("sector")
    colours = [ui.CHART["accent"] if (s == sel or not sel) else ui.CHART["accent_light"] for s in sec["Sector"]]
    fig = go.Figure(go.Bar(
        x=sec["mean"], y=sec["Sector"], orientation="h", marker=dict(color=colours),
        customdata=np.stack([sec["loans"], sec["rate"] * 100], axis=1),
        text=[ui.money(m) for m in sec["mean"]], textposition="outside", cliponaxis=False,
        textfont=dict(size=12, color=ui.CHART["muted"]),
        hovertemplate="<b>%{y}</b><br>Mean loan USD %{x:,.0f}<br>%{customdata[0]:,.0f} loans"
                      "<br>Default rate %{customdata[1]:.1f}%<br>Click to filter<extra></extra>"))
    ui.chart_layout(fig, f"Mean loan by sector{where}")
    fig.update_layout(xaxis=dict(title="Mean loan (USD)", range=[0, float(sec["mean"].max()) * 1.2] if len(sec) else None),
                      yaxis=dict(showgrid=False), bargap=0.25, clickmode="event+select", showlegend=False)
    event = clickable(fig, f"mx_sector_{v}")
    if U.apply_selection(event, "mx_sector", "sector", lookup=lambda pt: pt.get("y")):
        st.rerun()


def picture_guarantee():
    bands = manifest.get("portion_bands") or sorted(tp["PortionBand"].unique())
    g = by(tp).groupby("PortionBand")[["loans", "defaults"]].sum().reindex(bands)
    g = g[g["loans"].fillna(0) >= MIN_BAR_LOANS].copy()
    g["rate"] = g["defaults"] / g["loans"]
    colours = [ui.CHART["bad"] if r > overall else ui.CHART["good"] for r in g["rate"]]
    fig = go.Figure(go.Bar(x=list(g.index), y=g["rate"], marker=dict(color=colours), customdata=g["loans"],
                           text=[ui.pct(r) for r in g["rate"]], textposition="outside", cliponaxis=False,
                           textfont=dict(size=12, color=ui.CHART["muted"]),
                           hovertemplate="Guaranteed %{x}<br>Default rate %{y:.1%}<br>%{customdata:,.0f} loans<extra></extra>"))
    fig.add_hline(y=overall, line_color=ui.CHART["ink"], line_dash="dot", line_width=1.5,
                  annotation_text=f"all loans {overall:.1%}", annotation_position="top right",
                  annotation_font=dict(size=12, color=ui.CHART["muted"]))
    ui.chart_layout(fig, f"Default rate by guaranteed share{where}")
    fig.update_layout(xaxis=dict(title="Share of the loan guaranteed by the SBA", showgrid=False),
                      yaxis=dict(title="Default rate", tickformat=".0%", rangemode="tozero"), bargap=0.3, showlegend=False)
    ui.picture(fig, height=HEIGHT)


PICTURES = {"loans": picture_map, "default": picture_year, "mean": picture_sector, "guaranteed": picture_guarantee}

# ---------------------------------------------------------------------------
# The page: figure strip, picture left, filter and top states right, data strip
# ---------------------------------------------------------------------------

scope = ", ".join(names) or "All loans"
chosen = st.session_state.get("market_pick", "loans")
if chosen not in PICTURES:
    chosen = "loans"
ui.kpis([
    ("Loans", f"{n:,.0f}", scope if len(scope) < 28 else "Current filter"),
    ("Default rate", "n/a" if rate is None else ui.pct(rate), f"All loans {overall:.1%}"),
    ("Mean loan", ui.money(disb / n) if n else "n/a", "Disbursed per loan"),
    ("Guaranteed share", ui.pct(guar / disb, 0) if disb else "n/a", "Of disbursed"),
])

top_st = by(sy, state=False).groupby("State")[["loans", "defaults"]].sum()
top_st = top_st[top_st.index.isin(states)].sort_values("loans", ascending=False)
top_st["rate"] = top_st["defaults"] / top_st["loans"]

with st.container(key="two"):
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        PICTURES[chosen]()
        with st.container(key="pick_buttons"):
            cols = st.columns(4)
            for col, key, label in zip(cols, ("loans", "default", "mean", "guaranteed"),
                                       ("Year map", "By year", "By sector", "By guarantee")):
                if col.button(label, key=f"btn_figures_{key}", type="primary" if key == chosen else "secondary",
                              use_container_width=True):
                    st.session_state["market_pick"] = key
                    st.rerun()

    with right:
        with st.container(key="market_controls"):
            line = f"Filtered: {', '.join(names)}." if names else "Click a state or a bar, or choose here."
            U.html(ui.card_html("Filter", line, ""))
            st.session_state["mx_state"] = flt.get("state", "All")
            st.session_state["mx_sector"] = flt.get("sector", "All")
            c1, c2 = st.columns(2)
            with c1:
                st.selectbox("State", ["All"] + states, key="mx_state",
                             format_func=lambda s: "All states" if s == "All" else U.state_name(s),
                             on_change=lambda: U.set_filter("state", st.session_state["mx_state"]))
            with c2:
                st.selectbox("Sector", ["All"] + sectors, key="mx_sector",
                             format_func=lambda s: "All sectors" if s == "All" else short_sector(s),
                             on_change=lambda: U.set_filter("sector", st.session_state["mx_sector"]))
            if flt:
                # the Clear row: the kit's row part (as ui.row_button) with the button key the smoke test presses
                with st.container(key="row_mx_clear"):
                    U.html('<div class="ui-row"><div class="a">Clear all filters</div><div class="b"></div>'
                           '<span class="arrow">&#8250;</span></div>')
                    if st.button("Open Clear all filters", key="mx_clear", use_container_width=True):
                        U.clear_filters()
                        st.rerun()

        ui.card("Top states", "Most loans; rate against all loans.", ui.rows([
            {"key": s, "a": U.state_name(s), "a_small": f"{int(r['loans']):,} loans",
             "badge": "Low" if r["rate"] < overall else "High", "b_small": ui.pct(r["rate"])}
            for s, r in top_st.head(TOP_STATES).iterrows()], chosen=flt.get("state")), kind="panel")

        with st.container(key="sheet_rows"):
            U.html(ui.card_html("States and sectors", "Open a row for the full list or the method.", ""))
            if ui.row_button({"a": "See all", "a_small": f"{len(top_st)} states, every sector"}, key="mx_all"):
                st.session_state["market_sheet"] = "all"
            if ui.row_button({"a": "How it is worked out", "a_small": "Data, filters and limits"}, key="mx_how"):
                st.session_state["market_sheet"] = "how"

r0, r1 = pd.Timestamp(config.RECESSION_START).year, pd.Timestamp(config.RECESSION_END).year
inside = cur["ApprovalYear"].between(r0, r1)
rec, rest = cur[inside][["loans", "defaults"]].sum(), cur[~inside][["loans", "defaults"]].sum()
ui.strip([
    ui.strip_title("Recession"),
    ui.strip_value(f"approved {r0} to {r1}", ui.pct(rec["defaults"] / rec["loans"]) if rec["loans"] else "n/a"),
    ui.strip_value("outside the window", ui.pct(rest["defaults"] / rest["loans"]) if rest["loans"] else "n/a"),
    f'<span class="l">SBA 7(a) loans · 1962 to 2014 · {manifest.get("n_loans", 0):,} cleaned loans</span>',
])


# ---------------------------------------------------------------------------
# The sheets
# ---------------------------------------------------------------------------

@st.dialog("All states and sectors", width="large")
def sheet_all() -> None:
    U.html(ui.eyebrow(f"States{where}"))
    table = top_st.reset_index()[["State", "loans", "rate"]]
    table["State"] = table["State"].map(U.state_name)
    st.dataframe(table, hide_index=True, use_container_width=True, height=300, key="mx_states_table",
                 column_config={"loans": st.column_config.NumberColumn("Loans", format="localized"),
                                "rate": st.column_config.NumberColumn("Default rate", format="percent")})
    sec = by(sy, sector=False).groupby("Sector")[["loans", "defaults", "disbursed"]].sum().sort_values("loans", ascending=False)
    sec = sec.assign(rate=sec["defaults"] / sec["loans"], mean=(sec["disbursed"] / sec["loans"]).round(0)).reset_index()
    U.html(ui.eyebrow("Sectors"))
    st.dataframe(sec[["Sector", "loans", "rate", "mean"]], hide_index=True, use_container_width=True, height=300,
                 key="mx_sectors_table",
                 column_config={"loans": st.column_config.NumberColumn("Loans", format="localized"),
                                "rate": st.column_config.NumberColumn("Default rate", format="percent"),
                                "mean": st.column_config.NumberColumn("Mean loan (USD)", format="localized")})


@st.dialog("How it is worked out")
def sheet_how() -> None:
    for line in [
        f"Source: the committed market aggregates of {manifest.get('n_loans', 0):,} cleaned SBA 7(a) loans "
        "(market_state_sector_year.parquet and market_state_sector_term_portion.parquet).",
        "Default rate is charged-off loans divided by all loans in the current filter; the guaranteed share is the "
        "SBA-guaranteed amount divided by the amount disbursed.",
        f"The map shows the default rate of each state year by year, for approval years with at least "
        f"{MIN_YEAR_LOANS:,} loans nationally; it opens on the worst year. Run plays the years in order.",
        f"Sectors and guarantee bands with fewer than {MIN_BAR_LOANS} loans are left out of the bars.",
        "Click a state on the map or a sector bar to filter every figure; Clear all filters removes them.",
        "Recent years look safer partly because those loans had little time to default before the data ends.",
    ]:
        U.html(ui.note(line, "accent"))


sheet = st.session_state.pop("market_sheet", None)
if sheet == "all":
    sheet_all()
elif sheet == "how":
    sheet_how()
