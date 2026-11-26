"""Bank league: the figure strip, the league drawn as ranked bars on the left (click a bar to choose a lender)
with its pill row, and on the right the lender card, the selected lender in the dark panel and the rows that
open the full ranked table, the method and a scored case."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
import utils as U  # noqa: E402

ui.page("Bank league", "Lenders in the data ranked by volume; choose one to see its profile.", eyebrow="Explore",
        heading="Lenders, side by side.")

league = U.load_aggregate("bank_league.parquet")
years = U.load_aggregate("bank_years.parquet")
manifest = U.aggregate_manifest()
if league is None or years is None:
    U.html(ui.empty("The bank aggregates are missing: run python app/build_assets.py aggregates and commit app/assets/aggregates."))
    st.stop()

BARS = 15             # lenders drawn in the middle
BEST_MIN_LOANS = 500  # a best default rate needs a book of at least this many loans
HEIGHT = 400

league = league.copy()
league["rank_volume"] = league["loans"].rank(ascending=False, method="min").astype(int)
league = league.sort_values("rank_volume").reset_index(drop=True)
overall = float(league["defaults"].sum() / league["loans"].sum())

selected = st.session_state.get("bank_selected")
if selected not in set(league["Bank"]):
    selected = league.iloc[0]["Bank"]
b = league.set_index("Bank").loc[selected]
eligible = league[league["loans"] >= BEST_MIN_LOANS]
best = eligible.loc[eligible["default_rate"].idxmin()]
largest = league.loc[league["total_disbursed"].idxmax()]


# ---------------------------------------------------------------------------
# The middle pictures
# ---------------------------------------------------------------------------

def bars(df, col, title, xlabel, fmt, text):
    df = df.iloc[::-1]
    colours = [ui.CHART["ink"] if bank == selected else ui.CHART["accent"] for bank in df["Bank"]]
    fig = go.Figure(go.Bar(
        x=df[col], y=[f"{int(r)}. {bank.title()}" for r, bank in zip(df["rank_volume"], df["Bank"])], orientation="h",
        marker=dict(color=colours), text=text(df), textposition="outside", cliponaxis=False,
        textfont=dict(size=12, color=ui.CHART["muted"]), customdata=df[["loans", "default_rate", "Bank"]].to_numpy(),
        hovertemplate="<b>%{y}</b><br>%{customdata[0]:,.0f} loans<br>Default rate %{customdata[1]:.1%}"
                      "<br>Click to choose this lender<extra></extra>"))
    ui.chart_layout(fig, title)
    fig.update_layout(xaxis=dict(title=xlabel, tickformat=fmt, range=[0, float(df[col].max()) * 1.18]),
                      yaxis=dict(showgrid=False), bargap=0.25, showlegend=False, clickmode="event+select")
    with st.container(key="picture_fixed"):
        event = U.chart(fig, height=HEIGHT, key=f"bank_bars_{st.session_state.get('bank_pick_fig', 'ranked')}", on_select="rerun")
    points = []
    try:
        points = list(event.selection.points) if event is not None else []
    except AttributeError:
        points = list((event or {}).get("selection", {}).get("points", []))
    sig = repr(points)
    if points and st.session_state.get("_bank_seen") != sig:
        st.session_state["_bank_seen"] = sig
        idx = int(points[0].get("point_index", points[0].get("point_number", -1)))
        if 0 <= idx < len(df):
            st.session_state["bank_selected"] = df.iloc[idx]["Bank"]
            st.rerun()


def picture_ranked():
    bars(league.head(BARS), "loans", "Lenders ranked by number of loans", "Loans", ",.0f",
         lambda d: [f"{int(x):,}" for x in d["loans"]])


def picture_best():
    bars(eligible.sort_values("default_rate").head(BARS), "default_rate",
         f"Lowest default rates, lenders with {BEST_MIN_LOANS}+ loans", "Default rate", ".0%",
         lambda d: [ui.pct(x) for x in d["default_rate"]])


def picture_largest():
    bars(league.sort_values("total_disbursed", ascending=False).head(BARS), "total_disbursed",
         "Largest books by amount disbursed", "Disbursed (USD)", "~s", lambda d: [ui.money(x) for x in d["total_disbursed"]])


def picture_selected():
    yr = years[years["Bank"] == selected].sort_values("ApprovalYear")
    rate = yr["defaults"] / yr["loans"].clip(lower=1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yr["ApprovalYear"], y=yr["loans"], name="Loans approved", marker_color=ui.CHART["accent_light"],
                         hovertemplate="%{x}<br>%{y:,.0f} loans<extra></extra>"))
    fig.add_trace(go.Scatter(x=yr["ApprovalYear"], y=rate, name="Default rate", yaxis="y2", mode="lines",
                             line=dict(color=ui.CHART["bad"], width=2.5), hovertemplate="%{x}<br>Default rate %{y:.1%}<extra></extra>"))
    ui.chart_layout(fig, f"{selected.title()}: approvals and default rate by year")
    fig.update_layout(xaxis=dict(title="Approval year"), yaxis=dict(title="Loans"),
                      yaxis2=dict(title="Default rate", overlaying="y", side="right", showgrid=False, tickformat=".0%",
                                  rangemode="tozero", tickfont=dict(color=ui.CHART["muted"], size=12),
                                  title_font=dict(color=ui.CHART["muted"], size=12)),
                      margin=dict(r=56))
    ui.picture(fig, height=HEIGHT)


PICTURES = {"ranked": picture_ranked, "best": picture_best, "largest": picture_largest, "selected": picture_selected}

# ---------------------------------------------------------------------------
# The page: figure strip, ranked bars left, lender card and profile right, data strip
# ---------------------------------------------------------------------------

chosen = st.session_state.get("bank_pick_fig", "ranked")
if chosen not in PICTURES:
    chosen = "ranked"
ui.kpis([
    ("Lenders ranked", f"{len(league):,}", f"{manifest.get('bank_min_loans', 50)}+ loans each"),
    ("Best default rate", ui.pct(best["default_rate"]), best["Bank"].title() if len(best["Bank"]) < 26 else f"{BEST_MIN_LOANS}+ loans"),
    ("Largest book", ui.money(largest["total_disbursed"]), "Disbursed, all years"),
    ("Selected lender", ui.pct(b["default_rate"]), f"All lenders {overall:.1%}"),
])

with st.container(key="two"):
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        PICTURES[chosen]()
        with st.container(key="pick_buttons"):
            cols = st.columns(4)
            for col, key, label in zip(cols, ("ranked", "best", "largest", "selected"),
                                       ("By volume", "Best rate", "Largest book", "Selected")):
                if col.button(label, key=f"btn_figures_{key}", type="primary" if key == chosen else "secondary",
                              use_container_width=True):
                    st.session_state["bank_pick_fig"] = key
                    st.rerun()

    with right:
        with st.container(key="bank_controls"):
            U.html(ui.card_html("Lender", "Search, then choose a lender, or click a bar.", ""))
            c1, c2 = st.columns(2)
            with c1:
                query = st.text_input("Search lenders", key="bank_search", placeholder="For example WELLS")
            names = list(league["Bank"])
            if query:
                names = [n for n in names if query.strip().lower() in n.lower()]
            names = [selected] + [n for n in names if n != selected]
            st.session_state["bank_pick"] = selected
            with c2:
                st.selectbox("Lender", names, key="bank_pick", format_func=lambda n: n.title(),
                             on_change=lambda: st.session_state.update(bank_selected=st.session_state["bank_pick"]))

        ui.card("Selected lender", f"{selected.title()}, rank {int(b['rank_volume'])} by volume.", ui.rows([
            {"a": "Loans", "a_small": f"{int(b['first_year'])} to {int(b['last_year'])}", "b": f"{int(b['loans']):,}"},
            {"a": "Default rate", "badge": "Low" if b["default_rate"] < overall else "High", "b_small": ui.pct(b["default_rate"])},
            {"a": "Mean loan", "a_small": f"Term {b['average_term_years']:.1f} years", "b": ui.money(b["average_loan"])},
        ]), kind="panel")

        with st.container(key="sheet_rows"):
            U.html(ui.card_html("All lenders", "Open the full table, the method or a scored case.", ""))
            if ui.row_button({"a": "See all", "a_small": f"{len(league):,} lenders ranked"}, key="bank_all"):
                st.session_state["bank_sheet"] = "all"
            if ui.row_button({"a": "How it is worked out", "a_small": "Ranking, data and limits"}, key="bank_how"):
                st.session_state["bank_sheet"] = "how"
            if ui.row_button({"a": "Score with this lender", "a_small": "A borderline application"}, key="bank_to_score"):
                case = {**U.BORDERLINE, "Bank": selected, "BankState": b["bank_state"]}
                import analytics as A
                st.session_state.pop("_case_loaded", None)
                st.query_params.clear()
                st.query_params["case"] = A.encode_case(case)
                U.switch_page("pages/score_loan.py")

ui.strip([
    ui.strip_title("Selected lender"),
    ui.strip_value("guaranteed by the SBA", ui.pct(b["guaranteed_share"], 0)),
    ui.strip_value("borrowers in the same state", ui.pct(b["same_state_share"], 0)),
    f'<span class="l">SBA 7(a) loans · 1962 to 2014 · {manifest.get("n_loans", 0):,} cleaned loans</span>',
])


# ---------------------------------------------------------------------------
# The sheets
# ---------------------------------------------------------------------------

@st.dialog("All lenders", width="large")
def sheet_all() -> None:
    table = league[["rank_volume", "Bank", "bank_state", "loans", "default_rate", "average_loan", "guaranteed_share"]].copy()
    table["Bank"] = table["Bank"].str.title()
    table["average_loan"] = table["average_loan"].round(0)
    st.dataframe(table, hide_index=True, use_container_width=True, height=420, key="bank_table",
                 column_config={
                     "rank_volume": st.column_config.NumberColumn("Rank", help="Rank by number of loans.", width="small"),
                     "bank_state": st.column_config.TextColumn("State", width="small"),
                     "loans": st.column_config.NumberColumn("Loans", format="localized"),
                     "default_rate": st.column_config.NumberColumn("Default rate", format="percent"),
                     "average_loan": st.column_config.NumberColumn("Mean loan (USD)", format="localized"),
                     "guaranteed_share": st.column_config.NumberColumn("Guaranteed", format="percent"),
                 })


@st.dialog("How it is worked out")
def sheet_how() -> None:
    for line in [
        "Source: the committed aggregate of all cleaned loans (bank_league.parquet and bank_years.parquet).",
        f"Lenders with at least {manifest.get('bank_min_loans', 'n/a')} loans; {len(league):,} lenders in all, "
        f"combined default rate {overall:.1%}.",
        "Rank is by number of loans. The best default rate counts only lenders with at least "
        f"{BEST_MIN_LOANS} loans, so a small book with no defaults does not top the list.",
        "Default rates are historical and include loans of every age.",
        "Score with this lender opens a borderline application with the lender filled in.",
        "Click a bar to choose that lender; the selected lender is drawn dark.",
    ]:
        U.html(ui.note(line, "accent"))


sheet = st.session_state.pop("bank_sheet", None)
if sheet == "all":
    sheet_all()
elif sheet == "how":
    sheet_how()
