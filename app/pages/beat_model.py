"""Decision exercise: decide five real held-out loans, then compare with the model and the recorded outcome.

Composition (from design-system/concepts/decision_exercise.webp): the title block, a pill strip with the
01 to 05 stepper and the three running figures, then two columns. Left: the application card (sector and
state, requested amount, the fact grid, Approve and Decline, and after a choice what happened). Right: the
dark emerald panel (which application this is, then the running tally per loan, and the round summary at
the end). Along the bottom: the method sheet row and the data note.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import analytics as A  # noqa: E402
import ui  # noqa: E402
import utils as U  # noqa: E402

ui.page("Decision exercise", "Decide five held-out loans, then compare with the model and recorded outcome.",
        eyebrow="Decision exercise", heading="Your judgment. Then the evidence.")

sample = U.sample_portfolio(U.model_version_key())
t = U.threshold()
gain_all, loss_all = U.value_components(sample["DisbursementGross"], sample["Portion"])
sample = sample.assign(gain=gain_all, loss=loss_all)
PAST = {"Approve": "Approved", "Refer": "Referred", "Decline": "Declined"}


def new_game(seed: int | None = None) -> None:
    seed = int(time.time() * 1000) % 2_000_000_000 if seed is None else seed
    loans = A.draw_game_loans(sample, seed=seed)
    st.session_state["game"] = {"seed": seed, "ids": list(loans.index), "i": 0, "choices": [], "revealed": False}
    st.session_state.pop("game_pick", None)


if "game" not in st.session_state or any(i not in sample.index for i in st.session_state["game"]["ids"]):
    new_game()
game = st.session_state["game"]
loans = sample.loc[game["ids"]]
model_approves = (loans["p"] < t).to_numpy()
n = len(loans)
done = len(game["choices"])
over = game["i"] >= n
awaiting = not over and not game["revealed"]

ys = loans["Default"].to_numpy()
officer_each = [A.game_outcome(c, ys[k], loans["gain"].iloc[k], loans["loss"].iloc[k]) for k, c in enumerate(game["choices"])]
model_each = [A.game_outcome(bool(model_approves[k]), ys[k], loans["gain"].iloc[k], loans["loss"].iloc[k]) for k in range(done)]
officer_value, model_value = sum(officer_each), sum(model_each)
agreed = [bool(game["choices"][k]) == bool(model_approves[k]) for k in range(done)]


def model_word(k: int) -> str:
    return PAST[str(U.decisions([loans["p"].iloc[k]], t)[0])]


def officer_word(k: int) -> str:
    return "Approved" if game["choices"][k] else "Declined"


def outcome_word(k: int) -> str:
    return "Charged off" if int(ys[k]) else "Repaid"


def decide(approve: bool) -> None:
    game["choices"].append(approve)
    game["revealed"] = True
    st.rerun()


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def application_facts(r) -> list[tuple[str, str]]:
    return [
        ("Term", ui.months(round(r["TermYears"] * 12))),
        ("Guaranteed", f"{r['Portion']:.0%}"),
        ("Employees", f"{int(r['NoEmp']):,}"),
        ("Jobs created or kept", f"{int(r['CreateJob'] + r['RetainedJob']):,}"),
        ("Lender", str(r["Bank"]).title()[:30].strip()),
        ("Lender in same state", "Yes" if r["SameStateBank"] == 1 else "No"),
        ("Revolving line", {"Y": "Yes", "N": "No", "T": "Code T"}.get(r["RevLineCr"], str(r["RevLineCr"]))),
        ("Ran through 2007 to 2009", "Yes" if r["Recession"] == 1 else "No"),
    ]


def application_card(i: int) -> None:
    """The glass application card: header, facts, the decision buttons or what happened."""
    r = loans.iloc[i]
    business = "New business · 2 years or less" if r["NewExist"] == "New" else "Existing business · more than 2 years"
    with st.container(key="game_controls"):
        head, amount = st.columns([1.6, 1], gap="small")
        with head:
            U.html(ui.card_html(f"{r['Sector']} / {U.state_name(r['State'])}", business, ""))
        with amount:
            U.html(ui.pair([("Requested amount", f"USD {ui.num(float(r['DisbursementGross']))}", None)]))
        U.html(ui.facts(application_facts(r), columns=4))
        if awaiting:
            b1, b2 = st.columns(2)
            if b1.button("Approve", type="primary", use_container_width=True, key="game_approve"):
                decide(True)
            if b2.button("Decline", use_container_width=True, key="game_decline"):
                decide(False)
            U.html(ui.note("Model decision and recorded outcome appear after your choice.", "accent"))
        else:
            U.html(ui.eyebrow("What happened") + ui.rows([
                {"a": "Model PD", "a_small": f"Model approves below {t:.0%}", "b": ui.pct(loans["p"].iloc[i])},
                {"a": "Model decision", "badge": model_word(i)},
                {"a": "Officer decision", "badge": officer_word(i)},
                {"a": "Outcome", "badge": outcome_word(i)},
                {"a": "Value to the bank", "b": ui.money(officer_each[i]), "b_small": "Your decision"},
            ]))
            if st.button("Summary" if i == n - 1 else "Next loan", type="primary", use_container_width=True, key="game_next"):
                game["i"] += 1
                game["revealed"] = False
                st.rerun()


def round_card() -> None:
    """At the end of the round the left card lists the five loans, officer against model."""
    items = []
    for k in range(n):
        r = loans.iloc[k]
        items.append({"a": f"Loan {k + 1}: {str(r['Sector'])[:24].strip()}",
                      "a_small": f"Officer {officer_word(k).lower()}, model {model_word(k).lower()}",
                      "badge": outcome_word(k), "b_small": "Agreed" if agreed[k] else "Differed"})
    ui.card("Officer and model by loan", f"You and the model agreed on {sum(agreed)} of {n} loans.", ui.rows(items))


def tally_chart() -> None:
    """Officer against model value per loan, drawn small on the dark panel."""
    names = [f"Loan {k + 1}" for k in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=officer_each + [None] * (n - done), name="Officer", marker_color="#ffffff",
                         hovertemplate="%{x}<br>Officer value USD %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=names, y=model_each + [None] * (n - done), name="Model", marker_color=ui.CHART["accent_light"],
                         hovertemplate="%{x}<br>Model value USD %{y:,.0f}<extra></extra>"))
    ui.chart_layout(fig)
    fig.add_hline(y=0, line_color="#ffffff", line_width=1)
    fig.update_layout(
        barmode="group", bargap=0.35, height=230, margin=dict(l=8, r=8, t=30, b=8),
        font=dict(color="#ffffff"), legend=dict(font=dict(color="#ffffff"), y=1.12),
        xaxis=dict(categoryorder="array", categoryarray=names, showgrid=False, linecolor=ui.CHART["accent_light"],
                   tickfont=dict(color="#ffffff")),
        yaxis=dict(tickformat="~s", showgrid=False, zeroline=False, tickfont=dict(color="#ffffff"), title=""),
    )
    # drawn straight with Plotly: the house chart helper resets the text to ink, which is unreadable on the panel
    st.plotly_chart(fig, use_container_width=True, theme=None, key="game_tally",
                    config={"displayModeBar": False, "responsive": True})


def measures_rows() -> str:
    return ui.rows([
        {"a": "Your accuracy", "a_small": "Approved loans repaid, declined loans charged off"},
        {"a": "Model accuracy", "a_small": f"The model approves below {t:.0%}"},
        {"a": "Net value to the bank", "a_small": "Margin if repaid, loss and workout cost if charged off"},
    ])


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

ui.strip([
    ui.steps(n, done, None if over else game["i"] + 1),
    ui.strip_value("Decided", f"{done} of {n}"),
    ui.strip_value("Officer net value", ui.money(officer_value)),
    ui.strip_value("Model net value", ui.money(model_value)),
])

with st.container(key="two"):
    left, right = st.columns([1.1, 1], gap="medium")

    with left:
        if over:
            round_card()
        else:
            application_card(game["i"])

    with right:
        with st.container(key="game_panel"):
            if over:
                s = A.game_summary(game["choices"], model_approves, loans["Default"], loans["gain"], loans["loss"])
                U.html(ui.card_html("Round summary", "Your accuracy against the model on this round.", ui.rows([
                    {"a": "Officer accuracy", "b": ui.pct(s["user_accuracy"], 0), "b_small": f"{s['user_correct']} of {s['n']}"},
                    {"a": "Model accuracy", "b": ui.pct(s["model_accuracy"], 0), "b_small": f"{s['model_correct']} of {s['n']}"},
                    {"a": "Net value", "a_small": f"Model {ui.money(s['model_value'])}", "b": ui.money(s["user_value"]),
                     "b_small": "Officer"},
                    {"a": "Approve all", "a_small": f"Best possible {ui.money(s['best_value'])}",
                     "b": ui.money(s["approve_all_value"])},
                ]), kind="panel"))
                tally_chart()
                if st.button("New round", use_container_width=True, key="game_again"):
                    new_game(game["seed"] + 1)
                    st.rerun()
            else:
                i = game["i"]
                line = "Review the facts before deciding." if awaiting else "See what happened, then go to the next loan."
                body = (ui.eyebrow("Value to the bank per loan") if done else
                        ui.eyebrow("What the exercise measures") + measures_rows())
                U.html(ui.card_html(f"Application {i + 1} of {n}", line, body, kind="panel"))
                if done:
                    tally_chart()

with st.container(key="game_foot"):
    f1, f2 = st.columns([1.1, 1], gap="medium")
    with f1:
        with st.container(key="sheet_rows"):
            if ui.row_button({"a": "Method and data", "a_small": "Loans, values and accuracy"}, key="game_method"):
                st.session_state["game_sheet"] = "method"
    with f2:
        ui.strip([f'<span class="l">{A.GAME_DEFAULTS} of the {n} loans were charged off</span>'])


# ---------------------------------------------------------------------------
# The sheet: method and data
# ---------------------------------------------------------------------------

@st.dialog("Method and data")
def method_sheet() -> None:
    for text in [
        f"Loans are drawn from the committed 20,000-loan sample of the held-out test split; each round holds "
        f"{A.GAME_DEFAULTS} loans that were charged off.",
        "The facts shown are those available when the loan was made.",
        f"The model approves when its probability of default is below {t:.1%}; refer and decline count as not approved.",
        "A decision is correct when an approved loan was repaid or a declined loan was charged off.",
        "Net value counts the margin on approved loans that were repaid and the unguaranteed loss plus workout cost "
        "on approved loans that were charged off, using the cost matrix of the model.",
    ]:
        U.html(ui.note(text, "accent"))
    U.html(ui.rows([{"a": "Round", "b": f"{game['seed']}"}]))


if st.session_state.get("game_sheet"):
    st.session_state.pop("game_sheet")
    method_sheet()
