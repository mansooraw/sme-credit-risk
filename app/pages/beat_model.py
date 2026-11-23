"""Beat the model: decide on five real held-out loans, then see the model's call and what actually happened."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

import analytics as A  # noqa: E402
import experience as X  # noqa: E402
import utils as U  # noqa: E402

X.page_start("game")
U.page_header(
    "Beat the model",
    f"Five real loans from the held-out test set, one at a time. You see what the credit desk saw at application; "
    f"approve or decline, then the card turns over to show the model's estimate and whether the loan was repaid. "
    f"Each round holds {A.GAME_DEFAULTS} loans that were charged off.",
    eyebrow="Game",
)

sample = U.sample_portfolio(U.model_version_key())
t = U.threshold()
gain_all, loss_all = U.value_components(sample["DisbursementGross"], sample["Portion"])
sample = sample.assign(gain=gain_all, loss=loss_all)


def new_game(seed: int | None = None) -> None:
    seed = int(time.time() * 1000) % 2_000_000_000 if seed is None else seed
    loans = A.draw_game_loans(sample, seed=seed)
    st.session_state["game"] = {"seed": seed, "ids": list(loans.index), "i": 0, "choices": [], "revealed": False}


if "game" not in st.session_state or any(i not in sample.index for i in st.session_state["game"]["ids"]):
    new_game()
game = st.session_state["game"]
loans = sample.loc[game["ids"]]
model_approves = (loans["p"] < t).to_numpy()
n = len(loans)
done = len(game["choices"])

# Running score
user_right = sum(int(c == (y == 0)) for c, y in zip(game["choices"], loans["Default"].iloc[:done]))
model_right = sum(int(m == (y == 0)) for m, y in zip(model_approves[:done], loans["Default"].iloc[:done]))
steps = "".join(f'<i class="{"done" if k < done else ("now" if k == game["i"] and k >= done else "")}"></i>' for k in range(n))
U.html(f'<div class="cl-scoreboard"><div class="side"><span>You</span><b>{user_right}</b></div>'
       f'<div class="mid">Loan {min(game["i"] + 1, n)} of {n}<div class="cl-progress">{steps}</div></div>'
       f'<div class="side right"><span>Model</span><b>{model_right}</b></div></div>')


def facts(r) -> str:
    items = [
        ("Sector", r["Sector"]), ("State", U.state_name(r["State"])),
        ("Business", "New (2 years or less)" if r["NewExist"] == "New" else "Existing"),
        ("Employees", f"{int(r['NoEmp']):,}"), ("Jobs created or kept", f"{int(r['CreateJob'] + r['RetainedJob']):,}"),
        ("Amount", U.fmt_usd(float(r["DisbursementGross"]))), ("Term", f"{r['TermYears'] * 12:.0f} months"),
        ("Guaranteed", f"{r['Portion']:.0%}"), ("Lender", str(r["Bank"]).title()[:34]),
        ("Lender in same state", "Yes" if r["SameStateBank"] == 1 else "No"),
        ("Revolving line", {"Y": "Yes", "N": "No", "T": "Code T"}.get(r["RevLineCr"], r["RevLineCr"])),
        ("Ran through 2007 to 2009", "Yes" if r["Recession"] == 1 else "No"),
    ]
    return '<div class="cl-facts">' + "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in items) + "</div>"


if game["i"] < n:
    i = game["i"]
    r = loans.iloc[i]
    revealed = game["revealed"]
    head = (f'<div class="cl-game-head"><h4>{U.icon("briefcase", 20)} Application {i + 1}</h4>'
            f'{U.chip(r["Sector"][:28], U.COLOURS["navy"])}</div>')
    body = facts(r)
    if revealed:
        choice = game["choices"][i]
        m_app = bool(model_approves[i])
        y = int(r["Default"])
        model_dec = str(U.decisions([r["p"]], t)[0])
        you_right = choice == (y == 0)
        model_right_now = m_app == (y == 0)
        value = A.game_outcome(choice, y, r["gain"], r["loss"])
        outcome = (f'<b style="color:{U.COLOURS["red"]}">Charged off</b>' if y else
                   f'<b style="color:{U.COLOURS["teal"]}">Repaid in full</b>')
        body += (
            '<div class="cl-reveal-row">'
            f'<div><span>Model PD</span><b>{r["p"]:.1%}</b><div style="margin-top:6px">{U.chip(model_dec, U.DECISION_COLOURS[model_dec], True)}</div></div>'
            f'<div><span>You chose</span><b>{"Approve" if choice else "Decline"}</b>'
            f'<div class="cl-muted" style="font-size:0.82rem;margin-top:4px">Value to the bank {U.fmt_usd(value)}</div></div>'
            f'<div><span>What happened</span>{outcome}</div></div>'
        )
        if you_right and not model_right_now:
            verdict = ("win", "You beat the model on this one.")
        elif you_right:
            verdict = ("win", "Right call, and the model agreed.")
        elif model_right_now:
            verdict = ("lose", "The model got this one and you did not.")
        else:
            verdict = ("lose", "Neither of you got this one.")
        body += f'<div class="cl-verdict {verdict[0]}">{verdict[1]}</div>'
    cls = "flip" if revealed else "deal"
    U.html(f'<div class="cl-game-card {cls}" data-k="{game["seed"]}-{i}-{int(revealed)}"><div class="cl-game-inner">{head}{body}</div></div>')

    with st.container(key="game_buttons"):
        if not revealed:
            b1, b2 = st.columns(2)
            if b1.button("Approve", type="primary", use_container_width=True, key="game_approve", icon=":material/check_circle:"):
                game["choices"].append(True)
                game["revealed"] = True
                st.rerun()
            if b2.button("Decline", use_container_width=True, key="game_decline", icon=":material/block:"):
                game["choices"].append(False)
                game["revealed"] = True
                st.rerun()
        else:
            last = i == n - 1
            if st.button("See the result" if last else "Next loan", type="primary", use_container_width=True,
                         key="game_next", icon=":material/arrow_forward:"):
                game["i"] += 1
                game["revealed"] = False
                st.rerun()
else:
    s = A.game_summary(game["choices"], model_approves, loans["Default"], loans["gain"], loans["loss"])
    won = s["user_value"] > s["model_value"] or (s["user_value"] == s["model_value"] and s["user_correct"] > s["model_correct"])
    title = "You beat the model" if won else ("A draw" if s["user_value"] == s["model_value"] else "The model wins this round")
    seed_chip = U.chip("Round " + str(game["seed"]), U.COLOURS["grey"])
    U.html(f'<div class="cl-game-card flip"><div class="cl-game-inner"><div class="cl-game-head"><h4>{U.icon("check", 20)} {title}</h4>'
           f'{seed_chip}</div>'
           '<p class="cl-muted" style="margin:0 0 6px">Value counts the margin on approved loans that were repaid and the '
           'loss on approved loans that defaulted, with the cost matrix of the model.</p></div></div>')
    U.tiles([
        {"label": "Your accuracy", "value": f"{s['user_accuracy']:.0%}", "sub": f"{s['user_correct']} of {s['n']} right",
         "accent": U.COLOURS["teal"], "help": "A decision is right when an approved loan was repaid or a declined loan was charged off."},
        {"label": "Model accuracy", "value": f"{s['model_accuracy']:.0%}", "sub": f"{s['model_correct']} of {s['n']} right",
         "accent": U.COLOURS["navy"], "help": f"The model approves when the PD is below {t:.1%}; refer and decline count as not approved."},
        {"label": "Your value to the bank", "value": U.fmt_usd(s["user_value"]),
         "sub": f"Model {U.fmt_usd(s['model_value'])}", "accent": U.COLOURS["teal"] if s["user_value"] >= s["model_value"] else U.COLOURS["red"],
         "help": "Sum over the loans you approved: margin if repaid, minus the unguaranteed loss and workout cost if charged off."},
        {"label": "Best possible", "value": U.fmt_usd(s["best_value"]), "sub": f"Approve all {U.fmt_usd(s['approve_all_value'])}",
         "accent": U.COLOURS["amber"], "help": "Approving exactly the loans that were repaid, against approving all five."},
    ])
    diff = s["user_value"] - s["model_value"]
    U.html(f'<div class="cl-insight">{U.icon("coins", 18)}<span>'
           + (f"Your decisions were worth {U.fmt_usd(diff)} more than the model's on these five loans."
              if diff > 0 else f"The model's decisions were worth {U.fmt_usd(-diff)} more than yours on these five loans."
              if diff < 0 else "You and the model made decisions of the same value on these five loans.")
           + "</span></div>")
    if st.button("Play again with five new loans", type="primary", use_container_width=True, key="game_again",
                 icon=":material/replay:"):
        new_game(game["seed"] + 1)
        st.rerun()

st.caption("Loans are drawn from the committed 20,000-loan sample of the held-out test split and scored with the "
           f"loaded model; the model approves below its threshold of {t:.1%}.")
