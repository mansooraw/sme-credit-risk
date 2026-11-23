"""Home: cinematic hero with the animated loan flow, headline results, how it works and why it matters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import experience as X  # noqa: E402
import utils as U  # noqa: E402

meta = U.load_metadata()
t, d = U.threshold(), U.decline_threshold()
c = U.cost_assumptions()

# Headline results (from the metadata)
metrics = meta.get("metrics_test", {}) or {}
policy = U.model_policy_results(meta)
approval = U.find_value(policy, ["approval_rate"])
loss_red = U.find_value(meta.get("business_results", {}), ["loss_reduction_pct", "loss_reduction", "loss_reduction_percent"])


def pct_value(v):
    if v is None:
        return None
    v = float(v)
    return v * 100 if abs(v) <= 1.5 else v


def tile(label, value, decimals, suffix, sub, accent, help_text):
    if value is None:
        return {"label": label, "value": "n/a", "sub": sub, "accent": accent, "help": help_text}
    return {"label": label, "value": f"{value:,.{decimals}f}{suffix}", "sub": sub, "accent": accent, "help": help_text,
            "count": {"to": round(value, decimals), "decimals": decimals, "suffix": suffix}}


roc = metrics.get("roc_auc")
mix = X.decision_mix()

U.runtime()
with st.container(key="hero"):
    logo = U.LOGO_LIGHT_PATH.read_text(encoding="utf-8")
    left, right = st.columns([1.3, 1], gap="large", vertical_alignment="center")
    with left:
        U.html(
            f'<div class="cl-hero-logo">{logo}</div>'
            '<div class="cl-hero-eyebrow"><span class="pulse"></span>Decision support for guaranteed SME lending</div>'
            '<div class="cl-hero-title">Default risk scoring for the <span>SME credit desk</span></div>'
            '<div class="cl-hero-sub">CreditLens estimates the chance that a government-guaranteed SME loan is charged off, '
            "recommends approve, refer or decline, and explains the score in plain language. "
            "The credit officer keeps the final decision.</div>"
        )
        if U.artefact_source()["is_dev"]:
            U.html('<div class="cl-devnote" style="margin-top:8px">Interim development model in use</div>')
        b1, b2, b3 = st.columns(3)
        if b1.button("Score a loan", type="primary", use_container_width=True, key="hero_score", icon=":material/speed:"):
            U.switch_page("pages/score_loan.py")
        if b2.button("Portfolio", use_container_width=True, key="hero_portfolio", icon=":material/tune:"):
            U.switch_page("pages/portfolio_threshold.py")
        if b3.button("Take the tour", use_container_width=True, key="hero_tour", icon=":material/explore:"):
            X.start_tour()
            st.rerun()
        U.html(
            '<div class="cl-hero-legend">'
            f'<span><i style="background:{U.COLOURS["teal"]}"></i>Approved <b>{mix["approve"]:.1%}</b></span>'
            f'<span><i style="background:{U.COLOURS["amber"]}"></i>Referred <b>{mix["refer"]:.1%}</b></span>'
            f'<span><i style="background:{U.COLOURS["red"]}"></i>Declined <b>{mix["decline"]:.1%}</b></span></div>'
        )
    with right:
        U.html('<div class="cl-tile-label" style="color:#A9BDD4;margin-top:8px">Watch it score</div>'
               + X.watch_strip(U.score_examples(U.model_version_key())))
    U.tiles([
        tile("ROC-AUC", float(roc) if roc is not None else None, 3, "", "Ranking quality on held-out loans",
             U.COLOURS["navy"], "Area under the ROC curve on the test split: the chance that a randomly chosen "
             "defaulted loan scores higher than a randomly chosen repaid one. From metadata metrics_test."),
        tile("Recall on defaults", pct_value(metrics.get("recall")), 1, "%", "Defaults not approved at the threshold",
             U.COLOURS["teal"], "Defaulted test loans with PD at or above the approval threshold, divided by all "
             "defaulted test loans. From metadata metrics_test."),
        tile("Approval rate", pct_value(approval), 1, "%", f"Business minimum {c['MIN_APPROVAL_RATE']:.0%}",
             U.COLOURS["teal"], "Test loans with PD below the approval threshold, divided by all test loans. "
             "From metadata business_results."),
        tile("Charge-off loss avoided", pct_value(loss_red), 1, "%", "Compared with approving every loan",
             U.COLOURS["amber"], "1 minus the charge-off loss of the approved loans divided by the charge-off loss "
             "if every test loan were approved. From metadata business_results."),
    ], glass=True)
X.hero_canvas(mix)
X.tour_card("home")

n_test = meta.get("n_test")
if isinstance(n_test, int):
    st.caption(f"Measured on {n_test:,} held-out test loans the model never saw in training. Source: model metadata. "
               "The particles in the banner pass through a scoring gate in the same approve, refer and decline proportions.")

# How it works
U.section_title("How it works", "spark")
U.html(
    '<div class="cl-steps">'
    f'<div class="cl-step"><div class="cl-step-icon">{U.icon("file", 24)}</div><div class="cl-step-num">01</div>'
    "<h4>Application</h4><p>The officer enters the loan as received: amount, guarantee, term, business profile and lender, "
    "or uploads a file of applications.</p></div>"
    f'<div class="cl-step"><div class="cl-step-icon">{U.icon("gauge", 24)}</div><div class="cl-step-num">02</div>'
    "<h4>Score and explain</h4><p>The inputs are cleaned and engineered exactly as in training, scored by the model and "
    "broken down into the factors that raise or lower the risk.</p></div>"
    f'<div class="cl-step"><div class="cl-step-icon">{U.icon("check", 24)}</div><div class="cl-step-num">03</div>'
    "<h4>Decide</h4><p>CreditLens recommends approve, refer or decline and prices the decision in USD with the bank's cost "
    "assumptions. The officer decides.</p></div></div>"
)

# Risk bands and decisions
U.section_title("Reading a score", "gauge")
bands = "".join(U.chip(f"{b['name']} <span class='cl-chip-mono'>{b['min']:.0%} to {b['max']:.0%}</span>", b["colour"])
                for b in U.risk_bands())
U.html(f"<div>{bands}</div>")
st.caption("The probability of default (PD) is the estimated chance that the loan is charged off rather than repaid in full.")
U.html(
    '<div class="cl-decisions">'
    f'<div class="cl-dcard"><span class="cl-dcard-bar" style="background:{U.DECISION_COLOURS["Approve"]}"></span>{U.chip("Approve", U.DECISION_COLOURS["Approve"], True)}'
    f'<div class="rule">PD &lt; {t:.1%}</div><p>Below the threshold that maximises portfolio net value at an approval rate '
    f'of at least {c["MIN_APPROVAL_RATE"]:.0%}.</p></div>'
    f'<div class="cl-dcard"><span class="cl-dcard-bar" style="background:{U.DECISION_COLOURS["Refer"]}"></span>{U.chip("Refer", U.DECISION_COLOURS["Refer"], True)}'
    f'<div class="rule">{t:.1%} to {d:.1%}</div><p>Sent to a credit officer for manual review with the explanation attached.</p></div>'
    f'<div class="cl-dcard"><span class="cl-dcard-bar" style="background:{U.DECISION_COLOURS["Decline"]}"></span>{U.chip("Decline", U.DECISION_COLOURS["Decline"], True)}'
    f'<div class="rule">PD &ge; {d:.1%}</div><p>The expected loss outweighs the margin; the applicant receives the main reasons.</p></div>'
    "</div>"
)

# Why it matters
U.section_title("Why it matters", "scale")
left, right = st.columns([1.1, 1], gap="large")
with left:
    U.html(
        '<div class="cl-card"><h4>The cost matrix</h4>'
        '<table class="cl-matrix"><tr><th>Decision</th><th>Loan repaid</th><th>Loan defaults</th></tr>'
        f'<tr><td><b>Approve</b></td><td class="pos">+{c["NET_INTEREST_MARGIN"]:.0%} of disbursed (margin)</td>'
        f'<td class="neg">-{c["LOSS_GIVEN_DEFAULT"]:.0%} of unguaranteed principal, -{c["WORKOUT_COST_RATE"]:.0%} of disbursed</td></tr>'
        "<tr><td><b>Decline</b></td><td>0</td><td>0</td></tr></table>"
        '<p style="margin-top:12px">Constants from src/config.py, the same values used to set the threshold.</p></div>'
    )
with right:
    schema = U.load_schema()
    n_train, rate = schema.get("n_training_loans"), schema.get("training_default_rate")
    train_text = (f" The training split holds {n_train:,} loans with a default rate of {rate:.1%}." if n_train and rate else "")
    U.html(
        '<div class="cl-card"><h4>Data and setting</h4><p>Trained on US Small Business Administration 7(a) loan records '
        "approved between 1962 and 2014 (Li, Mickel and Taylor, 2018; Kaggle, CC BY-SA 4.0)." + train_text +
        " A guaranteed loan that defaults costs the bank the unguaranteed principal plus workout cost, while a wrongly "
        "declined borrower costs the margin and the jobs the loan would have supported. The Emirates Development Bank "
        "credit guarantee scheme works the same way, so a UAE bank would retrain the model on its own portfolio.</p></div>"
    )
