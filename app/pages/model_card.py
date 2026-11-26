"""Model card and about: model documentation, team, data provenance, licence and phone install.

The figure strip (model, training date, training loans, threshold), then the chosen section of
docs/model_card.md in a glass card on the left with the pill row of sections under it; on the right the decision
rule in the dark panel and the rows that open the full card, data provenance, the team and phone install.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import ui  # noqa: E402
import utils as U  # noqa: E402

ui.page("Model card", "Choose a section to read that part of the model documentation.", eyebrow="Model",
        heading="The model, documented.")
meta = U.load_metadata()
t, d = U.threshold(), U.decline_threshold()
card_text = U.MODEL_CARD_PATH.read_text(encoding="utf-8") if U.MODEL_CARD_PATH.exists() else None


def sections(text: str) -> dict[str, tuple[str, str]]:
    """Split the model card into its level-two sections: {heading: (heading, body)}."""
    out = {}
    for block in re.split(r"(?m)^## ", text)[1:]:
        head, _, body = block.partition("\n")
        out[head.strip()] = (head.strip(), body.strip())
    return out


SECTIONS = sections(card_text) if card_text else {}
SECTION_FOR = {"model": "Intended use", "trained": "Training data", "threshold": "Threshold", "limits": "Limitations"}


def section(key: str) -> tuple[str, str] | None:
    want = SECTION_FOR[key]
    for head, value in SECTIONS.items():
        if head.startswith(want):
            return value
    return None


name = str(meta.get("model_name", "")) or None
short = None if not name else ("LightGBM" if "lightgbm" in name.lower() else name.split("(")[0].strip())
trained = meta.get("trained_on")
n_train, n_test = meta.get("n_train"), meta.get("n_test")

chosen = st.session_state.get("about_pick", "model")
if chosen not in SECTION_FOR:
    chosen = "model"
ui.kpis([
    ("Model", short or "Not recorded", "Tuned" if name and "tuned" in name.lower() else "Final model"),
    ("Trained", ui.day(trained) if trained else "Not recorded", str(trained)[:4] if trained else "No date recorded"),
    ("Training loans", f"{n_train:,}" if isinstance(n_train, int) else "Not recorded",
     f"Tested on {n_test:,}" if isinstance(n_test, int) else "Test size not recorded"),
    ("Threshold", ui.pct(t, 0), f"Decline from {d:.0%}"),
])

TEAM = [
    ("Mansoor Alwahedi", "Mansoor Mohamed Mansoor Saeed Alwahedi", "Lead, modelling",
     "Team lead: business understanding, modelling and tuning, report integration"),
    ("Salem Almarri", "Salem Adel Salem Alshamlan Almarri", "Data and cleaning",
     "Data collection, data understanding and cleaning, data dictionary"),
    ("Sultan Alkindi", "Sultan Zayed Ahmed Saeed Alkindi", "Analysis and app",
     "Exploratory analysis, feature engineering, Streamlit app and deployment"),
]

with st.container(key="two"):
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        part = section(chosen) if card_text else None
        if part is not None:
            head, body = part
            # the chosen section as a document card; the full card opens in a sheet
            ui.card(head, "From docs/model_card.md", "\n\n" + body + "\n\n")
        else:
            rows = [
                {"a": "Model", "b": name or "Not recorded"},
                {"a": "Trained on", "b": ui.day(trained) if trained else "Not recorded"},
                {"a": "Training loans", "b": f"{n_train:,}" if isinstance(n_train, int) else "Not recorded"},
                {"a": "Test loans", "b": f"{n_test:,}" if isinstance(n_test, int) else "Not recorded"},
                {"a": "Approve below", "b": f"{t:.1%}"},
                {"a": "Decline from", "b": f"{d:.1%}"},
                {"a": "Intended use", "b": "Decision support at application"},
                {"a": "Not intended for", "b": "Automated declines"},
            ]
            ui.card("Model summary", "docs/model_card.md is not available, so this is read from the model metadata.",
                    ui.rows(rows))
        with st.container(key="pick_buttons"):
            cols = st.columns(4)
            for col, key, label in zip(cols, ("model", "trained", "threshold", "limits"),
                                       ("Intended use", "Training data", "Threshold", "Limitations")):
                if col.button(label, key=f"btn_figures_{key}", type="primary" if key == chosen else "secondary",
                              use_container_width=True):
                    st.session_state["about_pick"] = key
                    st.rerun()

    with right:
        ui.card("Decision rule", "Probability of default against the two thresholds.", ui.rows([
            {"a": "Approve", "a_small": f"PD below {t:.0%}", "badge": "Approved"},
            {"a": "Refer", "a_small": f"PD {t:.0%} to {d:.0%}", "badge": "Referred"},
            {"a": "Decline", "a_small": f"PD {d:.0%} and above", "badge": "Declined"},
        ]), kind="panel")

        with st.container(key="sheet_rows"):
            U.html(ui.card_html("About CreditLens", "Open a row for the document, the data or the team.", ""))
            if card_text and ui.row_button({"a": "Read the full card", "a_small": "Every section"}, key="about_full"):
                st.session_state["about_sheet"] = "card"
            if ui.row_button({"a": "Data provenance and licence", "a_small": "Source, citation, repository",
                              "b": "CC BY-SA 4.0"}, key="about_provenance"):
                st.session_state["about_sheet"] = "provenance"
            if ui.row_button({"a": "Team", "a_small": ", ".join(a.split()[0] for a, *_ in TEAM)}, key="about_team"):
                st.session_state["about_sheet"] = "team"
            if ui.row_button({"a": "Install on a phone", "a_small": "Safari: Share, then Add to Home Screen"}, key="about_install"):
                st.session_state["about_sheet"] = "install"

ui.strip([
    ui.strip_title("Data"),
    ui.strip_value("SBA 7(a) loans", "1962 to 2014"),
    ui.strip_value("dataset licence", "CC BY-SA 4.0"),
    '<span class="l">Kaggle, Li, Mickel and Taylor (2018) · CDS4003, Higher Colleges of Technology</span>',
])


# ---------------------------------------------------------------------------
# Sheets: the full model card, data provenance, the team and phone install
# ---------------------------------------------------------------------------

@st.dialog("Model card", width="large")
def card_sheet() -> None:
    st.markdown(card_text)


@st.dialog("Data provenance and licence")
def provenance_sheet() -> None:
    U.html(ui.rows([
        {"a": "Course", "b": "CDS4003, Higher Colleges of Technology"},
        {"a": "Dataset", "b": "Should This Loan be Approved or Denied?", "b_small": "Kaggle, SBAnational.csv"},
        {"a": "Coverage", "b": "SBA 7(a) loans, 1962 to 2014"},
        {"a": "Dataset licence", "b": "CC BY-SA 4.0"},
        {"a": "Paper", "b": "Li, Mickel and Taylor (2018)", "b_small": "J. Statistics Education 26(1), CC BY 4.0"},
        {"a": "Repository", "b": U.REPOSITORY_URL},
    ]))
    U.html(ui.note("CreditLens is a course project and decision-support prototype, not a production credit system.", "warning"))


@st.dialog("Team")
def team_sheet() -> None:
    U.html(ui.rows([{"a": full, "a_small": role} for _, full, _, role in TEAM]))


@st.dialog("Install on a phone")
def install_sheet() -> None:
    U.html(ui.rows([
        {"a": "iPhone, Safari", "a_small": "Share, then Add to Home Screen"},
        {"a": "Android, Chrome", "a_small": "Menu, then Install app"},
        {"a": "From the home screen", "a_small": "Opens like any other app"},
    ]))


SHEETS = {"card": card_sheet, "provenance": provenance_sheet, "team": team_sheet, "install": install_sheet}
sheet = st.session_state.pop("about_sheet", None)
if sheet in SHEETS:
    SHEETS[sheet]()
