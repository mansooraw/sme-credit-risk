"""Component kit: every part of the design system on one page. Not in the navigation; run it with
`streamlit run app/pages/kit.py` from the repository root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
import utils as U  # noqa: E402

ui.page("Component kit", "Every part, in its states.", eyebrow="Design system", heading="The parts, in one place.")
ui.kpis([("Figure strip", "0.981", "meta line"), ("Share", ui.pct(0.792), None), ("Money", ui.money(1_250_000), None), ("Not recorded", "Not recorded", "words, never 0")])
ui.strip([ui.strip_title("Strip"), ui.steps(5, 2, 3), ui.strip_value("Decided", "2 of 5"), ui.strip_value("Officer net value", ui.money(12_500))])

with st.container(key="two"):
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        with st.container(key="picture_fixed"):
            fig = go.Figure(go.Scatter(x=[0, 1, 2, 3, 4], y=[1, 3, 2, 4, 3], mode="lines", line=dict(color=ui.CHART["accent"], width=2.5)))
            ui.chart_layout(fig, "The chart is the card")
            U.chart(fig, height=300, key="kit_chart")
        U.html(ui.card_html("Facts", "A grid of labelled facts.", ui.facts([("Term", ui.months(56)), ("Guaranteed", "50%"), ("Employees", "1"), ("Jobs created or kept", "2"),
                                                                            ("Lender", "Bank of America"), ("Lender in same state", "No"), ("Revolving line", "No"), ("Ran through 2007 to 2009", "No")])))
        with st.container(key="kit_controls"):
            U.html(ui.card_html("Controls card", "Widgets inside a glass card.", ""))
            c1, c2, c3 = st.columns(3)
            c1.button("Primary", type="primary", key="kit_primary")
            c2.button("Secondary", key="kit_secondary")
            c3.button("Disabled", disabled=True, key="kit_disabled")
            st.slider("Slider", 0.0, 1.0, 0.22, key="kit_slider")
            st.selectbox("Select", ["One", "Two"], key="kit_select")
            st.tabs(["Tab one", "Tab two"])
    with right:
        ui.card("Dark panel", "The key result sits here.", ui.rows([
            {"a": "Approved", "a_small": "PD below 22%", "b": "79.2%"}, {"a": "Referred", "a_small": "PD 22% to 50%", "b": "4.1%"},
            {"a": "Declined", "a_small": "PD 50% and above", "b": "16.7%"}]) + ui.stack([("good", 0.792, "79.2%"), ("warning", 0.041, "4.1%"), ("bad", 0.167, "16.7%")]), kind="panel")
        ui.card("Badges and notes", "One status map, four tones.", "".join(
            f'<div class="ui-row"><div class="a">{w}</div><div class="b">{ui.badge(w)}</div></div>' for w in ["Approved", "Referred", "Declined", "None"])
            + ui.note("Accent note.") + ui.note("Good note.", "good") + ui.note("Warning note.", "warning") + ui.note("Bad note.", "bad"))
        ui.card("Tinted card and pair", None, ui.pair([("If repaid", "+5%", "of disbursed"), ("If it defaults", "-50%", "of unguaranteed")]), kind="tint")
        with st.container(key="sheet_rows"):
            U.html(ui.card_html("Rows that open sheets", "Press one.", ""))
            if ui.row_button({"a": "Borderline", "a_small": "USD 100 k · 48 months", "badge": "Referred", "b_small": "PD 39.4%"}, key="kit_one"):
                st.toast("Pressed")
            if ui.row_button({"a": "Low risk", "a_small": "USD 500 k · 240 months", "badge": "Approved", "b_small": "PD 0.1%"}, key="kit_two"):
                st.toast("Pressed")
ui.strip([ui.strip_title("Bottom strip"), ui.strip_value("If repaid", "+5%"), ui.strip_value("If it defaults", "-50%")])
