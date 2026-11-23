"""Model card and about: model documentation, team, data provenance, licence and phone install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import experience as X  # noqa: E402
import utils as U  # noqa: E402

X.page_start("about")
U.page_header("Model card and about", "How the model was built, what it is for and who built it.", eyebrow="Documentation")
meta = U.load_metadata()

tab_card, tab_about = st.tabs(["Model card", "About"])
with tab_card:
    if U.MODEL_CARD_PATH.exists():
        with st.container(border=True):
            st.markdown(U.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    else:
        U.tiles([
            {"label": "Model", "value": "LightGBM" if "lightgbm" in str(meta.get("model_name", "")).lower() else "Model",
             "sub": meta.get("model_name", "n/a"), "accent": U.COLOURS["navy"]},
            {"label": "Trained on", "value": meta.get("trained_on", "n/a"), "sub": "Training date", "accent": U.COLOURS["teal"]},
            {"label": "Training loans", "value": f"{meta['n_train']:,}" if isinstance(meta.get("n_train"), int) else "n/a",
             "sub": f"Test loans {meta['n_test']:,}" if isinstance(meta.get("n_test"), int) else "", "accent": U.COLOURS["teal"]},
            {"label": "Threshold", "value": f"{U.threshold():.3f}", "sub": f"Decline from {U.decline_threshold():.3f}",
             "accent": U.COLOURS["amber"]},
        ])
        st.caption("docs/model_card.md is not yet available, so this card is built from the model metadata.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            U.html(
                f'<div class="cl-card"><h4>{U.icon("check", 18)} Intended use</h4><p>Decision support for an SME credit officer '
                "assessing guaranteed loans at application time. The officer keeps the final decision and every decline "
                "is reviewed by a person.</p></div>"
                f'<div class="cl-card"><h4>{U.icon("shield", 18, U.COLOURS["red"])} Not intended for</h4><p>Fully automated '
                "declines, consumer lending, or use outside the guaranteed SME segment without retraining and validation on "
                "local data.</p></div>"
            )
        with c2:
            U.html(
                f'<div class="cl-card"><h4>{U.icon("info", 18, U.COLOURS["amber"])} Known limitations</h4><p>The data covers US '
                "loans approved from 1962 to 2014 under changing programme rules; recent years are right-censored; location "
                "and sector can act as proxies for groups that should not be disadvantaged, so error rates should be "
                "monitored by sector and state.</p></div>"
                f'<div class="cl-card"><h4>{U.icon("grid", 18)} Inputs</h4><p>'
                + ", ".join(U.feature_label(f) for f in U.deployment_features()) + ".</p></div>"
            )
        if meta.get("decision_rule"):
            st.caption(f"Decision rule: {meta['decision_rule']}")
        versions = meta.get("library_versions") or {}
        if versions:
            st.caption("Library versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()))

with tab_about:
    c1, c2 = st.columns([1.2, 1], gap="large")
    team = [
        ("Mansoor Mohamed Mansoor Saeed Alwahedi", "Team lead: business understanding, modelling and tuning, report integration"),
        ("Salem Adel Salem Alshamlan Almarri", "Data collection, data understanding and cleaning, data dictionary"),
        ("Sultan Zayed Ahmed Saeed Alkindi", "Exploratory analysis, feature engineering, Streamlit app and deployment"),
    ]
    rows = "".join(
        f'<tr><td style="width:48px"><span class="cl-avatar">{"".join(w[0] for w in n.split()[:1])}{n.split()[-1][0]}</span></td>'
        f'<td><div class="who">{n}</div><div class="role">{r}</div></td></tr>' for n, r in team)
    with c1:
        U.html(f'<div class="cl-card"><h4>{U.icon("users", 18)} Team</h4><table class="cl-team">{rows}</table></div>')
        U.html(
            f'<div class="cl-card"><h4>{U.icon("book", 18)} Course</h4><p>CDS4003 Data Science for Business Application, '
            "Higher Colleges of Technology. The project follows the six CRISP-DM phases; this app is the Deployment phase.</p></div>"
        )
        U.html(
            f'<div class="cl-card"><h4>{U.icon("file", 18)} Data provenance and licence</h4><p>Kaggle dataset "Should This Loan '
            'be Approved or Denied?" (SBAnational.csv), uploaded by Mirbek Toktogaraev, licensed CC BY-SA 4.0. Original source: '
            "US Small Business Administration 7(a) loan records, published with Li, M., Mickel, A. and Taylor, S. (2018) "
            "'Should This Loan be Approved or Denied?: A Large Dataset with Class Assignment Guidelines', Journal of Statistics "
            "Education, 26(1), pp. 55-66, doi:10.1080/10691898.2018.1434342 (CC BY 4.0).</p></div>"
        )
    with c2:
        U.html(
            f'<div class="cl-card"><h4>{U.icon("phone", 18)} Use CreditLens on a phone</h4>'
            "<p style='margin-bottom:12px'>The app adapts to small screens. Add it to the home screen to open it like an app.</p>"
            "<p><b style='color:var(--cl-ink)'>iPhone (Safari)</b></p><ol class='cl-steps-list' style='margin-top:8px'>"
            "<li>Open the app link in Safari.</li><li>Tap the Share button at the bottom of the screen.</li>"
            "<li>Choose Add to Home Screen, then Add.</li></ol>"
            "<p style='margin-top:12px'><b style='color:var(--cl-ink)'>Android (Chrome)</b></p><ol class='cl-steps-list' style='margin-top:8px'>"
            "<li>Open the app link in Chrome.</li><li>Tap the three-dot menu at the top right.</li>"
            "<li>Choose Add to Home screen (or Install app), then confirm.</li></ol></div>"
        )
        U.html(
            f'<div class="cl-card"><h4>{U.icon("grid", 18)} Repository</h4><p>Source code, notebooks and documentation: '
            f'<code>{U.REPOSITORY_URL.replace("<", "&lt;").replace(">", "&gt;")}</code></p></div>'
        )
    st.caption("CreditLens is a course project and decision-support prototype, not a production credit system.")
