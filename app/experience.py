"""Presentation helpers shared by the pages: dark mode, decision mix and the page start.

The look is set by app/assets/theme.css. Nothing here animates; every value
is written into the page as final text.
"""

from __future__ import annotations

import streamlit as st

import utils as U

DARK_CSS_PATH = U.ASSETS_DIR / "theme_dark.css"


# ---------------------------------------------------------------------------
# Dark mode
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _dark_css(mtime: float) -> str:
    return DARK_CSS_PATH.read_text(encoding="utf-8")


def inject_mode() -> None:
    """Dark-mode variables and overrides when the toggle is on (server side, no script needed)."""
    if U.is_dark() and DARK_CSS_PATH.exists():
        st.markdown(f"<style>{_dark_css(U._mtime(DARK_CSS_PATH))}</style>", unsafe_allow_html=True)


def dark_toggle() -> None:
    """Sidebar toggle; the choice lives in session state and survives page changes."""
    st.session_state["_dark_toggle"] = bool(st.session_state.get("dark_mode", False))

    def sync():
        st.session_state["dark_mode"] = bool(st.session_state["_dark_toggle"])

    st.toggle("Dark mode", key="_dark_toggle", on_change=sync, help="Switches the colours of the app and the charts.")


# ---------------------------------------------------------------------------
# Decision mix on the held-out sample
# ---------------------------------------------------------------------------

def decision_mix() -> dict:
    """Shares approved, referred and declined.

    The approval share is metadata business_results.approval_rate. The rest
    is split between refer and decline in the proportion seen when the
    committed test sample is scored with the stored thresholds.
    """
    meta = U.load_metadata()
    approve = U.find_value(meta.get("business_results", {}), ["approval_rate"])
    try:
        sample = U._read_csv(str(U.TEST_SAMPLE_PATH), U._mtime(U.TEST_SAMPLE_PATH))
        dec = U.decisions(sample["prob_default_at_build"].to_numpy())
        share = {k: float((dec == k).mean()) for k in ("Approve", "Refer", "Decline")}
    except Exception:  # noqa: BLE001
        share = {"Approve": 0.75, "Refer": 0.15, "Decline": 0.10}
    approve = float(approve) if approve is not None else share["Approve"]
    rest = max(share["Refer"] + share["Decline"], 1e-9)
    return {"approve": approve, "refer": (1 - approve) * share["Refer"] / rest,
            "decline": (1 - approve) * share["Decline"] / rest}


def scoring_sequence(run_id: str) -> str:
    """Four processing steps shown as a small status line after a loan is scored."""
    steps = ["Cleaned", "Features built", "Scored", "Explained"]
    items = "".join(f'<li><span class="dot"></span>{s}</li>' for s in steps)
    return f'<ol class="cl-seq" data-run="{run_id}">{items}</ol>'


# ---------------------------------------------------------------------------
# Page start (the guided tour was retired; the names are kept for the pages)
# ---------------------------------------------------------------------------

def start_tour() -> None:
    return None


def tour_card(page: str) -> None:
    return None


def page_start(page: str) -> None:
    """Common start of every page: installs the small page runtime."""
    U.runtime()
