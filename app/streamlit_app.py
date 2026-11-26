"""CreditLens: SME loan default risk scoring (Streamlit entry point).

Streamlit Community Cloud runs `streamlit run app/streamlit_app.py` from the
repository root. The repository root is added to sys.path so that the app
can import the project modules in src/ (cleaning, features, config).
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
for path in (APP_DIR, APP_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st  # noqa: E402

import experience as X  # noqa: E402
import utils as U  # noqa: E402

st.set_page_config(
    page_title="CreditLens | SME credit risk",
    page_icon=str(U.ICON_PATH),
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, collapsed on a phone
)
U.inject_theme()
X.inject_mode()
st.logo(str(U.LOGO_PATH), size="large", icon_image=str(U.ICON_PATH))

P = st.Page
pages = {
    "Decide": [
        P("pages/home.py", title="Overview", icon=":material/dashboard:", default=True),
        P("pages/score_loan.py", title="Score a loan", icon=":material/speed:", url_path="score"),
        P("pages/compare.py", title="Compare applicants", icon=":material/compare_arrows:", url_path="compare"),
        P("pages/beat_model.py", title="Decision exercise", icon=":material/rule:", url_path="exercise"),
        P("pages/batch_scoring.py", title="Batch scoring", icon=":material/upload_file:", url_path="batch"),
    ],
    "Portfolio": [
        P("pages/portfolio_threshold.py", title="Portfolio and threshold", icon=":material/tune:", url_path="portfolio"),
        P("pages/stress_test.py", title="Stress test", icon=":material/query_stats:", url_path="stress"),
        P("pages/monte_carlo.py", title="Monte Carlo", icon=":material/functions:", url_path="montecarlo"),
    ],
    "Explore": [
        P("pages/market_explorer.py", title="Market explorer", icon=":material/map:", url_path="market"),
        P("pages/bank_league.py", title="Bank league", icon=":material/account_balance:", url_path="banks"),
    ],
    "Model": [
        P("pages/model_lab.py", title="Model lab", icon=":material/science:", url_path="lab"),
        P("pages/model_insights.py", title="Model insights", icon=":material/insights:", url_path="insights"),
        P("pages/model_card.py", title="Model card", icon=":material/description:", url_path="about"),
    ],
}
navigation = st.navigation(pages)

with st.sidebar:
    X.dark_toggle()
    U.sidebar_status()

navigation.run()
