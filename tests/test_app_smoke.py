"""Smoke tests for the CreditLens Streamlit app.

Run from the repository root:

    python -m pytest tests/test_app_smoke.py -q        (or)    python tests/test_app_smoke.py

Each page is run with Streamlit's AppTest, widgets are set, forms are
submitted and the test checks that no exception is raised and that the
results render.
"""

import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
PAGES = APP / "pages"
sys.path.insert(0, str(APP))
TIMEOUT = 180


def run_page(name: str) -> AppTest:
    at = AppTest.from_file(str(PAGES / name), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [t.value for t in at.title]
    parts += [s.value for s in at.subheader] + [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(str(p) for p in parts)


def test_entry_point_and_navigation():
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert "Default risk scoring" in all_text(at)
    at.button(key="hero_score").click()  # hero button navigates with st.switch_page
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert "Probability of default" in all_text(at)
    for page in ["score_loan.py", "compare.py", "beat_model.py", "batch_scoring.py", "portfolio_threshold.py",
                 "stress_test.py", "monte_carlo.py", "market_explorer.py", "bank_league.py", "model_lab.py",
                 "model_insights.py", "model_card.py"]:
        at.switch_page(f"pages/{page}")
        at.run()
        assert not at.exception, (page, [e.message for e in at.exception])


def test_home():
    at = run_page("home.py")
    text = all_text(at)
    for expected in ["ROC-AUC", "Recall on defaults", "How it works", "Why it matters", "cost matrix"]:
        assert expected in text, expected


def test_score_default_loan_is_fast_and_explained():
    at = run_page("score_loan.py")
    at.button[3].click()  # form submit button (after the three example buttons)
    at.run()  # first scoring
    assert not at.exception, [e.message for e in at.exception]
    at.number_input[3].set_value(60)  # term in months, so the second run scores a different loan
    at.button[3].click()
    start = time.perf_counter()
    at.run()  # second scoring, model and explainer cached
    elapsed = time.perf_counter() - start
    assert not at.exception, [e.message for e in at.exception]
    reported = [c.value for c in at.caption if c.value.startswith("Scored and explained in")]
    assert reported, "timing caption missing"
    seconds = float(reported[0].split(" in ")[1].split(" ")[0])
    assert seconds < 3.0, reported[0]
    print(f"single-loan scoring: {reported[0]} (whole page rerun {elapsed:.2f} s)")
    text = all_text(at)
    assert "Probability of default" in text
    assert "What drives this score" in text
    assert "Expected net value" in text
    assert "What if" in text
    assert "SHAP" in text or "contributions" in text


def test_what_if_panel_rescores():
    at = run_page("score_loan.py")  # the borderline example is scored on first load
    assert "PD in this scenario" in all_text(at)
    before = [m for m in all_text(at).splitlines() if "PD in this scenario" in m][0]
    term = [s for s in at.slider if s.label == "Term (months)"][0]
    term.set_value(240)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    after = [m for m in all_text(at).splitlines() if "PD in this scenario" in m][0]
    assert before != after
    assert at.get("download_button")  # decision memo


def test_score_examples_and_validation():
    at = run_page("score_loan.py")
    for i in range(3):
        at.button[i].click()
        at.run()
        assert not at.exception, [e.message for e in at.exception]
        assert "Probability of default" in all_text(at)
    # Guaranteed amount above the approved amount must be rejected
    at = run_page("score_loan.py")
    at.number_input[0].set_value(50_000.0)   # approved
    at.number_input[1].set_value(80_000.0)   # guaranteed
    at.button[3].click()
    at.run()
    assert not at.exception
    assert any("cannot be above the approved amount" in e.value for e in at.error)


def test_score_with_share_and_new_bank():
    at = run_page("score_loan.py")
    at.radio[0].set_value("Guaranteed share")
    at.run()
    bank = [s for s in at.selectbox if s.label == "Lending bank"][0]
    bank.set_value("Other / new bank")
    at.toggle[0].set_value(False)
    at.button[3].click()
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert "Probability of default" in all_text(at)


def test_batch_with_sample_files():
    at = run_page("batch_scoring.py")
    at.toggle[0].set_value(True)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("passed validation" in s.value for s in at.success)
    assert len(at.dataframe) >= 2
    assert "Approval rate" in all_text(at) and "Portfolio summary" in all_text(at)


def test_batch_form_layout_via_utils():
    import pandas as pd
    import utils as U
    from src import cleaning

    form = pd.read_csv(U.SAMPLE_FORM_PATH)
    assert U.detect_layout(form) == "form"
    clean, _ = cleaning.handle_outliers(U.form_to_clean(form), params=U.load_cleaning_params())
    assert (U.validate_clean(clean) == "").all()
    p = U.predict(U.load_model(), U.build_model_frame(clean))
    assert len(p) == len(form) and ((p >= 0) & (p <= 1)).all()
    raw = pd.read_csv(U.SAMPLE_RAW_PATH, keep_default_na=False, na_values=[""])
    p_raw = U.predict(U.load_model(), U.build_model_frame(U.raw_to_clean(raw, U.load_cleaning_params())))
    assert len(p_raw) == len(raw)


def test_portfolio_threshold():
    at = run_page("portfolio_threshold.py")
    assert "Approval rate" in all_text(at) and "Confusion matrix" in all_text(at)
    at.slider[0].set_value(0.40)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_model_insights_and_card():
    at = run_page("model_insights.py")
    assert len(at.tabs) == 4 and [t.label for t in at.tabs] == ["Performance", "Drivers", "Tuning", "Over time"]
    at = run_page("model_card.py")
    text = all_text(at)
    assert "Data provenance and licence" in text and "Add to Home Screen" in text


# ---------------------------------------------------------------------------
# New pages and interactions
# ---------------------------------------------------------------------------

def test_dark_mode_and_tour_from_entry_point():
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=TIMEOUT)
    at.run()
    at.toggle(key="_dark_toggle").set_value(True)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state["dark_mode"] is True
    assert any("--cl-bg: #0A1628" in m.value for m in at.markdown)  # dark variables injected
    at.button(key="hero_tour").click()
    at.run()
    assert at.session_state["tour_step"] == 0
    assert "Step 1 of 5" in all_text(at)
    import experience as X
    # Each step's page shows the highlighted card; Next moves the tour on (st.switch_page is followed by the
    # browser; AppTest runs each page on its own, where the helper falls back to a toast)
    for step, item in enumerate(X.TOUR):
        page = AppTest.from_file(str(APP / item["path"]), default_timeout=TIMEOUT)
        page.session_state["tour_step"] = step
        page.run()
        assert not page.exception, (step, [e.message for e in page.exception])
        assert f"Step {step + 1} of 5" in all_text(page), step
        page.button(key="tour_next").click()
        page.run()
        if step < len(X.TOUR) - 1:
            assert page.session_state["tour_step"] == step + 1
        else:
            assert "tour_step" not in page.session_state
    # The last step pre-fills and scores the borderline example on the Score page
    page = AppTest.from_file(str(APP / X.TOUR[-1]["path"]), default_timeout=TIMEOUT)
    page.session_state["tour_step"] = len(X.TOUR) - 2
    page.session_state["example"] = "borderline"
    page.session_state["auto_score"] = True
    page.run()
    assert "Probability of default" in all_text(page)


def test_home_watch_strip_and_glass_tiles():
    at = run_page("home.py")
    text = all_text(at)
    assert "Watch it score" in text and "cl-watch-card" in text and "cl-tiles glass" in text
    assert text.count("cl-watch-card") == 3
    assert "How it is computed" in text or "cl-how" in text


def test_score_shared_case_link_and_similar_loans():
    import analytics as A
    import utils as U
    case = A.encode_case({**U.EXAMPLES["high"], "Recession": 0})
    at = AppTest.from_file(str(PAGES / "score_loan.py"), default_timeout=TIMEOUT)
    at.query_params["case"] = case
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    text = all_text(at)
    assert "Ten most similar past loans" in text and "How this applicant compares" in text
    assert at.session_state["score_result"]["row"]["Term"] == U.EXAMPLES["high"]["Term"]
    assert at.query_params["case"][0] == case if isinstance(at.query_params["case"], list) else at.query_params["case"] == case
    labels = [b.label for b in at.get("download_button")]
    assert "PDF-ready memo" in labels


def test_compare_applicants_and_copy():
    at = run_page("compare.py")
    text = all_text(at)
    assert "What drives each score" in text and "Recommendation" in text
    base = at.session_state["cmp_base"]
    assert {"PD", "Decision"} <= set(base.columns) and len(base) == 3
    at.selectbox(key="cmp_from").set_value("C")
    at.run()
    at.selectbox(key="cmp_to").set_value("A")
    at.button(key="cmp_copy").click()
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    base = at.session_state["cmp_base"]
    assert abs(base.loc["A", "PD"] - base.loc["C", "PD"]) < 1e-9  # copied applicant rescored to the same PD


def test_game_logic_helpers():
    import numpy as np
    import pandas as pd
    import analytics as A
    df = pd.DataFrame({"Default": [0] * 30 + [1] * 10})
    a, b = A.draw_game_loans(df, seed=7), A.draw_game_loans(df, seed=7)
    assert list(a.index) == list(b.index) and len(a) == A.GAME_ROUND
    assert int(a["Default"].sum()) == A.GAME_DEFAULTS
    assert list(A.draw_game_loans(df, seed=8).index) != list(a.index)
    s = A.game_summary([True, False, True], [True, True, False], [0, 1, 1], [10, 10, 10], [100, 100, 100])
    assert s["user_correct"] == 2 and s["model_correct"] == 2
    assert s["user_value"] == 10 - 100 and s["model_value"] == 10 - 100
    assert s["best_value"] == 10 and s["approve_all_value"] == 10 - 200
    assert A.game_outcome(False, 1, 5, 50) == 0 and A.game_outcome(True, 1, 5, 50) == -50
    assert np.isclose(A.game_outcome(True, 0, 5, 50), 5)


def test_beat_the_model_full_round():
    at = run_page("beat_model.py")
    for i in range(5):
        at.button(key="game_approve" if i % 2 == 0 else "game_decline").click()
        at.run()
        assert not at.exception, [e.message for e in at.exception]
        assert "What happened" in all_text(at)
        at.button(key="game_next").click()
        at.run()
    text = all_text(at)
    assert "Your accuracy" in text and "Model accuracy" in text
    seed = at.session_state["game"]["seed"]
    at.button(key="game_again").click()
    at.run()
    assert at.session_state["game"]["seed"] == seed + 1 and at.session_state["game"]["i"] == 0


def test_batch_filters_and_live_edit():
    at = run_page("batch_scoring.py")
    at.toggle[0].set_value(True)
    at.run()
    assert "Decisions" in all_text(at) or len(at.get("plotly_chart")) >= 3
    out = at.session_state["batch"]["out"]
    label = out.index[0]
    before = float(out.at[label, "probability_of_default"])
    # Simulate an edit of the first row in the editor: term to 300 months
    version = at.session_state["batch"]["version"]
    at.session_state[f"batch_editor_{version}"] = {"edited_rows": {0: {"Term": 300}}, "added_rows": [], "deleted_rows": []}
    at.session_state["batch_view_index"] = list(out.index)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    out = at.session_state["batch"]["out"]
    assert int(out.at[label, "Term"]) == 300 and float(out.at[label, "probability_of_default"]) != before
    at.multiselect(key="batch_dec").set_value(["Approve"])
    at.run()
    assert not at.exception
    assert any(c.value.startswith("Showing") for c in at.caption)
    assert "Download portfolio memo" in [b.label for b in at.get("download_button")]


def test_portfolio_objective_chips_and_shared_threshold():
    at = run_page("portfolio_threshold.py")
    at.button(key="obj_recall_target").click()
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    t = at.session_state["shared_threshold"]
    assert at.slider(key="thr_portfolio").value == round(t / 0.005) * 0.005 or abs(at.slider(key="thr_portfolio").value - t) < 0.006


def test_stress_test_scenario():
    at = run_page("stress_test.py")
    assert "Which segments break first" in all_text(at)
    at.slider(key="st_lgd").set_value(0.8)
    at.slider(key="st_shift").set_value(-10)
    at.toggle(key="st_recession").set_value(False)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert "Under this scenario" in all_text(at)


def test_break_even_lgd():
    import numpy as np
    import analytics as A
    p, d, g = np.array([0.1, 0.1]), np.array([100.0, 100.0]), np.array([0.5, 0.5])
    L = A.break_even_lgd(p, d, g, margin=0.05, workout=0.05)
    gain, loss = A.gain_loss(d, g, 0.05, L, 0.05)
    assert abs(A.expected_net(p, gain, loss).sum()) < 1e-9


def test_monte_carlo_page_and_simulation():
    import numpy as np
    import analytics as A
    p = np.array([0.0, 1.0, 0.5])
    sims = A.monte_carlo(p, gain=np.array([1.0, 1.0, 1.0]), loss=np.array([10.0, 10.0, 10.0]),
                         thresholds=[0.4, 2.0], runs=500, seed=1)
    assert sims.shape == (2, 500)
    assert np.allclose(sims[0], 1.0)  # only the PD 0 loan is approved at 0.4
    assert set(np.unique(sims[1])) <= {1 - 10 + 1, 1 - 10 - 10}  # third loan repaid or not
    assert np.array_equal(sims, A.monte_carlo(p, np.ones(3), np.full(3, 10.0), [0.4, 2.0], runs=500, seed=1))
    at = run_page("monte_carlo.py")
    assert "Chance of a loss-making year" in all_text(at)
    at.slider(key="thr_mc").set_value(0.4)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_market_explorer_filters():
    at = run_page("market_explorer.py")
    assert "Default rate by state, year by year" in all_text(at)
    at.session_state["filters"] = {"state": "FL", "sector": "Retail trade"}
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    text = all_text(at)
    assert "Florida" in text and "Retail trade" in text and "Filtered:" in text
    at.button(key="mx_clear").click()
    at.run()
    assert at.session_state["filters"] == {}


def test_selection_helper_sets_filter_once():
    import utils as U
    at = AppTest.from_string("""
import sys
sys.path.insert(0, %r)
import streamlit as st
import utils as U
event = {"selection": {"points": [{"location": "TX"}]}}
changed = U.apply_selection(event, "map", "state", lookup=lambda pt: pt.get("location"))
st.session_state["changed_" + str(st.session_state.get("n", 0))] = changed
st.session_state["n"] = st.session_state.get("n", 0) + 1
""" % str(APP))
    at.run()
    at.run()
    assert at.session_state["filters"] == {"state": "TX"}
    assert at.session_state["changed_0"] is True and at.session_state["changed_1"] is False  # a stale selection is ignored


def test_bank_league_profile():
    at = run_page("bank_league.py")
    league = __import__("utils").load_aggregate("bank_league.parquet")
    pick = league.iloc[3]["Bank"]
    at.session_state["bank_selected"] = pick
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert pick.title() in all_text(at)
    at.text_input(key="bank_search").input("BANK")
    at.run()
    assert not at.exception


def test_model_lab():
    at = run_page("model_lab.py")
    text = all_text(at)
    assert "Why LightGBM" in text and "Technical reasons" in text and "Business reasons" in text
    at.slider(key="thr_lab").set_value(0.5)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state["shared_threshold"] == 0.5


def test_case_encoding_and_neighbours():
    import numpy as np
    import pandas as pd
    import analytics as A
    import utils as U
    row = {**U.EXAMPLES["low"], "Bank": "A & B BANK, N.A.", "Recession": 1}
    back = A.decode_case(A.encode_case(row))
    assert back["Bank"] == row["Bank"] and back["Term"] == row["Term"] and back["Recession"] is True
    assert A.decode_case("garbage") is None
    ref = pd.DataFrame({"a": np.arange(100.0), "NoEmp": np.arange(100.0)})
    Z, mean, scale = A.prepare_reference(ref, ["a", "NoEmp"])
    idx, _ = A.nearest(Z, mean, scale, pd.DataFrame({"a": [50.2], "NoEmp": [50.0]}), ["a", "NoEmp"], k=3)
    assert idx[0] == 50
    assert A.percentile_rank(np.arange(100), 50) == 50.5
    assert all(len(f.name) for f in U.AGG_DIR.glob("*.parquet"))
    for f in U.AGG_DIR.glob("*.parquet"):
        assert f.stat().st_size < 5e6, f.name


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        start = time.perf_counter()
        try:
            test()
            print(f"PASS {test.__name__} ({time.perf_counter() - start:.1f} s)")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {exc!r}")
    sys.exit(1 if failed else 0)
