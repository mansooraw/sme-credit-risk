"""Score a loan, built from the client's concept: the application form on the left (example buttons, the
guarantee mode, the form in three groups), the dark result panel on the right (PD, decision, value), then the
drivers of the score and the actions. The what-if, the comparison and the similar past loans open in one sheet."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import analytics as A  # noqa: E402
import ui  # noqa: E402
import utils as U  # noqa: E402

# Model and SHAP explainer are loaded once per session, before the first score
model = U.load_model()
try:
    U.warm_explainer()
except Exception:  # noqa: BLE001
    pass
ui.page("Score a loan", "Every application, examined clearly.", eyebrow="Decide", heading="Score a loan")

fields = U.schema_fields()
raw_f, feat_f = fields["raw"], fields["features"]


def rng(name: str, key: str, fallback):
    f = raw_f.get(name) or feat_f.get(name) or {}
    value = f.get(key)
    return float(value) if value is not None else fallback


def help_text(name: str) -> str:
    f = raw_f.get(name) or feat_f.get(name) or {}
    return f.get("help", "")


score_rows = U.score_form_rows
WORDS = {"Approve": "Approved", "Refer": "Referred", "Decline": "Declined"}

EXAMPLES = dict(U.EXAMPLES)
EXAMPLE_LABELS = U.EXAMPLE_LABELS

# A case shared by link (?case=...) is loaded as its own example and scored on arrival
shared = A.decode_case(st.query_params.get("case"))
if shared is not None:
    EXAMPLES["shared"] = {**U.BORDERLINE, **shared}
    if st.session_state.get("_case_loaded") != st.query_params.get("case"):
        st.session_state["example"] = "shared"
        st.session_state["_case_loaded"] = st.query_params.get("case")
        st.session_state["auto_score"] = True
query_example = st.query_params.get("example")
if query_example in EXAMPLES and st.session_state.get("_example_loaded") != query_example:
    st.session_state["example"] = query_example
    st.session_state["_example_loaded"] = query_example
    st.session_state["auto_score"] = True
if "score_result" not in st.session_state and "_first_load_done" not in st.session_state:
    st.session_state["auto_score"] = True  # show the borderline example on first visit
    st.session_state["_first_load_done"] = True

sectors = (feat_f.get("Sector") or {}).get("allowed") or sorted(set(U.SECTOR_TO_NAICS))
sectors = sorted(s for s in sectors if s != "Unknown") + ["Unknown"]
for _ex in list(EXAMPLES.values()):
    if _ex.get("Sector") not in sectors:
        EXAMPLES = {k: v for k, v in EXAMPLES.items() if v is not _ex}
states = sorted(s for s in (feat_f.get("State") or {}).get("allowed", []) if s != "Unknown")
bank_states = sorted(s for s in (feat_f.get("BankState") or {}).get("allowed", []) if s != "Unknown")
banks = (feat_f.get("Bank") or {}).get("allowed", [])
bank_options = [U.NEW_BANK_OPTION] + sorted(b for b in banks if b != "Unknown")
for _k, _ex in list(EXAMPLES.items()):
    if _ex.get("Bank") not in bank_options:
        EXAMPLES[_k] = {**_ex, "Bank": U.NEW_BANK_OPTION}
    if _ex.get("State") not in states or _ex.get("BankState") not in bank_states:
        EXAMPLES.pop(_k, None)
    for _f, _opts in (("NewExist", ["Existing", "New"]), ("UrbanRural", ["Urban", "Rural"]),
                      ("RevLineCr", ["N", "Y"]), ("LowDoc", ["N", "Y"]), ("IsFranchise", ["N", "Y"])):
        if _k in EXAMPLES and EXAMPLES[_k].get(_f) not in _opts:
            EXAMPLES.pop(_k, None)
yes_no = U.YES_NO

t, d = U.threshold(), U.decline_threshold()
result = st.session_state.get("score_result")


def pressed(key: str) -> bool:
    """Whether a button was pressed for this run, read before it is drawn so it is drawn in its new state."""
    try:
        return bool(st.session_state.get(key))
    except Exception:  # noqa: BLE001
        return False


# An example button is read first, so the chosen one is drawn as the primary pill in this run
for _k in ["borderline", "low", "high"]:
    if pressed(f"btn_{_k}"):
        st.session_state["example"] = _k
        st.session_state["auto_score"] = True
if pressed("btn_row_edit"):  # Edit application: the form is always open on the left; this only marks the intent
    st.session_state["score_edit"] = True


def form_values(key: str) -> dict:
    """The values the form opens with: the example, or the last scored application of that example."""
    ex = dict(EXAMPLES[key])
    row = (result or {}).get("row")
    if row and (result or {}).get("example") == key:
        ex.update(row)
        ex["Portion"] = float(row["SBA_Appv"]) / float(row["GrAppv"]) if row["GrAppv"] else ex.get("Portion", 0.75)
        ex["Recession"] = bool(row["Recession"])
    return ex


def validate(v: dict) -> tuple[list[str], list[str], float]:
    errors, warnings = [], []
    gr, sba, share, disb, term = v["gr"], v["sba"], v["share"], v["disb"], v["term"]
    sba_value = sba if sba is not None else (share or 0) * gr
    if share is not None and not 0 < share <= 1:
        errors.append("The guaranteed share must be above 0 and at most 1.")
    if sba is not None and sba > gr:
        errors.append("The guaranteed amount cannot be above the approved amount.")
    if sba is not None and sba <= 0:
        errors.append("The guaranteed amount must be above zero.")
    if not 1 <= term <= 480:
        errors.append("The term must be between 1 and 480 months.")
    portion = sba_value / gr if gr else 0
    lo, hi = rng("Portion", "min", 0.05), rng("Portion", "max", 1.0)
    if not errors and not lo <= portion <= hi:
        warnings.append(f"Guaranteed share {portion:.0%} is outside the training range ({lo:.0%} to {hi:.0%}).")
    ratio = disb / gr if gr else 0
    p01, p99 = rng("DisbursedToApprovedRatio", "p01", 0.7), rng("DisbursedToApprovedRatio", "p99", 3.9)
    if not p01 <= ratio <= p99:
        warnings.append(f"Disbursed is {ratio:.2f} times approved; 98% of training loans lie between {p01:.2f} and {p99:.2f}.")
    for name, value in [("NoEmp", v["noemp"]), ("Term", term), ("GrAppv", gr)]:
        lo_, hi_ = rng(name, "p01", None), rng(name, "p99", None)
        if lo_ is not None and hi_ is not None and not lo_ <= value <= hi_:
            label = (raw_f.get(name) or feat_f.get(name) or {}).get("label", name)
            warnings.append(f"{label} of {value:,.0f} is outside the range of 98% of training loans.")
    return errors, warnings, sba_value


def download_row(label: str, key: str, data: bytes, file_name: str) -> None:
    """A row of the Actions card whose press downloads a file: the kit's row with the download button over it."""
    with st.container(key=f"row_{key}"):
        U.html(f'<div class="ui-row"><div class="a">{ui.esc(label)}</div><div class="b"></div>'
               '<span class="arrow">&#8250;</span></div>')
        st.download_button(label, data, file_name=file_name, mime="text/html", key=key, use_container_width=True)




def sheet_row(item: dict, key: str) -> bool:
    """A row that opens a sheet, drawn as the kit's row, with its own button key (kept for the tests)."""
    a_small = f'<small>{ui.esc(item["a_small"])}</small>' if item.get("a_small") else ""
    b_small = f'<small>{ui.esc(item["b_small"])}</small>' if item.get("b_small") else ""
    with st.container(key=f"row_{key}"):
        U.html(f'<div class="ui-row"><div class="a">{ui.esc(item["a"])}{a_small}</div>'
               f'<div class="b">{ui.esc(item.get("b", ""))}{b_small}</div><span class="arrow">&#8250;</span></div>')
        return st.button(f"Open {item['a']}", key=key, use_container_width=True)


def whatif_moved() -> bool:
    """True when a what-if slider changed in a full rerun, so the sheet that holds it is opened again."""
    snap = st.session_state.get("_wi_snap") or {}
    try:
        return any(k in st.session_state and st.session_state[k] != v for k, v in snap.items())
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Pictures for the sheet
# ---------------------------------------------------------------------------

def what_if(base: dict, rid: str, p_base: float) -> None:
    """Rescore this applicant as three terms change; inside the sheet only the sheet reruns."""
    with st.container(key="whatif"):
        line = st.container()
        gr0 = float(base["GrAppv"])
        portion0 = float(np.clip(base["SBA_Appv"] / gr0, 0.05, 1.0))
        dmax = float(min(rng("DisbursementGross", "max", 1.2e7), max(4 * base["DisbursementGross"], 500_000)))
        s1, s2, s3 = st.columns(3)
        w_term = s1.slider("Term (months)", 1, 480, int(base["Term"]), key=f"wi_term_{rid}")
        w_share = s2.slider("Guaranteed share", 0.05, 1.0, round(portion0, 2), 0.01, key=f"wi_share_{rid}")
        w_disb = s3.slider("Amount disbursed (USD)", 5_000, int(dmax), int(round(base["DisbursementGross"], -3)),
                           5_000, key=f"wi_disb_{rid}")
        st.session_state["_wi_snap"] = {f"wi_term_{rid}": w_term, f"wi_share_{rid}": w_share, f"wi_disb_{rid}": w_disb}
        grid = sorted(set(list(range(12, 481, 12)) + [w_term]))
        rows = pd.DataFrame([{**base, "Term": m, "SBA_Appv": w_share * gr0, "DisbursementGross": float(w_disb)}
                             for m in grid])
        _, p_grid = score_rows(rows)
        p_new = float(p_grid[grid.index(w_term)])
        dec_new = WORDS[str(U.decisions([p_new], t, d)[0])]
        ev_new, el_new = U.expected_value([p_new], [float(w_disb)], [w_share])
        change = (p_new - p_base) * 100
        with line:
            U.html(ui.rows([{"a": "What if: PD in this scenario", "a_small": f"Value {ui.money(float(ev_new[0]))}, "
                                                                            f"loss {ui.money(float(el_new[0]))}",
                             "badge": dec_new, "b_small": f"PD {ui.pct(p_new)}, {change:+.1f} pts"}]))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=np.asarray(p_grid) * 100, mode="lines", name="PD by term",
                                 line=dict(color=ui.CHART["accent"], width=2.5),
                                 hovertemplate="%{x} months<br>PD %{y:.1f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=[w_term], y=[p_new * 100], mode="markers", name="This scenario",
                                 marker=dict(size=12, color=ui.tone_colour(dec_new), line=dict(color="#ffffff", width=2)),
                                 hovertemplate="This scenario<br>%{x} months, PD %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=t * 100, line_color=ui.CHART["ink"], line_dash="dot", line_width=1.5,
                      annotation_text=f"approval threshold {t:.0%}", annotation_position="top left",
                      annotation_font=dict(size=12, color=ui.CHART["muted"]))
        fig.add_hline(y=d * 100, line_color=ui.CHART["bad"], line_dash="dot", line_width=1.5,
                      annotation_text=f"decline from {d:.0%}", annotation_position="top left",
                      annotation_font=dict(size=12, color=ui.CHART["bad"]))
        ui.chart_layout(fig, "PD by term")
        fig.update_layout(showlegend=False, xaxis=dict(title="Term (months)", range=[0, 486]),
                          yaxis=dict(title="PD (%)", range=[0, max(100 * float(np.max(p_grid)) * 1.2, d * 120)]))
        U.chart(fig, height=340, key="whatif_curve")


def picture_value(p: float, gain: float, loss: float, ev: float) -> None:
    grid = np.linspace(0, 1, 201)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=(1 - grid) * gain, mode="lines", name="Gain if repaid, weighted",
                             line=dict(color=ui.CHART["good"], width=2, dash="dot"),
                             hovertemplate="PD %{x:.0%}<br>(1 - PD) x gain USD %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=-grid * loss, mode="lines", name="Loss if default, weighted",
                             line=dict(color=ui.CHART["bad"], width=2, dash="dot"),
                             hovertemplate="PD %{x:.0%}<br>PD x loss USD %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=(1 - grid) * gain - grid * loss, mode="lines", name="Expected net value",
                             line=dict(color=ui.CHART["accent"], width=2.5),
                             hovertemplate="PD %{x:.0%}<br>Expected net value USD %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[p], y=[ev], mode="markers", name="This loan", showlegend=False,
                             marker=dict(size=12, color=ui.CHART["ink"], line=dict(color="#ffffff", width=2)),
                             hovertemplate="This loan<br>PD %{x:.1%}, value USD %{y:,.0f}<extra></extra>"))
    even = gain / (gain + loss) if gain + loss else 0.0
    fig.add_vline(x=t, line_color=ui.CHART["ink"], line_dash="dot", line_width=1.5)
    fig.add_annotation(x=t, y=1.0, yref="paper", text=f"approval threshold {t:.0%}", showarrow=False, xanchor="left",
                       xshift=6, yshift=-6, font=dict(size=12, color=ui.CHART["muted"]))
    fig.add_hline(y=0, line_color=ui.CHART["hairline"], line_width=1)
    fig.add_annotation(x=even, y=0, text=f"break-even PD {even:.1%}", showarrow=False, yshift=12, xanchor="left",
                       xshift=6, font=dict(size=12, color=ui.CHART["muted"]))
    ui.chart_layout(fig, "Expected value against PD")
    fig.update_layout(xaxis=dict(title="Probability of default", tickformat=".0%", range=[0, 1]),
                      yaxis=dict(title="Value to the bank (USD)", tickformat=",.0f"))
    U.chart(fig, height=380, key="value_curve")


def picture_band(feats: pd.DataFrame, ref) -> None:
    if ref is None:
        U.html(ui.empty("The reference sample of training loans is missing: run notebook 05 to write it."))
        return
    ref_df = ref[0]
    row0 = feats.iloc[0]
    jobs = float(row0["CreateJob"] + row0["RetainedJob"])
    dims = [
        ("Term", float(row0["TermYears"]), ref_df["TermYears"], f"{row0['TermYears']:.1f} years"),
        ("Amount disbursed", float(row0["DisbursementGross"]), ref_df["DisbursementGross"],
         ui.money(float(row0["DisbursementGross"]))),
        ("Guaranteed share", float(row0["Portion"]), ref_df["Portion"], f"{row0['Portion']:.0%}"),
        ("Employees", float(row0["NoEmp"]), ref_df["NoEmp"], f"{row0['NoEmp']:,.0f}"),
        ("Jobs created or retained", jobs, ref_df["CreateJob"] + ref_df["RetainedJob"], f"{jobs:,.0f}"),
    ][::-1]
    pcts = [A.percentile_rank(v.to_numpy(), x) for _, x, v, _ in dims]
    fig = go.Figure(go.Bar(
        x=pcts, y=[n for n, *_ in dims], orientation="h", marker_color=ui.CHART["accent"],
        text=[f"{lab}, {q:.0f}th" for (_, _, _, lab), q in zip(dims, pcts)], textposition="outside",
        textfont=dict(size=12, color=ui.CHART["ink"]), cliponaxis=False,
        hovertemplate="%{y}<br>Percentile %{x:.0f}<extra></extra>"))
    fig.add_vline(x=50, line_color=ui.CHART["ink"], line_dash="dot", line_width=1.5)
    fig.add_annotation(x=50, y=1.0, yref="paper", text="median of training loans", showarrow=False, xanchor="left",
                       xshift=6, yshift=-6, font=dict(size=12, color=ui.CHART["muted"]))
    ui.chart_layout(fig, "How this applicant compares")
    fig.update_layout(showlegend=False, bargap=0.35,
                      xaxis=dict(title=f"Percentile among {len(ref_df):,} training loans", range=[0, 118]),
                      yaxis=dict(automargin=True))
    U.chart(fig, height=360, key="band_bars")


# ---------------------------------------------------------------------------
# The page: the form on the left, the result on the right
# ---------------------------------------------------------------------------

if st.session_state.get("example") not in EXAMPLES:
    st.session_state["example"] = "borderline"
form_key = st.session_state.get("example", "borderline")
example = form_values(form_key)
vals = None
submitted = False

with st.container(key="two"):
    left, right = st.columns([1.05, 1], gap="medium")

    with left:
        with st.container(key="form_card"):
            with st.container(key="examples"):
                ec = st.columns(3)
                for col, key in zip(ec, ["borderline", "low", "high"]):
                    col.button(EXAMPLE_LABELS[key], key=f"btn_{key}", use_container_width=True,
                               type="primary" if key == form_key else "secondary")  # a press was read at the top
            guarantee_mode = st.radio("Guarantee entered as", ["Guaranteed amount (USD)", "Guaranteed share"],
                                      horizontal=True, key=f"mode_{form_key}",
                                      help="Give either the guaranteed amount or the share of the approved amount it covers.")
            err_slot = st.container()
            with st.form(f"loan_form_{form_key}", border=False):
                U.html(ui.eyebrow("Loan"))
                c1, c2, c3 = st.columns(3)
                gr = c1.number_input("Approved (USD)", min_value=rng("GrAppv", "min", 1000.0),
                                     max_value=rng("GrAppv", "max", 5e6), value=float(example["GrAppv"]), step=5_000.0,
                                     format="%.0f", help=help_text("GrAppv") or "Loan amount approved.")
                if guarantee_mode == "Guaranteed amount (USD)":
                    sba = c2.number_input("Guaranteed (USD)", min_value=0.0, max_value=rng("SBA_Appv", "max", 5e6),
                                          value=float(example["SBA_Appv"]), step=5_000.0, format="%.0f",
                                          help=help_text("SBA_Appv"))
                    share = None
                else:
                    share = c2.number_input("Guaranteed share", min_value=0.0, max_value=1.0,
                                            value=float(example["Portion"]), step=0.05, format="%.2f",
                                            help=help_text("Portion"))
                    sba = None
                disb = c3.number_input("Disbursed (USD)", min_value=rng("DisbursementGross", "min", 4000.0),
                                       max_value=rng("DisbursementGross", "max", 1.2e7),
                                       value=float(example["DisbursementGross"]), step=5_000.0, format="%.0f",
                                       help=help_text("DisbursementGross"))
                c1, c2, c3 = st.columns(3)
                term = c1.number_input("Term (months)", min_value=1, max_value=480, value=int(example["Term"]), step=6,
                                       help="Scheduled term of the loan in months, 1 to 480.")
                revline = c2.selectbox("Revolving line", ["N", "Y"], index=["N", "Y"].index(example["RevLineCr"]),
                                       format_func=yes_no.get, help="Whether the loan is a revolving line of credit.")
                lowdoc = c3.selectbox("LowDoc programme", ["N", "Y"], index=["N", "Y"].index(example["LowDoc"]),
                                      format_func=yes_no.get, help=(feat_f.get("LowDoc") or {}).get("help", ""))
                U.html(ui.eyebrow("Business"))
                c1, c2, c3 = st.columns(3)
                noemp = c1.number_input("Employees", min_value=0, max_value=int(rng("NoEmp", "max", 300)),
                                        value=int(example["NoEmp"]), help=help_text("NoEmp"))
                create = c2.number_input("Jobs created", min_value=0, max_value=int(rng("CreateJob", "max", 200)),
                                         value=int(example["CreateJob"]), help=help_text("CreateJob"))
                retain = c3.number_input("Jobs retained", min_value=0, max_value=int(rng("RetainedJob", "max", 280)),
                                         value=int(example["RetainedJob"]), help=help_text("RetainedJob"))
                c1, c2, c3 = st.columns(3)
                new_exist = c1.selectbox("Business age", ["Existing", "New"],
                                         index=["Existing", "New"].index(example["NewExist"]),
                                         format_func=lambda v: {"Existing": "Over 2 years", "New": "2 years or less"}[v])
                franchise = c2.selectbox("Franchise", ["N", "Y"], index=["N", "Y"].index(example["IsFranchise"]),
                                         format_func=yes_no.get, help=(feat_f.get("IsFranchise") or {}).get("help", ""))
                urban = c3.selectbox("Location", ["Urban", "Rural"], index=["Urban", "Rural"].index(example["UrbanRural"]),
                                     help="Urban or rural location of the business.")
                U.html(ui.eyebrow("Lender and scenario"))
                c1, c2, c3 = st.columns(3)
                sector = c1.selectbox("Industry sector", sectors, index=sectors.index(example["Sector"]),
                                      help=(feat_f.get("Sector") or {}).get("help", ""))
                state = c2.selectbox("Borrower state", states, index=states.index(example["State"]),
                                     help=(feat_f.get("State") or {}).get("help", ""))
                bank_state = c3.selectbox("Bank state", bank_states, index=bank_states.index(example["BankState"]),
                                          help=(feat_f.get("BankState") or {}).get("help", ""))
                c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
                bank = c1.selectbox("Lending bank", bank_options, index=bank_options.index(example["Bank"]),
                                    help="Type to search the banks in the training data. A bank not listed is scored "
                                         "with the overall historical default rate.")
                recession = c2.toggle("Recession stress", value=bool(example["Recession"]),
                                      help="Scores the loan as if it ran through the 2007 to 2009 recession.")
                submitted = st.form_submit_button("Score application", type="primary", use_container_width=True)
            vals = {"gr": gr, "sba": sba, "share": share, "disb": disb, "term": term, "noemp": noemp, "create": create,
                    "retain": retain, "new_exist": new_exist, "franchise": franchise, "urban": urban,
                    "revline": revline, "lowdoc": lowdoc, "sector": sector, "state": state, "bank": bank,
                    "bank_state": bank_state, "recession": recession}

    if st.session_state.pop("auto_score", False):
        # an example, a shared link or the first visit: score the example as it is stored
        submitted = True
        ex = form_values(form_key)
        vals = {"gr": float(ex["GrAppv"]), "sba": float(ex["SBA_Appv"]), "share": None,
                "disb": float(ex["DisbursementGross"]), "term": int(ex["Term"]), "noemp": int(ex["NoEmp"]),
                "create": int(ex["CreateJob"]), "retain": int(ex["RetainedJob"]), "new_exist": ex["NewExist"],
                "franchise": ex["IsFranchise"], "urban": ex["UrbanRural"], "revline": ex["RevLineCr"],
                "lowdoc": ex["LowDoc"], "sector": ex["Sector"], "state": ex["State"], "bank": ex["Bank"],
                "bank_state": ex["BankState"], "recession": bool(ex["Recession"])}

    if submitted and vals is not None:
        errors, warnings, sba_value = validate(vals)
        if errors:
            st.session_state.pop("score_result", None)
            result = None
            with err_slot:
                U.html("".join(ui.note(e, "bad") for e in errors))
        else:
            v = vals
            row = {
                "GrAppv": v["gr"], "SBA_Appv": sba_value, "DisbursementGross": v["disb"], "Term": v["term"],
                "NoEmp": v["noemp"], "CreateJob": v["create"], "RetainedJob": v["retain"], "NewExist": v["new_exist"],
                "IsFranchise": v["franchise"], "UrbanRural": v["urban"], "RevLineCr": v["revline"], "LowDoc": v["lowdoc"],
                "Sector": v["sector"], "State": v["state"], "Bank": v["bank"], "BankState": v["bank_state"],
                "Recession": int(v["recession"]),
            }
            start = time.perf_counter()
            feats, p = score_rows(pd.DataFrame([row]))
            p = float(p[0])
            try:
                contrib, method = U.local_contributions(model, feats)
            except Exception as exc:  # noqa: BLE001
                contrib, method = None, f"unavailable ({type(exc).__name__})"
            elapsed = time.perf_counter() - start
            st.session_state["score_result"] = {"p": p, "feats": feats, "contrib": contrib, "method": method, "row": row,
                                                "elapsed": elapsed, "warnings": warnings,
                                                "id": f"{time.time():.0f}{p:.6f}", "example": form_key}
            case = A.encode_case(row)
            st.session_state["_case_loaded"] = case
            try:
                st.query_params["case"] = case
            except Exception:  # noqa: BLE001
                pass
            st.session_state["score_edit"] = False
            result = st.session_state["score_result"]

    ref = U.reference_space() if result else None

    with right:
        if not result:
            with st.container(key="score_panel"):
                U.html(ui.card_html("Probability of default", None,
                                    ui.empty("No loan scored yet. Complete the form and select Score.")))
        else:
            p, feats, contrib, row = result["p"], result["feats"], result["contrib"], result["row"]
            band = U.band_for(p)
            decision = str(U.decisions([p], t, d)[0])
            ev, el = U.expected_value([p], feats["DisbursementGross"], feats["Portion"])
            gain, loss = U.value_components(feats["DisbursementGross"], feats["Portion"])
            ev, el, gain, loss = float(ev[0]), float(el[0]), float(gain[0]), float(loss[0])
            portion_v = float(feats["Portion"].iloc[0])

            # The dark result panel: PD, decision and band, the policy line, then value, loss and gain
            with st.container(key="score_panel"):
                U.html(ui.figure_html("", ui.pct(p), "Probability of default", hero=True,
                                      count_to=round(p * 100, 1), decimals=1, suffix="%", fid="pd"))
                U.html(f"<div>{ui.badge(decision, large=True)} {ui.badge(band['name'])}</div>")
                U.html(ui.card_html("", f"Approve below {t:.1%} / Decline from {d:.1%}", ""))
                U.html('<div style="--n:3">' + ui.pair([
                    ("Expected net value", ui.money(ev), None),
                    ("Expected loss", ui.money(el), None),
                    ("Gain if repaid", ui.money(gain), None),
                ]) + "</div>")
                if result["warnings"]:
                    U.html("".join(ui.note(w, "warning") for w in result["warnings"]))

            # What drives this score: the top inputs as a compact table, then the row that opens the sheet
            with st.container(key="sheet_rows_drivers"):
                line = f"SHAP contributions in log-odds. Scored in {result['elapsed']:.2f} s"
                if contrib is not None and len(contrib):
                    body = ui.rows([
                        {"a": U.short_label(r["feature"]), "a_small": str(r["value"]) if r["value"] else "",
                         "badge": "Raises risk" if r["contribution"] > 0 else "Lowers risk",
                         "b_small": f"{r['contribution']:+.2f} log-odds"}
                        for _, r in contrib.head(3).iterrows()
                    ])
                else:
                    body = ui.empty(f"No per-loan explanation for this model ({result['method']}).")
                U.html(ui.card_html("What drives this score", line, body))
                near_b, near_bs = "", ""
                if ref is not None:
                    ref_df, Z, mean, scale, ref_cols = ref
                    idx, dist = A.nearest(Z, mean, scale, feats.iloc[[0]], ref_cols, k=10)
                    k_def = int(ref_df.iloc[idx]["Default"].sum())
                    near_b, near_bs = f"{k_def} of {len(idx)}", "similar loans defaulted"
                    st.session_state["score_near"] = (idx, dist)
                if sheet_row({"a": "See the what-if and similar loans",
                              "a_small": "What if by term · How this applicant compares · Ten most similar past loans",
                              "b": near_b, "b_small": near_bs}, key="btn_figures_decision"):
                    st.session_state["score_sheet"] = "whatif"

            # Actions: the memo downloads, the share link and the form
            drivers = U.explanation_sentences(contrib) if contrib is not None and len(contrib) else []
            memo_inputs = {
                "Loan amount approved": f"USD {row['GrAppv']:,.0f}", "Guaranteed amount": f"USD {row['SBA_Appv']:,.0f}",
                "Guaranteed share": f"{portion_v:.0%}", "Amount disbursed": f"USD {row['DisbursementGross']:,.0f}",
                "Term": f"{row['Term']} months", "Revolving line of credit": yes_no[row["RevLineCr"]],
                "LowDoc programme": yes_no[row["LowDoc"]], "Employees": row["NoEmp"], "Jobs created": row["CreateJob"],
                "Jobs retained": row["RetainedJob"], "Business age": row["NewExist"],
                "Franchise": yes_no[row["IsFranchise"]], "Location": row["UrbanRural"], "Industry sector": row["Sector"],
                "Borrower state": row["State"], "Lending bank": row["Bank"].title(), "Bank state": row["BankState"],
                "Recession stress": "On" if row["Recession"] else "Off",
            }
            memo = U.memo_html(memo_inputs, p, band, decision, ev, el, gain, loss, drivers, t, d)
            memo_print = U.memo_html(memo_inputs, p, band, decision, ev, el, gain, loss, drivers, t, d,
                                     contrib=contrib, printable=True)
            with st.container(key="sheet_rows_actions"):
                U.html(ui.card_html("Actions", None, ""))
                download_row("Download memo", "memo", memo.encode("utf-8"), "creditlens_credit_memo.html")
                download_row("PDF-ready memo", "memo_print", memo_print.encode("utf-8"), "creditlens_credit_memo_print.html")
                if ui.row_button({"a": "Share link", "a_small": "Opening it scores this application again"},
                                 key="score_share"):
                    st.session_state["score_sheet"] = "share"
                ui.row_button({"a": "Edit application", "a_small": "The form on the left"}, key="edit")  # read at the top


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

@st.dialog("What if and similar loans", width="large")
def whatif_sheet(res: dict) -> None:
    feats, p = res["feats"], res["p"]
    ev, _ = U.expected_value([p], feats["DisbursementGross"], feats["Portion"])
    gain, loss = U.value_components(feats["DisbursementGross"], feats["Portion"])
    ev, gain, loss = float(ev[0]), float(gain[0]), float(loss[0])
    tabs = st.tabs(["What if by term", "How this applicant compares", "Similar loans", "Value against PD", "Method"])
    with tabs[0]:
        what_if(res["row"], res["id"], p)
    with tabs[1]:
        picture_band(feats, U.reference_space())
    with tabs[2]:
        near_state = st.session_state.get("score_near")
        ref = U.reference_space()
        if ref is None or near_state is None:
            U.html(ui.empty("The reference sample of training loans is missing: run notebook 05 to write it."))
        else:
            idx, dist = near_state
            ref_df = ref[0]
            near = ref_df.iloc[idx]
            k_def = int(near["Default"].sum())
            U.html(ui.pair([("Defaulted", f"{k_def / len(near):.0%}", f"{k_def} of {len(near)}"),
                            ("Model PD", ui.pct(p), U.band_for(p)["name"])]))
            U.html(ui.eyebrow("Ten most similar past loans"))
            table = pd.DataFrame({
                "Term (months)": near["Term"].astype(int).to_numpy(),
                "Disbursed (USD)": near["DisbursementGross"].round(0).to_numpy(),
                "Guaranteed share": near["Portion"].to_numpy(),
                "Employees": near["NoEmp"].astype(int).to_numpy(),
                "Sector": near["Sector"].to_numpy(), "State": near["State"].to_numpy(),
                "Outcome": np.where(near["Default"] == 1, "Charged off", "Repaid"),
                "Distance": np.asarray(dist).round(2),
            })
            st.dataframe(table, hide_index=True, use_container_width=True, column_config={
                "Disbursed (USD)": st.column_config.NumberColumn(format="localized"),
                "Guaranteed share": st.column_config.NumberColumn(format="percent")})
            U.html(ui.note(f"Nearest neighbours in the standardised numeric inputs, from a committed sample of "
                           f"{len(ref_df):,} training loans.", "accent"))
    with tabs[3]:
        picture_value(p, gain, loss, ev)
    with tabs[4]:
        if res["contrib"] is not None and len(res["contrib"]):
            U.html(ui.eyebrow("This application")
                   + "".join(ui.note(s, "accent") for s in U.explanation_sentences(res["contrib"])))
        U.html(ui.eyebrow("How this is worked out") + "".join(ui.note(s, "accent") for s in [
            f"Decision: approve below {t:.1%}, refer from {t:.1%} to {d:.1%}, decline from {d:.1%}.",
            f"Drivers are SHAP contributions in log-odds; one-hot levels are summed per input ({res['method']}).",
            "What if rescores this applicant only; the form is not changed.",
            "Similar loans are the nearest neighbours in the standardised numeric inputs (counts and money per employee "
            "on a log scale), from a committed sample of training loans.",
        ]))
        disb_v, portion_v = float(feats["DisbursementGross"].iloc[0]), float(feats["Portion"].iloc[0])
        c = U.cost_assumptions()
        st.code(
            U.formula_text() + "\n\n"
            f"This loan: disbursed USD {disb_v:,.0f}, guaranteed share {portion_v:.0%}, PD {p:.4f}\n"
            f"Loss if default = {c['LOSS_GIVEN_DEFAULT']:.2f} x {1 - portion_v:.2f} x {disb_v:,.0f} + "
            f"{c['WORKOUT_COST_RATE']:.2f} x {disb_v:,.0f} = USD {loss:,.0f}\n"
            f"Expected value = (1 - {p:.4f}) x {gain:,.0f} - {p:.4f} x {loss:,.0f} = USD {ev:,.0f}",
            language=None,
        )


@st.dialog("Share link")
def share_sheet(res: dict) -> None:
    case = A.encode_case(res["row"])
    host = st.context.headers.get("host", "") if hasattr(st, "context") else ""
    scheme = "http" if host.startswith(("localhost", "127.0.0.1")) else "https"
    link = f"{scheme}://{host}/score?case={case}" if host else f"/score?case={case}"
    U.html(ui.note("The link holds the form values only; opening it scores the application again.", "accent"))
    st.code(link, language=None)


sheet = st.session_state.pop("score_sheet", None)
if sheet is None and result and whatif_moved():
    sheet = "whatif"  # a what-if slider moved in a full rerun: keep its sheet open
if sheet == "whatif" and result:
    whatif_sheet(result)
elif sheet == "share" and result:
    share_sheet(result)
else:
    st.session_state.pop("_wi_snap", None)
