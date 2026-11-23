"""Score a loan: application form, PD gauge, decision, value, explanation, what-if panel and memo."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import analytics as A  # noqa: E402
import experience as X  # noqa: E402
import utils as U  # noqa: E402

# Model and SHAP explainer are loaded once per session, before the first score
model = U.load_model()
try:
    U.warm_explainer()
except Exception:  # noqa: BLE001
    pass
X.page_start("score")
U.page_header(
    "Score a loan",
    "Enter the application as the credit desk receives it. Inputs are checked against the training ranges, "
    "scored, priced with the cost matrix and explained.",
    eyebrow="Single application",
)

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


def form_head(text: str, icon_name: str) -> None:
    U.html(f'<div class="cl-form-head"><span class="ic">{U.icon(icon_name, 17)}</span>{text}</div>')


left, right = st.columns([1.08, 1], gap="large")

with left:
    with st.container(key="examples"):
        U.html('<div class="cl-tile-label" style="margin-bottom:6px">Load an example case</div>')
        ec = st.columns([1.15, 1, 1, 0.8])
        for col, key in zip(ec, ["borderline", "low", "high"]):
            if col.button(EXAMPLE_LABELS[key], key=f"btn_{key}"):
                st.session_state["example"] = key
                st.session_state["auto_score"] = True
    if st.session_state.get("example") not in EXAMPLES:
        st.session_state["example"] = "borderline"
    example = EXAMPLES[st.session_state["example"]]
    form_key = st.session_state.get("example", "borderline")

    guarantee_mode = st.radio("Guarantee entered as", ["Guaranteed amount (USD)", "Guaranteed share"],
                              horizontal=True, key=f"mode_{form_key}",
                              help="Give either the guaranteed amount or the share of the approved amount it covers.")

    with st.form(f"loan_form_{form_key}", border=False):
        form_head("Loan", "coins")
        c1, c2 = st.columns(2)
        gr = c1.number_input("Loan amount approved (USD)", min_value=rng("GrAppv", "min", 1000.0),
                             max_value=rng("GrAppv", "max", 5e6), value=example["GrAppv"], step=5_000.0, format="%.0f",
                             help=help_text("GrAppv"))
        if guarantee_mode == "Guaranteed amount (USD)":
            sba = c2.number_input("Guaranteed amount (USD)", min_value=0.0, max_value=rng("SBA_Appv", "max", 5e6),
                                  value=example["SBA_Appv"], step=5_000.0, format="%.0f", help=help_text("SBA_Appv"))
            share = None
        else:
            share = c2.number_input("Guaranteed share", min_value=0.0, max_value=1.0, value=example["Portion"],
                                    step=0.05, format="%.2f", help=help_text("Portion"))
            sba = None
        c1, c2 = st.columns(2)
        disb = c1.number_input("Amount disbursed (USD)", min_value=rng("DisbursementGross", "min", 4000.0),
                               max_value=rng("DisbursementGross", "max", 1.2e7), value=example["DisbursementGross"],
                               step=5_000.0, format="%.0f", help=help_text("DisbursementGross"))
        term = c2.number_input("Term (months)", min_value=1, max_value=480, value=int(example["Term"]), step=6,
                               help="Scheduled term of the loan in months, 1 to 480.")
        c1, c2 = st.columns(2)
        revline = c1.selectbox("Revolving line of credit", ["N", "Y"], index=["N", "Y"].index(example["RevLineCr"]),
                               format_func=yes_no.get, help="Whether the loan is a revolving line of credit.")
        lowdoc = c2.selectbox("LowDoc programme", ["N", "Y"], index=["N", "Y"].index(example["LowDoc"]),
                              format_func=yes_no.get, help=(feat_f.get("LowDoc") or {}).get("help", ""))

        form_head("Business", "briefcase")
        c1, c2, c3 = st.columns(3)
        noemp = c1.number_input("Employees", min_value=0, max_value=int(rng("NoEmp", "max", 300)),
                                value=int(example["NoEmp"]), help=help_text("NoEmp"))
        create = c2.number_input("Jobs created", min_value=0, max_value=int(rng("CreateJob", "max", 200)),
                                 value=int(example["CreateJob"]), help=help_text("CreateJob"))
        retain = c3.number_input("Jobs retained", min_value=0, max_value=int(rng("RetainedJob", "max", 280)),
                                 value=int(example["RetainedJob"]), help=help_text("RetainedJob"))
        c1, c2, c3 = st.columns(3)
        new_exist = c1.selectbox("Business age", ["Existing", "New"], index=["Existing", "New"].index(example["NewExist"]),
                                 format_func=lambda v: {"Existing": "Existing (over 2 years)", "New": "New (2 years or less)"}[v])
        franchise = c2.selectbox("Franchise", ["N", "Y"], index=["N", "Y"].index(example["IsFranchise"]),
                                 format_func=yes_no.get, help=(feat_f.get("IsFranchise") or {}).get("help", ""))
        urban = c3.selectbox("Location", ["Urban", "Rural"], index=["Urban", "Rural"].index(example["UrbanRural"]),
                             help="Urban or rural location of the business.")
        c1, c2 = st.columns([2, 1])
        sector = c1.selectbox("Industry sector", sectors, index=sectors.index(example["Sector"]),
                              help=(feat_f.get("Sector") or {}).get("help", ""))
        state = c2.selectbox("Borrower state", states, index=states.index(example["State"]),
                             help=(feat_f.get("State") or {}).get("help", ""))

        form_head("Lender and scenario", "bank")
        c1, c2 = st.columns([2, 1])
        bank = c1.selectbox("Lending bank", bank_options, index=bank_options.index(example["Bank"]),
                            help="Type to search the banks in the training data. A bank not listed is scored with the "
                                 "overall historical default rate.")
        bank_state = c2.selectbox("Bank state", bank_states, index=bank_states.index(example["BankState"]),
                                  help=(feat_f.get("BankState") or {}).get("help", ""))
        recession = st.toggle("Recession stress: score as if the loan ran through 2007 to 2009",
                              value=bool(example["Recession"]),
                              help="Sets the recession indicator used by the model.")
        submitted = st.form_submit_button("Score this loan", type="primary", use_container_width=True)

if st.session_state.pop("auto_score", False):
    submitted = True


def validate() -> tuple[list[str], list[str], float]:
    errors, warnings = [], []
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
        warnings.append(f"A guaranteed share of {portion:.0%} is outside the range seen in training ({lo:.0%} to {hi:.0%}).")
    ratio = disb / gr if gr else 0
    p01, p99 = rng("DisbursedToApprovedRatio", "p01", 0.7), rng("DisbursedToApprovedRatio", "p99", 3.9)
    if not p01 <= ratio <= p99:
        warnings.append(f"The disbursed amount is {ratio:.2f} times the approved amount, which is unusual "
                        f"(98% of training loans lie between {p01:.2f} and {p99:.2f}).")
    for name, value in [("NoEmp", noemp), ("Term", term), ("GrAppv", gr)]:
        lo_, hi_ = rng(name, "p01", None), rng(name, "p99", None)
        if lo_ is not None and hi_ is not None and not lo_ <= value <= hi_:
            label = (raw_f.get(name) or feat_f.get(name) or {}).get("label", name)
            warnings.append(f"{label} of {value:,.0f} is outside the range of 98% of training loans; treat the score with care.")
    return errors, warnings, sba_value


if submitted:
    errors, warnings, sba_value = validate()
    if errors:
        with left:
            for e in errors:
                st.error(e)
        st.session_state.pop("score_result", None)
    else:
        row = {
            "GrAppv": gr, "SBA_Appv": sba_value, "DisbursementGross": disb, "Term": term, "NoEmp": noemp,
            "CreateJob": create, "RetainedJob": retain, "NewExist": new_exist, "IsFranchise": franchise,
            "UrbanRural": urban, "RevLineCr": revline, "LowDoc": lowdoc, "Sector": sector, "State": state,
            "Bank": bank, "BankState": bank_state, "Recession": int(recession),
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
                                            "elapsed": elapsed, "warnings": warnings, "id": f"{time.time():.0f}{p:.6f}",
                                            "example": form_key, "fresh": True}
        case = A.encode_case(row)
        st.session_state["_case_loaded"] = case
        try:
            st.query_params["case"] = case
        except Exception:  # noqa: BLE001
            pass

result = st.session_state.get("score_result")
t, d = U.threshold(), U.decline_threshold()

with right:
    if not result:
        U.html(f'<div class="cl-card cl-empty">{U.icon("gauge", 40, "#8D99AE")}<h4 style="margin-top:12px">'
               "No loan scored yet</h4><p>Complete the form and select Score this loan.</p></div>")
    else:
        p, feats = result["p"], result["feats"]
        band = U.band_for(p)
        decision = str(U.decisions([p], t, d)[0])
        ev, el = U.expected_value([p], feats["DisbursementGross"], feats["Portion"])
        gain, loss = U.value_components(feats["DisbursementGross"], feats["Portion"])
        ev, el, gain, loss = float(ev[0]), float(el[0]), float(gain[0]), float(loss[0])
        with st.container(key="result_card"):
            if result.pop("fresh", False):
                U.html(X.scoring_sequence(result["id"]))
            for w in result["warnings"]:
                st.warning(w)
            U.html(
                '<div class="cl-tile-label">Probability of default</div>'
                f'<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin:4px 0 10px">'
                f'<span class="cl-big" style="font-variant-numeric:tabular-nums;font-size:3.2rem;font-weight:800;'
                f'letter-spacing:-0.04em;line-height:1;color:{U.INK}">{p:.1%}</span>{U.chip(band["name"] + " risk", band["colour"])}</div>'
                f'<div style="margin:6px 0 6px">{U.decision_chip(decision)}</div>'
            )
            U.chart(U.gauge_figure(p, t), height=230, key="gauge")
            U.html(f'<div class="cl-page-intro" style="font-size:0.85rem;margin:-6px 0 14px">Approve below {t:.1%}, '
                   f"refer from {t:.1%} to {d:.1%}, decline from {d:.1%}.</div>")
            U.tiles([
                {"label": "Expected net value", "value": U.fmt_usd(ev), "sub": "If approved",
                 "accent": U.COLOURS["teal"] if ev >= 0 else U.COLOURS["red"]},
                {"label": "Expected loss", "value": U.fmt_usd(el), "sub": "PD x loss if default", "accent": U.COLOURS["red"]},
                {"label": "Gain if repaid", "value": U.fmt_usd(gain), "sub": "Net interest margin", "accent": U.COLOURS["navy"]},
            ], cols=3, min_width=112, compact=True)
            disb_v, portion_v = float(feats["DisbursementGross"].iloc[0]), float(feats["Portion"].iloc[0])
            c = U.cost_assumptions()
            with st.expander("How the expected value is calculated"):
                st.code(
                    U.formula_text() + "\n\n"
                    f"This loan: disbursed USD {disb_v:,.0f}, guaranteed share {portion_v:.0%}, PD {p:.4f}\n"
                    f"Loss if default = {c['LOSS_GIVEN_DEFAULT']:.2f} x {1 - portion_v:.2f} x {disb_v:,.0f} + "
                    f"{c['WORKOUT_COST_RATE']:.2f} x {disb_v:,.0f} = USD {loss:,.0f}\n"
                    f"Expected value = (1 - {p:.4f}) x {gain:,.0f} - {p:.4f} x {loss:,.0f} = USD {ev:,.0f}",
                    language=None,
                )
            contrib = result["contrib"]
            drivers = U.explanation_sentences(contrib) if contrib is not None and len(contrib) else []
            row = result["row"]
            memo_inputs = {
                "Loan amount approved": f"USD {row['GrAppv']:,.0f}", "Guaranteed amount": f"USD {row['SBA_Appv']:,.0f}",
                "Guaranteed share": f"{portion_v:.0%}", "Amount disbursed": f"USD {row['DisbursementGross']:,.0f}",
                "Term": f"{row['Term']} months", "Revolving line of credit": yes_no[row["RevLineCr"]],
                "LowDoc programme": yes_no[row["LowDoc"]], "Employees": row["NoEmp"], "Jobs created": row["CreateJob"],
                "Jobs retained": row["RetainedJob"], "Business age": row["NewExist"], "Franchise": yes_no[row["IsFranchise"]],
                "Location": row["UrbanRural"], "Industry sector": row["Sector"], "Borrower state": row["State"],
                "Lending bank": row["Bank"].title(), "Bank state": row["BankState"],
                "Recession stress": "On" if row["Recession"] else "Off",
            }
            memo = U.memo_html(memo_inputs, p, band, decision, ev, el, gain, loss, drivers, t, d)
            m1, m2 = st.columns(2)
            m1.download_button("Download decision memo", memo.encode("utf-8"), file_name="creditlens_credit_memo.html",
                               mime="text/html", type="secondary", use_container_width=True, key="memo",
                               icon=":material/description:")
            memo_print = U.memo_html(memo_inputs, p, band, decision, ev, el, gain, loss, drivers, t, d,
                                     contrib=contrib, printable=True)
            m2.download_button("PDF-ready memo", memo_print.encode("utf-8"), file_name="creditlens_credit_memo_print.html",
                               mime="text/html", type="secondary", use_container_width=True, key="memo_print",
                               icon=":material/print:",
                               help="A print-styled page with a Print or save as PDF button; open it in a browser.")
            case = A.encode_case(row)
            host = st.context.headers.get("host", "") if hasattr(st, "context") else ""
            scheme = "http" if host.startswith(("localhost", "127.0.0.1")) else "https"
            link = f"{scheme}://{host}/score?case={case}" if host else f"/score?case={case}"
            with st.expander("Share this case"):
                st.code(link, language=None)
                st.caption("Anyone with the link sees this application scored with the same model. The link holds the "
                           "form values only.")
            st.caption(f"Scored and explained in {result['elapsed']:.2f} seconds.")

if result:
    p, feats, contrib = result["p"], result["feats"], result["contrib"]
    band = U.band_for(p)
    decision = str(U.decisions([p], t, d)[0])

    U.section_title("What drives this score", "chart")
    c1, c2 = st.columns([1.5, 1], gap="large")
    if contrib is not None and len(contrib):
        top = contrib.head(10).iloc[::-1].reset_index(drop=True)
        vmax = float(top["contribution"].abs().max()) or 1.0
        labels = [f"{U.short_label(f)}: {val}" if val else U.short_label(f) for f, val in zip(top["feature"], top["value"])]
        labels = [s if len(s) <= 30 else s[:28] + ".." for s in labels]
        fig = go.Figure(go.Bar(
            x=top["contribution"], y=list(range(len(top))), orientation="h",
            marker=dict(color=[U.COLOURS["red"] if v > 0 else U.COLOURS["teal"] for v in top["contribution"]]),
            customdata=labels, hovertemplate="%{customdata}<br>Contribution %{x:.3f}<extra></extra>",
        ))
        for i, (v, lab) in enumerate(zip(top["contribution"], labels)):
            # The label sits on the empty side of the zero line, next to its bar
            fig.add_annotation(x=0, y=i, text=lab, showarrow=False, xanchor="right" if v > 0 else "left",
                               xshift=-8 if v > 0 else 8, font=dict(size=12, color=U.INK))
        fig.add_vline(x=0, line_color="#8D99AE", line_width=1)
        fig.update_layout(barcornerradius=7, bargap=0.28, yaxis=dict(visible=False),
                          xaxis=dict(title="Contribution to the score (log-odds)", range=[-1.12 * vmax, 1.12 * vmax],
                                     zeroline=False),
                          margin=dict(l=8, r=8, t=16, b=40))
        with c1:
            U.chart(fig, height=36 * len(top) + 90, key="shap_chart")
        with c2:
            if decision == "Approve":
                lead = f"below the approval threshold of {t:.1%}, so CreditLens recommends approval."
            elif decision == "Refer":
                lead = f"between the approval threshold of {t:.1%} and the decline level of {d:.1%}, so CreditLens recommends manual review."
            else:
                lead = f"at or above the decline level of {d:.1%}, so CreditLens recommends declining."
            items = "".join(f"<li>{s}</li>" for s in U.explanation_sentences(contrib))
            U.html(f'<div class="cl-card cl-reading"><h4>Plain-language reading</h4><p style="color:{U.INK}">'
                   f"A probability of default of <b>{p:.1%}</b> places this loan in the {band['name'].lower()} risk band, "
                   f"{lead}</p><ul style='margin-top:12px'>{items}</ul></div>")
            st.caption(f"Method: {result['method']}. Coral bars raise the risk, teal bars lower it; one-hot levels are "
                       "summed per input.")
    else:
        with c1:
            st.info("A per-loan explanation could not be computed for this model, so the global importance is shown.")
            g = U.global_importance(10)
            if g is not None:
                fig = go.Figure(go.Bar(x=g["importance"][::-1], y=g["label"][::-1], orientation="h",
                                       marker_color=U.COLOURS["navy"]))
                fig.update_layout(xaxis_title="Mean absolute SHAP value", barcornerradius=6)
                U.chart(fig, height=380)

    U.section_title("How this applicant compares", "users")
    ref = U.reference_space()
    if ref is None:
        st.info("The reference sample of training loans (app/assets/aggregates) is not available.")
    else:
        ref_df, Z, mean, scale, ref_cols = ref
        row0 = feats.iloc[0]
        jobs = float(row0["CreateJob"] + row0["RetainedJob"])
        dims = [
            ("Term", float(row0["TermYears"]), ref_df["TermYears"], f"{row0['TermYears']:.1f} years"),
            ("Amount", float(row0["DisbursementGross"]), ref_df["DisbursementGross"], U.fmt_usd(float(row0["DisbursementGross"]))),
            ("Guaranteed share", float(row0["Portion"]), ref_df["Portion"], f"{row0['Portion']:.0%}"),
            ("Employees", float(row0["NoEmp"]), ref_df["NoEmp"], f"{row0['NoEmp']:,.0f}"),
            ("Jobs", jobs, ref_df["CreateJob"] + ref_df["RetainedJob"], f"{jobs:,.0f} created or retained"),
        ]
        pct = [A.percentile_rank(v.to_numpy(), x) for _, x, v, _ in dims]
        names = [n for n, *_ in dims]
        r1, r2 = st.columns([1.1, 1], gap="large")
        with r1:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=[50] * 6, theta=names + names[:1], mode="lines", name="Median training loan",
                                          line=dict(color=U.COLOURS["grey"], dash="dot", width=1.5), hoverinfo="skip"))
            fig.add_trace(go.Scatterpolar(
                r=pct + pct[:1], theta=names + names[:1], fill="toself", name="This application",
                fillcolor=U.hex_rgba(U.COLOURS["teal"], 0.25), line=dict(color=U.COLOURS["teal"], width=3),
                marker=dict(size=9, color=U.COLOURS["teal"], line=dict(color="#FFFFFF", width=2)),
                customdata=[[lab] for *_, lab in dims] + [[dims[0][3]]],
                hovertemplate="<b>%{theta}</b>: %{customdata[0]}<br>Higher than %{r:.0f}% of training loans<extra></extra>"))
            fig.update_layout(title="Applicant against the training portfolio (percentile)",
                              polar=dict(bgcolor="rgba(0,0,0,0)",
                                         radialaxis=dict(range=[0, 100], ticksuffix="", tickvals=[25, 50, 75, 100],
                                                         gridcolor=U.GRID, tickfont=dict(size=10, color=U.MUTED)),
                                         angularaxis=dict(gridcolor=U.GRID, tickfont=dict(size=12, color=U.INK))),
                              legend=dict(orientation="h", y=-0.12), margin=dict(l=40, r=40, t=70, b=30))
            U.chart(fig, height=390, key="radar")
            st.caption(f"Percentile of each input among {len(ref_df):,} training loans: 50 is the median loan.")
        with r2:
            idx, dist = A.nearest(Z, mean, scale, feats.iloc[[0]], ref_cols, k=10)
            near = ref_df.iloc[idx]
            k_def = int(near["Default"].sum())
            rate = k_def / len(near)
            dots = "".join(f'<i style="background:{U.COLOURS["red"] if v else U.COLOURS["teal"]}" '
                           f'title="{"Charged off" if v else "Repaid"}"></i>' for i, v in enumerate(near["Default"]))
            if abs(rate - p) <= 0.10:
                reading = "which is close to the model's estimate for this application."
            elif rate > p:
                reading = "a worse record than the model's estimate, so the score leans on factors the neighbours do not share."
            else:
                reading = "a better record than the model's estimate, so other factors such as the lender or sector push the score up."
            U.html(
                f'<div class="cl-card"><h4>{U.icon("users", 18)} Ten most similar past loans</h4>'
                f'<div style="display:flex;align-items:baseline;gap:12px"><span class="cl-big-num">{rate:.0%}</span>'
                f'<span class="cl-muted">defaulted ({k_def} of {len(near)})</span></div>'
                f'<div class="cl-mix"><span style="width:{100 - rate * 100:.0f}%;background:{U.COLOURS["teal"]}"></span>'
                f'<span style="width:{rate * 100:.0f}%;background:{U.COLOURS["red"]}"></span></div>'
                f'<div class="cl-mix-legend"><span>Repaid {len(near) - k_def}</span><span>Charged off {k_def}</span></div>'
                f'<div class="cl-dots">{dots}</div>'
                f'<p>Of the ten training loans closest to this one in term, size, guarantee, staff and jobs, {k_def} '
                f'were charged off, {reading} The model puts this loan at {p:.1%}.</p></div>')
            with st.expander("See the ten loans"):
                show = pd.DataFrame({
                    "Term (months)": near["Term"].astype(int).to_numpy(),
                    "Disbursed (USD)": near["DisbursementGross"].round(0).to_numpy(),
                    "Guaranteed share": near["Portion"].to_numpy(),
                    "Employees": near["NoEmp"].astype(int).to_numpy(),
                    "Sector": near["Sector"].to_numpy(), "State": near["State"].to_numpy(),
                    "Outcome": np.where(near["Default"] == 1, "Charged off", "Repaid"),
                    "Distance": dist.round(2),
                })
                st.dataframe(show, hide_index=True, use_container_width=True, column_config={
                    "Disbursed (USD)": st.column_config.NumberColumn(format="localized"),
                    "Guaranteed share": st.column_config.NumberColumn(format="percent")})
                st.caption("Distance in the standardised space of the numeric model inputs (counts and money per "
                           "employee on a log scale), from a committed sample of training loans.")

    @st.fragment
    def what_if(base: dict, rid: str, p_base: float) -> None:
        """Rescore this applicant as three terms change; only this panel reruns."""
        with st.container(key="whatif"):
            U.html(f'<div class="cl-section-title" style="margin-top:0">{U.icon("sliders", 22)}<span>What if</span></div>'
                   '<div class="cl-page-intro" style="font-size:0.9rem;margin-bottom:8px">Change the structure of this '
                   "loan and see the score update. The form above is not changed.</div>")
            gr0 = float(base["GrAppv"])
            portion0 = float(np.clip(base["SBA_Appv"] / gr0, 0.05, 1.0))
            dmax = float(min(rng("DisbursementGross", "max", 1.2e7), max(4 * base["DisbursementGross"], 500_000)))
            s1, s2, s3 = st.columns(3)
            w_term = s1.slider("Term (months)", 1, 480, int(base["Term"]), key=f"wi_term_{rid}")
            w_share = s2.slider("Guaranteed share", 0.05, 1.0, round(portion0, 2), 0.01, key=f"wi_share_{rid}")
            w_disb = s3.slider("Amount disbursed (USD)", 5_000, int(dmax), int(round(base["DisbursementGross"], -3)),
                               5_000, key=f"wi_disb_{rid}")
            grid = sorted(set(list(range(12, 481, 12)) + [w_term]))
            rows = pd.DataFrame([{**base, "Term": m, "SBA_Appv": w_share * gr0, "DisbursementGross": float(w_disb)}
                                 for m in grid])
            feats_w, p_grid = score_rows(rows)
            i = grid.index(w_term)
            p_new = float(p_grid[i])
            dec_new = str(U.decisions([p_new], t, d)[0])
            ev_new, el_new = U.expected_value([p_new], [float(w_disb)], [w_share])
            change = (p_new - p_base) * 100
            k1, k2 = st.columns([1, 1.6], gap="large")
            with k1:
                U.tiles([
                    {"label": "PD in this scenario", "value": f"{p_new:.1%}",
                     "sub": f"<span class='{'cl-delta-down' if change > 0 else 'cl-delta-up'}'>{change:+.1f} pts</span> "
                            "against the scored loan", "accent": U.band_for(p_new)["colour"]},
                    {"label": "Expected net value", "value": U.fmt_usd(float(ev_new[0])),
                     "sub": f"Expected loss {U.fmt_usd(float(el_new[0]))}", "accent": U.COLOURS["navy"]},
                ], cols=1)
                U.html(f"<div style='margin:4px 0 8px'>{U.decision_chip(dec_new)}</div>")
            with k2:
                fig = go.Figure()
                for b in U.risk_bands():
                    fig.add_hrect(y0=b["min"] * 100, y1=min(b["max"], 1) * 100, fillcolor=b["colour"], opacity=0.09,
                                  line_width=0, layer="below")
                fig.add_trace(go.Scatter(x=grid, y=np.asarray(p_grid) * 100, mode="lines+markers", name="PD by term",
                                         line=dict(color=U.COLOURS["navy"], width=3), marker=dict(size=5, color=U.COLOURS["navy"]),
                                         hovertemplate="%{x} months<br>PD %{y:.1f}%<extra></extra>"))
                fig.add_trace(go.Scatter(x=[w_term], y=[p_new * 100], mode="markers", name="This scenario",
                                         marker=dict(size=13, color=U.DECISION_COLOURS[dec_new],
                                                     line=dict(color="#FFFFFF", width=2.5))))
                fig.add_hline(y=t * 100, line_color=U.INK, line_dash="dot", line_width=1.5,
                              annotation_text=f"threshold {t:.1%}", annotation_position="top left",
                              annotation_font=dict(size=11, color=U.INK))
                fig.update_layout(title="PD against term for this applicant", showlegend=False,
                                  xaxis=dict(title="Term (months)", range=[0, 486]),
                                  yaxis=dict(title="PD (%)", range=[0, 100]), margin=dict(l=8, r=8, t=40, b=40))
                U.chart(fig, height=300, key="whatif_curve")
                st.caption("Each point is a whole-year term with the other inputs held at the values above.")

    what_if(result["row"], result["id"], p)
