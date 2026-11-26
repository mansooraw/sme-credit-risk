"""Compare applicants, built from the compare concept: the copy strip, then the comparison table (a metric column and
one card per applicant with its PD, decision and inputs; the best expected value tinted), or the input table while
editing; below, the drivers of each score beside the recommendation and the rows that open the sheets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import analytics as A  # noqa: E402
import ui  # noqa: E402
import utils as U  # noqa: E402

model = U.load_model()
ui.page("Compare applicants", "Three profiles. One clear comparison.", eyebrow="Decide", heading="Compare applicants")

fields = U.schema_fields()
feat_f = fields["features"]
sectors = sorted(s for s in ((feat_f.get("Sector") or {}).get("allowed") or U.SECTOR_TO_NAICS) if s != "Unknown") + ["Unknown"]
states = sorted(s for s in (feat_f.get("State") or {}).get("allowed", []) if s != "Unknown")
bank_states = sorted(s for s in (feat_f.get("BankState") or {}).get("allowed", []) if s != "Unknown")
banks = [U.NEW_BANK_OPTION] + sorted(b for b in (feat_f.get("Bank") or {}).get("allowed", []) if b != "Unknown")
NAMES = ["A", "B", "C"]
INPUTS = ["GrAppv", "Portion", "DisbursementGross", "Term", "NoEmp", "CreateJob", "RetainedJob", "NewExist", "IsFranchise",
          "UrbanRural", "RevLineCr", "LowDoc", "Sector", "State", "Bank", "BankState", "Recession"]
OUTPUTS = ["PD", "Band", "Decision", "Expected value", "Expected loss"]
WORDS = {"Approve": "Approved", "Refer": "Referred", "Decline": "Declined"}


def from_example(ex: dict) -> dict:
    row = {k: ex[k] for k in INPUTS if k in ex}
    row["Portion"] = float(ex["SBA_Appv"]) / float(ex["GrAppv"])
    row["Recession"] = bool(ex["Recession"])
    if row["Bank"] not in banks:
        row["Bank"] = U.NEW_BANK_OPTION
    return row


def valid(row: dict) -> bool:
    return (row.get("Sector") in sectors and row.get("State") in states and row.get("BankState") in bank_states)


def score(base: pd.DataFrame) -> pd.DataFrame:
    """Rescore every applicant in the table and fill the output columns."""
    rows = base[INPUTS].copy()
    rows["GrAppv"] = pd.to_numeric(rows["GrAppv"], errors="coerce").clip(lower=1000).fillna(100_000)
    rows["Portion"] = pd.to_numeric(rows["Portion"], errors="coerce").clip(0.05, 1.0).fillna(0.75)
    rows["DisbursementGross"] = pd.to_numeric(rows["DisbursementGross"], errors="coerce").clip(lower=4000).fillna(rows["GrAppv"])
    rows["Term"] = pd.to_numeric(rows["Term"], errors="coerce").clip(1, 480).fillna(84).astype(int)
    for c in ["NoEmp", "CreateJob", "RetainedJob"]:
        rows[c] = pd.to_numeric(rows[c], errors="coerce").clip(lower=0).fillna(0).astype(int)
    rows["SBA_Appv"] = rows["Portion"] * rows["GrAppv"]
    rows["Recession"] = rows["Recession"].astype(bool).astype(int)
    feats, p = U.score_form_rows(rows)
    ev, el = U.expected_value(p, feats["DisbursementGross"], feats["Portion"])
    out = base.copy()
    for c in ["GrAppv", "Portion", "DisbursementGross", "Term", "NoEmp", "CreateJob", "RetainedJob"]:
        out[c] = rows[c].to_numpy()
    out["PD"] = p
    out["Band"] = U.band_names(p)
    out["Decision"] = U.decisions(p, U.shared_threshold(), max(U.decline_threshold(), U.shared_threshold()))
    out["Expected value"] = np.round(ev, 0)
    out["Expected loss"] = np.round(el, 0)
    st.session_state["cmp_feats"] = feats
    return out


def initial_table() -> pd.DataFrame:
    rows = {n: from_example(U.EXAMPLES[k]) for n, k in zip(NAMES, ["borderline", "low", "high"])}
    for n, param in zip(NAMES, ["a", "b", "c"]):  # shared link ?a=...&b=...&c=...
        case = A.decode_case(st.query_params.get(param))
        if case is not None:
            shared = from_example({**U.BORDERLINE, **case})
            if valid(shared):
                rows[n] = shared
    return score(pd.DataFrame.from_dict(rows, orient="index"))


version = int(st.session_state.get("cmp_version", 0))
if "cmp_base" not in st.session_state:
    st.session_state["cmp_base"] = initial_table()
base = st.session_state["cmp_base"]

# Apply the edits of the previous run to the stored table, rescore, and remount the editor with the new outputs
editor_key = f"cmp_editor_{version}"
edits = (st.session_state.get(editor_key) or {}).get("edited_rows") or {}
if edits:
    view_index = st.session_state.get("cmp_view_index", list(base.index))
    for pos, change in edits.items():
        label = view_index[int(pos)]
        for col, value in change.items():
            if col in INPUTS:
                base.at[label, col] = value
    base = score(base)
    st.session_state["cmp_base"] = base
    version += 1
    st.session_state["cmp_version"] = version
    editor_key = f"cmp_editor_{version}"


# ---------------------------------------------------------------------------
# The pill strip: three applicants, copy one applicant into another
# ---------------------------------------------------------------------------

with st.container(key="cmp_controls"):
    c0, c1, c2, c3, c4, c5 = st.columns([1.5, 0.95, 1.6, 0.35, 1.6, 0.9], gap="small", vertical_alignment="center")
    with c0:
        three = st.toggle("Three applicants", value=st.session_state.get("cmp_three", True), key="cmp_three")
    pool = NAMES if three else NAMES[:2]
    c1.caption("Copy applicant")
    src = c2.selectbox("Copy applicant", pool, key="cmp_from", label_visibility="collapsed")
    c3.caption("into")
    dst = c4.selectbox("Into", [n for n in pool if n != src], key="cmp_to", label_visibility="collapsed")
    if c5.button("Copy", use_container_width=True, key="cmp_copy",
                 help="Copies every input of one applicant into another, so you can change one thing at a time."):
        base.loc[dst, INPUTS] = base.loc[src, INPUTS].to_numpy()
        st.session_state["cmp_base"] = score(base)
        st.session_state["cmp_version"] = version + 1
        st.rerun()

shown_names = NAMES if three else NAMES[:2]
if st.session_state.get("btn_row_cmp_edit"):
    st.session_state["cmp_edit"] = not bool(st.session_state.get("cmp_edit"))
editing = bool(st.session_state.get("cmp_edit"))

t = U.shared_threshold()
d = max(U.decline_threshold(), t)
view = base.loc[shown_names].copy()
view["Decision"] = U.decisions(view["PD"].to_numpy(), t, d)  # follows the shared threshold
view["Band"] = U.band_names(view["PD"].to_numpy())
st.session_state["cmp_view_index"] = list(view.index)
names = list(view.index)
feats = st.session_state["cmp_feats"].loc[names]

contribs = {}
for n in names:
    try:
        contribs[n], _ = U.local_contributions(model, feats.loc[[n]])
    except Exception:  # noqa: BLE001
        contribs[n] = None

best = view["Expected value"].idxmax()
lowest = view["PD"].idxmin()

LABELS = {"GrAppv": "Approved (USD)", "Portion": "Guaranteed share", "DisbursementGross": "Disbursed (USD)",
          "Term": "Term (months)", "NoEmp": "Employees", "CreateJob": "Jobs created", "RetainedJob": "Jobs retained",
          "NewExist": "Business age", "IsFranchise": "Franchise", "UrbanRural": "Location", "RevLineCr": "Revolving line",
          "LowDoc": "LowDoc", "Sector": "Sector", "State": "State", "Bank": "Bank", "BankState": "Bank state",
          "Recession": "Recession stress"}


def show(col: str, v) -> str:
    if col in ("GrAppv", "DisbursementGross"):
        return ui.money(float(v), unit="").strip()
    if col == "Portion":
        return f"{float(v):.0%}"
    if col == "Recession":
        return "On" if bool(v) else "Off"
    if col == "Bank":
        return str(v).title()
    return str(v)


diff = [(c, [show(c, view.loc[n, c]) for n in names]) for c in INPUTS
        if len({show(c, view.loc[n, c]) for n in names}) > 1]

# The comparison table: one metric column, one card per applicant, rows aligned across the cards
METRICS = [
    ("Expected net value", lambda r: ui.money(float(r["Expected value"]))),
    ("Expected loss", lambda r: ui.money(float(r["Expected loss"]))),
    ("Approved", lambda r: ui.money(float(r["GrAppv"]))),
    ("Guaranteed share", lambda r: f"{float(r['Portion']):.0%}"),
    ("Term", lambda r: ui.months(int(r["Term"]))),
    ("Employees", lambda r: ui.num(float(r["NoEmp"]))),
    ("Business age", lambda r: "New" if str(r["NewExist"]) == "New" else "Existing"),
    ("Revolving line", lambda r: "Yes" if str(r["RevLineCr"]) == "Y" else "No"),
]


def decision_badge(n: str) -> str:
    """One badge for the decision and the band, in the tone of the decision ("Refer / High risk")."""
    decision = str(view.loc[n, "Decision"])
    word = f"{decision} / {view.loc[n, 'Band']} risk"
    return ui.badge(word, large=True).replace(f"ui-badge {ui.tone(word)}", f"ui-badge {ui.tone(decision)}", 1)


def applicant_card(n: str) -> None:
    r = view.loc[n]
    body = (ui.pair([(f"Applicant {n}", ui.pct(float(r["PD"])), "PD")]) + decision_badge(n)
            + ui.rows([{"a": fmt(r)} for _, fmt in METRICS]))
    ui.card("", None, body, kind="tint" if n == best else "")


def table_editor() -> None:
    U.html(ui.card_html("Inputs of each applicant", "Edit any white cell; the grey columns are computed and rescore.", ""))
    st.data_editor(
        view, key=editor_key, use_container_width=True, num_rows="fixed",
        disabled=OUTPUTS,
        column_order=OUTPUTS + INPUTS,
        column_config={
            "PD": st.column_config.ProgressColumn("PD", format="percent", min_value=0, max_value=1, width="small"),
            "Band": st.column_config.TextColumn("Band", width="small"),
            "Decision": st.column_config.TextColumn("Decision", width="small"),
            "Expected value": st.column_config.NumberColumn("Expected value (USD)", format="localized"),
            "Expected loss": st.column_config.NumberColumn("Expected loss (USD)", format="localized"),
            "GrAppv": st.column_config.NumberColumn("Approved (USD)", min_value=1000, step=5000, format="localized"),
            "Portion": st.column_config.NumberColumn("Guaranteed share", min_value=0.05, max_value=1.0, step=0.05, format="percent"),
            "DisbursementGross": st.column_config.NumberColumn("Disbursed (USD)", min_value=4000, step=5000, format="localized"),
            "Term": st.column_config.NumberColumn("Term (months)", min_value=1, max_value=480, step=6),
            "NoEmp": st.column_config.NumberColumn("Employees", min_value=0, step=1),
            "CreateJob": st.column_config.NumberColumn("Jobs created", min_value=0, step=1),
            "RetainedJob": st.column_config.NumberColumn("Jobs retained", min_value=0, step=1),
            "NewExist": st.column_config.SelectboxColumn("Business age", options=["Existing", "New"], required=True),
            "IsFranchise": st.column_config.SelectboxColumn("Franchise", options=["N", "Y"], required=True),
            "UrbanRural": st.column_config.SelectboxColumn("Location", options=["Urban", "Rural"], required=True),
            "RevLineCr": st.column_config.SelectboxColumn("Revolving line", options=["N", "Y"], required=True),
            "LowDoc": st.column_config.SelectboxColumn("LowDoc", options=["N", "Y"], required=True),
            "Sector": st.column_config.SelectboxColumn("Sector", options=sectors, required=True, width="medium"),
            "State": st.column_config.SelectboxColumn("State", options=states, required=True),
            "Bank": st.column_config.SelectboxColumn("Bank", options=banks, required=True, width="medium"),
            "BankState": st.column_config.SelectboxColumn("Bank state", options=bank_states, required=True),
            "Recession": st.column_config.CheckboxColumn("Recession stress"),
        },
    )


if editing:
    with st.container(key="cmp_edit_rows"):
        table_editor()
else:
    with st.container(key="cmp_table"):
        cols = st.columns([1.05] + [1] * len(names), gap="small")
        with cols[0]:
            ui.card("", None, ui.pair([("Metric", "PD", None)]) + ui.badge("Decision", large=True)
                    + ui.rows([{"a": label} for label, _ in METRICS]))
        for col, n in zip(cols[1:], names):
            with col:
                applicant_card(n)


def drivers_chart() -> None:
    if not all(c is not None for c in contribs.values()):
        U.html(ui.card_html("What drives each score", None,
                            ui.empty("This model has no per-loan explanations: run notebook 05 to write them.")))
        return
    strength = pd.concat([c.set_index("feature")["contribution"].abs().rename(n) for n, c in contribs.items()], axis=1)
    top = strength.max(axis=1).sort_values(ascending=False).head(6).index[::-1]
    colours = {"A": ui.CHART["accent"], "B": ui.CHART["series2"], "C": ui.CHART["series3"]}
    fig = go.Figure()
    for n in names:
        cf = contribs[n].set_index("feature")
        vals = [float(cf.loc[f, "contribution"]) if f in cf.index else 0.0 for f in top]
        shown = [str(cf.loc[f, "value"]) if f in cf.index else "" for f in top]
        fig.add_trace(go.Bar(
            y=[U.short_label(f) for f in top], x=vals, orientation="h", name=f"Applicant {n}",
            marker=dict(color=colours[n]), customdata=shown,
            hovertemplate=f"Applicant {n}<br>%{{y}}: %{{customdata}}<br>Contribution %{{x:+.3f}} log-odds<extra></extra>"))
    fig.add_vline(x=0, line_color=ui.CHART["ink"], line_width=1)
    ui.chart_layout(fig, "What drives each score")
    fig.add_annotation(text=f"Approval threshold {t:.1%}  ·  Decline from {d:.1%}", xref="paper", yref="paper",
                       x=1, y=1, xanchor="right", yanchor="bottom", yshift=14, showarrow=False,
                       font=dict(size=12, color=ui.CHART["ink"]), bgcolor=ui.CHART["canvas"],
                       bordercolor=ui.CHART["hairline"], borderwidth=1, borderpad=6)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06,
                      xaxis=dict(title="Lowers risk  ←  Log-odds  →  Raises risk", zeroline=False),
                      yaxis=dict(automargin=True),
                      legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle", traceorder="normal"))
    with st.container(key="picture_fixed"):
        U.chart(fig, height=330, key="cmp_drivers")


with st.container(key="two"):
    left, right = st.columns([2.6, 1], gap="medium")
    with left:
        drivers_chart()
    with right:
        gap = float(view.loc[best, "Expected value"] - view["Expected value"].drop(best).max()) if len(view) > 1 else 0.0
        with st.container(key="sheet_rows"):
            U.html(ui.card_html("Recommendation", f"Applicant {best}: highest expected net value, "
                                f"{ui.money(gap)} ahead of the next.", ""))
            ui.row_button({"a": "Edit the inputs", "a_small": "Back to the comparison" if editing else "Opens the input table"},
                          key="cmp_edit", chosen=editing)
            if diff:
                c, values = diff[0]
                joined = " / ".join(f"{n} {v}" for n, v in zip(names, values))
                first = f"{LABELS[c]}: {joined}" if len(joined) <= 24 else LABELS[c]
                if ui.row_button({"a": "Differences", "a_small": first, "b": f"{len(diff)} inputs"}, key="cmp_diff"):
                    st.session_state["cmp_sheet"] = "diff"
            else:
                U.html(ui.rows([{"a": "Differences", "a_small": "The applicants have the same inputs. Edit one to compare."}]))
            if ui.row_button({"a": "Share link", "a_small": "Holds every applicant's inputs"}, key="cmp_share"):
                st.session_state["cmp_sheet"] = "share"
            if ui.row_button({"a": "What drives each score", "a_small": "How this is worked out"}, key="cmp_method"):
                st.session_state["cmp_sheet"] = "method"


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

@st.dialog("Share link")
def share_sheet() -> None:
    links = "&".join(f"{p}={A.encode_case({**r.to_dict(), 'SBA_Appv': r['Portion'] * r['GrAppv'], 'Recession': int(bool(r['Recession']))})}"
                     for p, (_, r) in zip(["a", "b", "c"], view.iterrows()))
    host = st.context.headers.get("host", "") if hasattr(st, "context") else ""
    scheme = "http" if host.startswith(("localhost", "127.0.0.1")) else "https"
    U.html(ui.note("The link holds the inputs of each applicant; opening it scores them again.", "accent"))
    st.code(f"{scheme}://{host}/compare?{links}" if host else f"/compare?{links}", language=None)


@st.dialog("What drives each score")
def method_sheet() -> None:
    reasons = [f"Applicant {n}: {U.explanation_sentences(contribs[n], 1)[0]}" for n in names
               if contribs.get(n) is not None and len(contribs[n])]
    if reasons:
        U.html(ui.eyebrow("Main reason for each score") + "".join(ui.note(s, "accent") for s in reasons))
    gap = float(view.loc[best, "Expected value"] - view["Expected value"].drop(best).max()) if len(view) > 1 else 0.0
    tail = "and the lowest PD" if best == lowest else f"; applicant {lowest} has the lowest PD ({view.loc[lowest, 'PD']:.1%})"
    U.html(ui.eyebrow("How this is worked out") + "".join(ui.note(s, "accent") for s in [
        f"Applicant {best} has the highest expected value, {ui.money(gap)} ahead of the next, {tail}.",
        "The grey columns of the input table are computed; edit any other cell to rescore.",
        f"Decisions use the shared approval threshold ({t:.1%}) and the decline level ({d:.1%}).",
        "Expected value = (1 - PD) x gain if repaid - PD x loss if default, from the cost matrix.",
        "Drivers are SHAP contributions in log-odds; the eight strongest inputs across the applicants are shown.",
    ]))


@st.dialog("Differences")
def diff_sheet() -> None:
    table = pd.DataFrame([[LABELS[c]] + values for c, values in diff], columns=["Input"] + [f"Applicant {n}" for n in names])
    st.dataframe(table, hide_index=True, use_container_width=True)


sheet = st.session_state.pop("cmp_sheet", None)
if sheet == "share":
    share_sheet()
elif sheet == "method":
    method_sheet()
elif sheet == "diff" and diff:
    diff_sheet()
