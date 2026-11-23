"""Compare applicants: two or three applications side by side, edited in a live table and rescored on every change."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import analytics as A  # noqa: E402
import experience as X  # noqa: E402
import utils as U  # noqa: E402

model = U.load_model()
X.page_start("compare")
U.page_header(
    "Compare applicants",
    "Put two or three applications side by side. Edit any cell of the table: the probability of default, risk band, "
    "decision and expected value are recomputed straight away, and the charts below follow.",
    eyebrow="Side by side",
)

fields = U.schema_fields()
feat_f = fields["features"]
sectors = sorted(s for s in ((feat_f.get("Sector") or {}).get("allowed") or U.SECTOR_TO_NAICS) if s != "Unknown") + ["Unknown"]
states = sorted(s for s in (feat_f.get("State") or {}).get("allowed", []) if s != "Unknown")
bank_states = sorted(s for s in (feat_f.get("BankState") or {}).get("allowed", []) if s != "Unknown")
banks = [U.NEW_BANK_OPTION] + sorted(b for b in (feat_f.get("Bank") or {}).get("allowed", []) if b != "Unknown")
NAMES = ["A", "B", "C"]
APPLICANT_COLOURS = {"A": U.COLOURS["navy"], "B": "#4C86B8", "C": "#9AA9BF"}
INPUTS = ["GrAppv", "Portion", "DisbursementGross", "Term", "NoEmp", "CreateJob", "RetainedJob", "NewExist", "IsFranchise",
          "UrbanRural", "RevLineCr", "LowDoc", "Sector", "State", "Bank", "BankState", "Recession"]
OUTPUTS = ["PD", "Band", "Decision", "Expected value", "Expected loss"]


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
    out["Decision"] = U.decisions(p)
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

c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1.2], vertical_alignment="bottom")
three = c1.toggle("Compare three applicants", value=st.session_state.get("cmp_three", True), key="cmp_three")
src = c2.selectbox("Copy applicant", NAMES if three else NAMES[:2], key="cmp_from")
dst = c3.selectbox("into", [n for n in (NAMES if three else NAMES[:2]) if n != src], key="cmp_to")
if c4.button("Copy", icon=":material/content_copy:", use_container_width=True, key="cmp_copy",
             help="Copies every input of one applicant into another, so you can change one thing at a time."):
    base.loc[dst, INPUTS] = base.loc[src, INPUTS].to_numpy()
    st.session_state["cmp_base"] = score(base)
    st.session_state["cmp_version"] = version + 1
    st.rerun()

view = base.loc[NAMES if three else NAMES[:2]]
st.session_state["cmp_view_index"] = list(view.index)
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
st.caption("Grey columns are computed. Scroll the table sideways for every input; on a phone, tap a cell to edit it.")

t, d = U.threshold(), U.decline_threshold()
names = list(view.index)
feats_all = st.session_state["cmp_feats"]
feats = feats_all.loc[names]

# Grouped bar of PD and side-by-side gauges
g1, g2 = st.columns([1, 1.35], gap="large")
with g1:
    fig = go.Figure(go.Bar(
        x=[f"Applicant {n}" for n in names], y=view["PD"] * 100,
        marker=dict(color=[U.DECISION_COLOURS[v] for v in view["Decision"]], line=dict(color="#FFFFFF", width=1.5)),
        text=[f"{v:.1%}" for v in view["PD"]], textposition="outside", cliponaxis=False,
        customdata=np.stack([view["Band"], view["Decision"], view["Expected value"]], axis=1),
        hovertemplate="<b>%{x}</b><br>PD %{y:.1f}%<br>%{customdata[0]} risk, %{customdata[1]}"
                      "<br>Expected value USD %{customdata[2]:,.0f}<extra></extra>"))
    fig.add_hline(y=t * 100, line_dash="dot", line_color=U.INK, annotation_text=f"approve below {t:.0%}",
                  annotation_position="bottom right", annotation_font=dict(size=11, color=U.INK))
    fig.add_hline(y=d * 100, line_dash="dash", line_color=U.COLOURS["red"], annotation_text=f"decline from {d:.0%}",
                  annotation_position="top right", annotation_font=dict(size=11, color=U.COLOURS["red"]))
    fig.update_layout(title="Probability of default", yaxis=dict(title="PD (%)", range=[0, max(100 * view["PD"].max() * 1.3, d * 110)]),
                      barcornerradius=8, bargap=0.45, showlegend=False)
    U.chart(fig, height=360, key="cmp_bar")
with g2:
    cols = st.columns(len(names))
    for col, n in zip(cols, names):
        with col:
            r = view.loc[n]
            U.html(f'<div style="text-align:center"><div class="cl-tile-label">Applicant {n}</div>'
                   f'<div class="cl-big-num">{r["PD"]:.1%}</div></div>')
            U.chart(U.gauge_figure(float(r["PD"]), t, compact=True), height=150, key=f"cmp_gauge_{n}")
            U.html(f'<div style="text-align:center;margin-top:-6px">{U.decision_chip(r["Decision"])}</div>')

# Drivers per applicant
U.section_title("What drives each score", "chart")
contribs = {}
for n in names:
    try:
        contribs[n], _ = U.local_contributions(model, feats.loc[[n]])
    except Exception:  # noqa: BLE001
        contribs[n] = None
if all(c is not None for c in contribs.values()):
    strength = pd.concat([c.set_index("feature")["contribution"].abs().rename(n) for n, c in contribs.items()], axis=1)
    top = strength.max(axis=1).sort_values(ascending=False).head(7).index[::-1]
    fig = go.Figure()
    for n in names:
        cf = contribs[n].set_index("feature")
        vals = [float(cf.loc[f, "contribution"]) if f in cf.index else 0.0 for f in top]
        shown = [str(cf.loc[f, "value"]) if f in cf.index else "" for f in top]
        fig.add_trace(go.Bar(
            y=[U.short_label(f) for f in top], x=vals, orientation="h", name=f"Applicant {n}",
            marker=dict(color=APPLICANT_COLOURS[n]), customdata=shown,
            hovertemplate=f"<b>Applicant {n}</b><br>%{{y}}: %{{customdata}}<br>Contribution %{{x:+.3f}} log-odds"
                          "<br>Positive raises the risk<extra></extra>"))
    fig.add_vline(x=0, line_color=U.COLOURS["grey"], line_width=1)
    fig.update_layout(barmode="group", title="Top drivers, contribution to the score (log-odds)", barcornerradius=4,
                      xaxis=dict(title="Lowers the risk   |   Raises the risk"), bargap=0.25, bargroupgap=0.08,
                      margin=dict(l=8, r=8, t=78, b=44))
    U.chart(fig, height=60 * len(top) + 140, key="cmp_drivers")
else:
    st.info("Per-loan explanations are not available for this model.")

# Recommendation table
U.section_title("Recommendation", "check")
best = view["Expected value"].idxmax()
lowest = view["PD"].idxmin()
rec = pd.DataFrame({
    "Applicant": names,
    "PD": view["PD"].to_numpy(),
    "Risk band": view["Band"].to_numpy(),
    "Recommendation": view["Decision"].to_numpy(),
    "Expected value (USD)": view["Expected value"].round(0).to_numpy(),
    "Expected loss (USD)": view["Expected loss"].round(0).to_numpy(),
    "Main reason": [U.explanation_sentences(contribs[n], 1)[0] if contribs.get(n) is not None else "" for n in names],
})
st.dataframe(rec, hide_index=True, use_container_width=True, column_config={
    "PD": st.column_config.ProgressColumn("PD", format="percent", min_value=0, max_value=1),
    "Expected value (USD)": st.column_config.NumberColumn(format="localized"),
    "Expected loss (USD)": st.column_config.NumberColumn(format="localized"),
    "Main reason": st.column_config.TextColumn(width="large"),
})
gap = float(view.loc[best, "Expected value"] - view["Expected value"].drop(best).max()) if len(view) > 1 else 0.0
tail = ("and also has the lowest probability of default" if best == lowest else
        f"while applicant {lowest} has the lowest probability of default")
U.html(f'<div class="cl-insight">{U.icon("spark", 18)}<span>Applicant {best} offers the highest expected value '
       f'({U.fmt_usd(float(view.loc[best, "Expected value"]))}, {U.fmt_usd(gap)} ahead of the next) {tail} '
       f'({view.loc[lowest, "PD"]:.1%}).</span></div>')

# Shareable link
links = "&".join(f"{p}={A.encode_case({**r.to_dict(), 'SBA_Appv': r['Portion'] * r['GrAppv'], 'Recession': int(bool(r['Recession']))})}"
                 for p, (_, r) in zip(["a", "b", "c"], view.iterrows()))
host = st.context.headers.get("host", "") if hasattr(st, "context") else ""
scheme = "http" if host.startswith(("localhost", "127.0.0.1")) else "https"
with st.expander("Share this comparison"):
    st.code(f"{scheme}://{host}/compare?{links}" if host else f"/compare?{links}", language=None)
    st.caption("The link holds the inputs of each applicant; opening it rebuilds this table.")
