"""Batch scoring: upload a CSV of loans, validate, clean, score, summarise and download."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import experience as X  # noqa: E402
import utils as U  # noqa: E402
from src import cleaning  # noqa: E402

X.page_start("batch")
U.page_header(
    "Batch scoring",
    "Upload a CSV of applications in the raw SBA column layout or the CreditLens form layout. Every row is "
    "validated, cleaned with the project's cleaning functions and scored with the same model.",
    eyebrow="Portfolio of applications",
)

with st.expander("File layouts, templates and sample files"):
    st.markdown(
        "**Raw SBA layout**: the columns of the source file SBAnational.csv. Required: "
        + ", ".join(f"`{c}`" for c in U.RAW_REQUIRED)
        + ". Optional: `LoanNr_ChkDgt`, `Name`, `City`, `Zip`, `DisbursementDate`. Currency may be text such as "
        "`$60,000.00` and dates may be in the source format (`28-Feb-97`) or ISO format. Outcome columns are ignored."
    )
    st.markdown("**Form layout**: one column per field of the Score a loan form.")
    st.dataframe(pd.DataFrame({"Column": list(U.FORM_COLUMN_HELP), "Meaning": list(U.FORM_COLUMN_HELP.values())}),
                 hide_index=True, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, path, label in [
        (c1, U.TEMPLATE_FORM_PATH, "Form template"),
        (c2, U.SAMPLE_FORM_PATH, "Sample: form layout"),
        (c3, U.TEMPLATE_RAW_PATH, "Raw template"),
        (c4, U.SAMPLE_RAW_PATH, "Sample: raw layout"),
    ]:
        if path.exists():
            col.download_button(label, path.read_bytes(), file_name=path.name, mime="text/csv",
                                use_container_width=True, key=f"dl_{path.stem}")

uploaded = st.file_uploader("Upload a CSV file of loan applications", type=["csv"],
                            help="Up to 50 MB; one loan per row.")
use_sample = st.toggle("Use the 50-loan sample file instead", value=st.query_params.get("sample") == "1",
                       help="Scores app/assets/sample_loans_raw_layout.csv, 50 loans from the held-out test set.")


@st.cache_data(show_spinner=False)
def read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), keep_default_na=False, na_values=[""], low_memory=False)


if uploaded is not None:
    data, name = uploaded.getvalue(), uploaded.name
elif use_sample:
    data, name = U.SAMPLE_RAW_PATH.read_bytes(), U.SAMPLE_RAW_PATH.name
else:
    U.html(f'<div class="cl-card cl-empty">{U.icon("upload", 40, "#8D99AE")}<h4 style="margin-top:12px">'
           "Waiting for a file</h4><p>Upload a CSV above, or switch on the sample file to see a scored portfolio.</p></div>")
    st.stop()

try:
    df = read_csv(data)
except Exception as exc:  # noqa: BLE001
    st.error(f"The file could not be read as CSV ({type(exc).__name__}).")
    st.stop()
df.columns = [str(c).strip() for c in df.columns]
if df.empty:
    st.error("The file has no rows.")
    st.stop()

layout = U.detect_layout(df)
if layout is None:
    miss = U.missing_columns(df)
    st.error("The columns do not match either layout. Missing for the raw layout: "
             + ", ".join(miss["raw"]) + ". Missing for the form layout: " + ", ".join(miss["form"]) + ".")
    st.stop()

import hashlib  # noqa: E402

signature = hashlib.sha1(data).hexdigest() + layout
batch = st.session_state.get("batch")
if batch is None or batch["sig"] != signature:
    progress = st.progress(0.0, text="Cleaning the file")
    params = U.load_cleaning_params()
    try:
        if layout == "raw":
            clean = U.raw_to_clean(df, params)
            ids = df["LoanNr_ChkDgt"] if "LoanNr_ChkDgt" in df else pd.Series(np.arange(1, len(df) + 1))
        else:
            clean, _ = cleaning.handle_outliers(U.form_to_clean(df), params=params)
            ids = df["LoanID"] if "LoanID" in df else pd.Series(np.arange(1, len(df) + 1))
    except Exception as exc:  # noqa: BLE001
        progress.empty()
        st.error(f"The file could not be cleaned: {exc}")
        st.stop()

    reasons = U.validate_clean(clean)
    valid = (reasons == "").to_numpy()
    if not valid.any():
        progress.empty()
        st.error("No row passed validation.")
        st.stop()

    progress.progress(0.2, text="Building features")
    model = U.load_model()
    feats = U.build_model_frame(clean[valid])
    chunks = np.array_split(np.arange(len(feats)), min(10, len(feats)))
    parts = []
    for k, idx in enumerate(chunks):
        parts.append(U.predict(model, feats.iloc[idx]))
        progress.progress(0.2 + 0.8 * (k + 1) / len(chunks), text=f"Scoring loans: {idx[-1] + 1:,} of {len(feats):,}")
    p = np.concatenate(parts)
    progress.empty()

    scores = U.score_table(feats, p)
    out = pd.DataFrame({"loan_id": ids[valid].to_numpy()}, index=feats.index)
    for col in ["GrAppv", "DisbursementGross", "Portion", "Term", "Sector", "State", "Bank"]:
        out[col] = feats[col].to_numpy()
    for col in scores.columns:
        out[col] = scores[col].to_numpy()
    out[["expected_net_value_usd", "expected_loss_usd"]] = out[["expected_net_value_usd", "expected_loss_usd"]].round(0)
    bad = pd.DataFrame({"Row": np.flatnonzero(~valid) + 2, "Loan": ids[~valid].to_numpy(), "Reason": reasons[~valid].to_numpy()})
    batch = {"sig": signature, "clean": clean[valid].copy(), "out": out, "bad": bad, "n_rows": len(df), "version": 0,
             "edited": set()}
    st.session_state["batch"] = batch

out, clean = batch["out"], batch["clean"]
EDITABLE = ["GrAppv", "Portion", "DisbursementGross", "Term", "Sector"]
OUT_COLS = ["probability_of_default", "risk_band", "decision", "expected_net_value_usd", "expected_loss_usd"]


def rescore(labels) -> None:
    """Rebuild features and scores for edited rows (same cleaning and features as the upload)."""
    feats = U.build_model_frame(clean.loc[labels])
    p = U.predict(U.load_model(), feats)
    sc = U.score_table(feats, p)
    for col in ["GrAppv", "DisbursementGross", "Portion", "Term", "Sector"]:
        out.loc[labels, col] = feats[col].to_numpy()
    for col in OUT_COLS:
        out.loc[labels, col] = sc[col].to_numpy()
    out.loc[labels, ["expected_net_value_usd", "expected_loss_usd"]] = out.loc[labels, ["expected_net_value_usd", "expected_loss_usd"]].round(0)


editor_key = f"batch_editor_{batch['version']}"
edits = (st.session_state.get(editor_key) or {}).get("edited_rows") or {}
if edits:
    view_index = st.session_state.get("batch_view_index", list(out.index))
    changed = []
    for pos, change in edits.items():
        label = view_index[int(pos)]
        for col, value in change.items():
            if value is None:
                continue
            if col == "GrAppv":
                share = float(clean.at[label, "SBA_Appv"]) / float(clean.at[label, "GrAppv"])
                clean.at[label, "GrAppv"] = max(float(value), 1000.0)
                clean.at[label, "SBA_Appv"] = share * clean.at[label, "GrAppv"]
            elif col == "Portion":
                clean.at[label, "SBA_Appv"] = float(np.clip(value, 0.0, 1.0)) * float(clean.at[label, "GrAppv"])
            elif col == "DisbursementGross":
                clean.at[label, "DisbursementGross"] = max(float(value), 1.0)
            elif col == "Term":
                clean.at[label, "Term"] = int(np.clip(value, 1, 480))
            elif col == "Sector":
                clean.at[label, "NAICS"] = U.SECTOR_TO_NAICS.get(str(value), 0)
        changed.append(label)
    rescore(changed)
    batch["edited"].update(changed)
    batch["version"] += 1
    editor_key = f"batch_editor_{batch['version']}"

st.success(f"Read {batch['n_rows']:,} rows from {name} in the {'raw SBA' if layout == 'raw' else 'form'} layout. "
           f"{len(out):,} rows passed validation.")
if len(batch["bad"]):
    st.warning(f"{len(batch['bad']):,} rows cannot be scored and are left out.")
    st.dataframe(batch["bad"], hide_index=True, use_container_width=True)

t = U.threshold()
approved = (out["decision"] == "Approve").to_numpy()
counts = out["decision"].value_counts().reindex(["Approve", "Refer", "Decline"], fill_value=0)
U.section_title("Portfolio summary", "grid")
if batch["edited"]:
    st.caption(f"{len(batch['edited']):,} loans edited in the table below; the summary includes the edits.")
U.tiles([
    {"label": "Loans scored", "value": f"{len(out):,}",
     "sub": f"Approve {counts['Approve']:,}, refer {counts['Refer']:,}, decline {counts['Decline']:,}", "accent": U.COLOURS["navy"],
     "help": "Rows that passed validation and were scored with the loaded model."},
    {"label": "Approval rate", "value": f"{approved.mean():.1%}", "sub": f"PD below {t:.1%}", "accent": U.COLOURS["teal"],
     "help": f"Loans with a probability of default below the approval threshold of {t:.1%}, divided by all scored loans."},
    {"label": "Expected default rate", "value": f"{out.loc[approved, 'probability_of_default'].mean():.1%}" if approved.any() else "n/a",
     "sub": "Average PD of approved loans", "accent": U.COLOURS["amber"],
     "help": "Mean predicted probability of default of the approved loans."},
    {"label": "Expected net value", "value": U.fmt_usd(out.loc[approved, "expected_net_value_usd"].sum()),
     "sub": f"Expected loss {U.fmt_usd(out.loc[approved, 'expected_loss_usd'].sum())}", "accent": U.COLOURS["teal"],
     "help": "Sum over approved loans of (1 - PD) x margin x disbursed minus PD x loss if charged off."},
])

# Histogram as clickable bars: a click filters the table to that PD range
BIN = 0.025
edges = np.arange(0, 1 + BIN, BIN)
hist, _ = np.histogram(out["probability_of_default"].clip(0, 0.99999), bins=edges)
centres = edges[:-1] + BIN / 2
pd_range = st.session_state.get("batch_pd_range")
c1, c2, c3 = st.columns([1.5, 1, 1], gap="medium")
with c1:
    colours = []
    for lo in edges[:-1]:
        b = U.band_for(lo + BIN / 2)
        sel = pd_range is not None and pd_range[0] <= lo < pd_range[1]
        colours.append(U.COLOURS["navy"] if sel or pd_range is None else U.hex_rgba(U.COLOURS["navy"], 0.25))
    fig = go.Figure()
    for b in U.risk_bands():
        fig.add_vrect(x0=b["min"] * 100, x1=min(b["max"], 1) * 100, fillcolor=b["colour"], opacity=0.10, line_width=0,
                      layer="below")
    fig.add_trace(go.Bar(x=centres * 100, y=hist, width=BIN * 100 * 0.92, marker=dict(color=colours),
                         customdata=np.stack([edges[:-1] * 100, edges[1:] * 100], axis=1),
                         hovertemplate="PD %{customdata[0]:.1f}% to %{customdata[1]:.1f}%<br>%{y} loans"
                                       "<br>Click to filter the table<extra></extra>"))
    fig.add_vline(x=t * 100, line_color=U.INK, line_dash="dot", annotation_text=f"threshold {t:.1%}",
                  annotation_position="top right", annotation_font=dict(size=11, color=U.INK))
    fig.update_layout(title="Distribution of probability of default", xaxis=dict(title="Probability of default (%)", range=[0, 100]),
                      yaxis_title="Loans", barcornerradius=3, showlegend=False)
    event = U.chart(fig, height=340, key=f"hist_{st.session_state.get('batch_hist_v', 0)}", on_select="rerun")
    pts = list(event.selection.points) if event is not None and hasattr(event, "selection") else []
    sig = repr(pts)
    if pts and st.session_state.get("_batch_hist_seen") != sig:
        st.session_state["_batch_hist_seen"] = sig
        k = int(pts[0].get("point_index", pts[0].get("point_number", 0)))
        st.session_state["batch_pd_range"] = (float(edges[k]), float(edges[k + 1]))
        st.rerun()
with c2:
    bands = U.risk_bands()
    band_counts = out["risk_band"].value_counts().reindex([b["name"] for b in bands], fill_value=0)
    fig = go.Figure(go.Pie(labels=band_counts.index, values=band_counts.values, hole=0.62, sort=False,
                           marker=dict(colors=[b["colour"] for b in bands], line=dict(color="#FFFFFF", width=3)),
                           textinfo="percent", textfont=dict(size=12),
                           hovertemplate="<b>%{label}</b><br>%{value} loans, %{percent}<extra></extra>"))
    fig.add_annotation(text=f"<b>{len(out):,}</b><br><span style='font-size:11px'>loans</span>", showarrow=False,
                       font=dict(size=20, color=U.INK))
    fig.update_layout(title="Risk bands", showlegend=True, margin=dict(l=8, r=8, t=44, b=8),
                      legend=dict(orientation="h", y=-0.05, yanchor="top", x=0.5, xanchor="center"))
    U.chart(fig, height=340, key="donut")
with c3:
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.62, sort=False,
                           marker=dict(colors=[U.DECISION_COLOURS[k] for k in counts.index], line=dict(color="#FFFFFF", width=3)),
                           textinfo="percent", textfont=dict(size=12),
                           customdata=[out.loc[out["decision"] == k, "expected_net_value_usd"].sum() for k in counts.index],
                           hovertemplate="<b>%{label}</b><br>%{value} loans, %{percent}<br>Expected value "
                                         "USD %{customdata:,.0f}<extra></extra>"))
    fig.add_annotation(text=f"<b>{approved.mean():.0%}</b><br><span style='font-size:11px'>approved</span>", showarrow=False,
                       font=dict(size=20, color=U.INK))
    fig.update_layout(title="Decisions", showlegend=True, margin=dict(l=8, r=8, t=44, b=8),
                      legend=dict(orientation="h", y=-0.05, yanchor="top", x=0.5, xanchor="center"))
    U.chart(fig, height=340, key="decision_donut")

U.section_title("Scored loans", "file")
f1, f2, f3 = st.columns([1.2, 1.2, 1.3])
dec_filter = f1.multiselect("Decision", ["Approve", "Refer", "Decline"], key="batch_dec",
                            placeholder="All decisions")
band_filter = f2.multiselect("Risk band", [b["name"] for b in U.risk_bands()], key="batch_band", placeholder="All bands")
search = f3.text_input("Search sector, state or bank", key="batch_search", placeholder="For example Retail or FL")
mask = pd.Series(True, index=out.index)
if dec_filter:
    mask &= out["decision"].isin(dec_filter)
if band_filter:
    mask &= out["risk_band"].isin(band_filter)
if search:
    q = search.strip().lower()
    mask &= (out["Sector"].str.lower().str.contains(q, regex=False) | out["State"].str.lower().str.contains(q, regex=False)
             | out["Bank"].str.lower().str.contains(q, regex=False))
if pd_range is not None:
    mask &= (out["probability_of_default"] >= pd_range[0]) & (out["probability_of_default"] < pd_range[1])
    r1, r2 = st.columns([5, 1], vertical_alignment="center")
    r1.markdown(f'<div class="cl-filterbar"><span class="cl-filter-label">Filtered:</span>'
                f'{U.chip(f"PD {pd_range[0]:.1%} to {pd_range[1]:.1%}", U.COLOURS["navy"])}</div>', unsafe_allow_html=True)
    if r2.button("Clear", key="batch_clear_range", icon=":material/filter_alt_off:", use_container_width=True):
        st.session_state.pop("batch_pd_range", None)
        st.session_state["batch_hist_v"] = st.session_state.get("batch_hist_v", 0) + 1
        st.rerun()
view = out[mask]
st.session_state["batch_view_index"] = list(view.index)
sectors = sorted(set(U.SECTOR_TO_NAICS) | set(out["Sector"]))
st.data_editor(
    view, key=editor_key, hide_index=True, use_container_width=True, num_rows="fixed",
    height=min(38 * len(view) + 40, 520), disabled=[c for c in view.columns if c not in EDITABLE],
    column_order=["loan_id", "probability_of_default", "risk_band", "decision", "expected_net_value_usd",
                  "expected_loss_usd", "GrAppv", "Portion", "DisbursementGross", "Term", "Sector", "State", "Bank"],
    column_config={
        "loan_id": "Loan",
        "probability_of_default": st.column_config.ProgressColumn("PD", format="percent", min_value=0, max_value=1),
        "risk_band": "Risk band", "decision": "Decision",
        "Portion": st.column_config.NumberColumn("Guaranteed share", format="percent", min_value=0.0, max_value=1.0, step=0.05),
        "GrAppv": st.column_config.NumberColumn("Approved (USD)", format="localized", min_value=1000, step=5000),
        "DisbursementGross": st.column_config.NumberColumn("Disbursed (USD)", format="localized", min_value=1, step=5000),
        "Term": st.column_config.NumberColumn("Term (months)", min_value=1, max_value=480, step=6),
        "Sector": st.column_config.SelectboxColumn("Sector", options=sectors, required=True),
        "expected_net_value_usd": st.column_config.NumberColumn("Expected net value (USD)", format="localized"),
        "expected_loss_usd": st.column_config.NumberColumn("Expected loss (USD)", format="localized"),
    },
)
st.caption(f"Showing {len(view):,} of {len(out):,} loans. Click a column header to sort. Edit the approved amount, "
           "guarantee, disbursed amount, term or sector of any loan: it is cleaned and rescored at once, and the tiles "
           "and charts above follow. Click a bar of the histogram to filter by PD.")


def portfolio_memo() -> str:
    import datetime as _dt
    import html as _html
    sec = (out.assign(app=out["decision"] == "Approve")
           .groupby("Sector").agg(loans=("loan_id", "size"), pd_mean=("probability_of_default", "mean"),
                                  el=("expected_loss_usd", "sum")).sort_values("el", ascending=False).head(8))
    rows = "".join(f"<tr><td>{_html.escape(str(k))}</td><td>{r.loans:,}</td><td>{r.pd_mean:.1%}</td><td>{U.fmt_usd(r.el)}</td></tr>"
                   for k, r in sec.iterrows())
    bands = out["risk_band"].value_counts().reindex([b["name"] for b in U.risk_bands()], fill_value=0)
    band_rows = "".join(f"<tr><td>{k}</td><td>{v:,}</td><td>{v / len(out):.1%}</td></tr>" for k, v in bands.items())
    meta = U.load_metadata()
    return f"""<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><title>CreditLens portfolio memo</title>
<style>body{{font-family:Arial,sans-serif;color:#0B1F3A;margin:0;background:#F7F9FC}}
.page{{max-width:820px;margin:24px auto;background:#fff;border:1px solid #E3E8EF;border-radius:14px;padding:32px 40px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#5B6B82;margin:24px 0 8px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .k{{border:1px solid #E3E8EF;border-radius:10px;padding:10px}}
.k span{{display:block;font-size:11px;color:#5B6B82;text-transform:uppercase}} .k b{{font-size:18px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #EEF2F7}}
th{{color:#5B6B82}} footer{{margin-top:22px;font-size:11px;color:#8D99AE}}
@page{{size:A4;margin:16mm}} @media print{{body{{background:#fff}} .page{{border:none;margin:0}}}}</style></head><body><div class="page">
<h1>Portfolio memo: {_html.escape(name)}</h1><div style="color:#5B6B82;font-size:13px">{_dt.date.today().strftime('%d %B %Y')},
model {_html.escape(str(meta.get('model_name', 'n/a')))}, approve below {t:.1%}</div>
<h2>Summary</h2><div class="grid">
<div class="k"><span>Loans scored</span><b>{len(out):,}</b></div><div class="k"><span>Approval rate</span><b>{approved.mean():.1%}</b></div>
<div class="k"><span>Expected net value</span><b>{U.fmt_usd(out.loc[approved, 'expected_net_value_usd'].sum())}</b></div>
<div class="k"><span>Expected loss</span><b>{U.fmt_usd(out.loc[approved, 'expected_loss_usd'].sum())}</b></div></div>
<p style="font-size:13px">Approve {counts['Approve']:,}, refer {counts['Refer']:,}, decline {counts['Decline']:,}.
{len(batch['bad']):,} rows failed validation; {len(batch['edited']):,} loans were edited in the app before this memo.</p>
<h2>Risk bands</h2><table><tr><th>Band</th><th>Loans</th><th>Share</th></tr>{band_rows}</table>
<h2>Sectors with the largest expected loss</h2><table><tr><th>Sector</th><th>Loans</th><th>Mean PD</th><th>Expected loss</th></tr>{rows}</table>
<footer>CreditLens is a decision-support tool; every referral and decline is reviewed by a credit officer.</footer></div></body></html>"""


d1, d2 = st.columns(2)
d1.download_button("Download scored CSV", out.to_csv(index=False).encode("utf-8"),
                   file_name=f"creditlens_scored_{Path(name).stem}.csv", mime="text/csv", type="primary", key="dl_scored",
                   use_container_width=True, icon=":material/download:")
d2.download_button("Download portfolio memo", portfolio_memo().encode("utf-8"),
                   file_name=f"creditlens_portfolio_memo_{Path(name).stem}.html", mime="text/html", key="dl_memo",
                   use_container_width=True, icon=":material/description:")
