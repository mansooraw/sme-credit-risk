"""Batch scoring: upload a CSV of loans, validate, clean, score, summarise and download.

The title block and the figure strip. Before a file is loaded the left column is the Load a file card (the
uploader and the sample toggle) and the figures say "None yet". After loading, the left column is the picture
(the PD distribution, which filters the table, or decisions by sector, or the row editor) with its pill row;
the right column holds the file card with the table filters, the dark panel with the portfolio summary and
the rows with the downloads and the sheets (all loans, row editor, templates); the file line runs along the
bottom.
"""

import hashlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
import utils as U  # noqa: E402
from src import cleaning  # noqa: E402

ui.page("Batch scoring", "Score a CSV of applications in the raw SBA layout or the form layout.",
        eyebrow="Decide", heading="Every file, scored the same way.")

t, d = U.threshold(), U.decline_threshold()
LAYOUT = {"raw": "Raw SBA layout", "form": "Form layout"}
TEMPLATES = [
    (U.TEMPLATE_FORM_PATH, "Form template"),
    (U.SAMPLE_FORM_PATH, "Sample: form layout"),
    (U.TEMPLATE_RAW_PATH, "Raw template"),
    (U.SAMPLE_RAW_PATH, "Sample: raw layout"),
]


@st.cache_data(show_spinner=False)
def read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), keep_default_na=False, na_values=[""], low_memory=False)


def fail(message: str) -> None:
    """Drop the file and show the upload card again with one sentence saying what to do."""
    st.session_state["batch_error"] = message
    st.session_state.pop("batch_src", None)
    st.session_state.pop("batch", None)
    st.rerun()


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

@st.dialog("Templates and layouts", width="large")
def templates_sheet() -> None:
    U.html(ui.note("Raw SBA layout: the columns of SBAnational.csv. Required: " + ", ".join(U.RAW_REQUIRED)
                   + ". Currency may be text such as $60,000.00; outcome columns are ignored.", "accent"))
    U.html(ui.note("Form layout: one column per field of the Score a loan form, listed below.", "accent"))
    U.html(ui.note("Every row is validated and cleaned with the project's cleaning functions; rows that fail are "
                   "listed and left out. Edits to amount, guarantee, term or sector are cleaned and rescored at once.",
                   "accent"))
    cols = st.columns(2)
    for k, (path, label) in enumerate(TEMPLATES):
        if path.exists():
            cols[k % 2].download_button(label, path.read_bytes(), file_name=path.name, mime="text/csv",
                                        use_container_width=True, key=f"dl_{path.stem}")
    st.dataframe(pd.DataFrame({"Column": list(U.FORM_COLUMN_HELP), "Meaning": list(U.FORM_COLUMN_HELP.values())}),
                 hide_index=True, use_container_width=True, height=220)


# ---------------------------------------------------------------------------
# Before a file is loaded: the upload card in the middle
# ---------------------------------------------------------------------------

src = st.session_state.get("batch_src")
if src is None:
    error = st.session_state.pop("batch_error", None)
    ui.kpis([
        ("Loans scored", "None yet", "No file loaded"),
        ("Approval rate", "None yet", f"PD below {t:.0%}"),
        ("Referred", "None yet", f"PD {t:.0%} to {d:.0%}"),
        ("Declined", "None yet", f"PD {d:.0%} and above"),
    ])
    with st.container(key="two"):
        left, right = st.columns([1.2, 1], gap="medium")
        with left:
            with st.container(key="batch_upload_controls"):
                U.html(ui.card_html("Load a file", "Upload a CSV with one loan per row, or use the 50-loan sample.", ""))
                if error:
                    U.html(ui.note(error, "bad"))
                uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Up to 50 MB; one loan per row.",
                                            key="batch_upload")
                use_sample = st.toggle("Use the 50-loan sample", value=st.query_params.get("sample") == "1",
                                       help="Scores app/assets/sample_loans_raw_layout.csv, 50 loans from the "
                                            "held-out test set.")
        with right:
            ui.card("Portfolio summary", "Expected value and loss of the scored file.",
                    ui.empty("The summary appears once a file is scored."), kind="panel")
            with st.container(key="sheet_rows"):
                U.html(ui.card_html("Templates", "Download a template to fill in.", ""))
                if ui.row_button({"a": "Templates and layouts", "a_small": "Four CSV files"}, key="batch_templates"):
                    st.session_state["batch_sheet"] = "templates"
    ui.strip([
        ui.strip_title("Layouts"),
        '<span class="l">Raw SBA layout or form layout · one loan per row · up to 50 MB</span>',
        ui.strip_value("approve below", ui.pct(t)),
        ui.strip_value("decline from", ui.pct(d)),
    ])
    if uploaded is not None:
        st.session_state["batch_src"] = {"data": uploaded.getvalue(), "name": uploaded.name}
        st.rerun()
    if use_sample:
        st.session_state["batch_src"] = {"data": U.SAMPLE_RAW_PATH.read_bytes(), "name": U.SAMPLE_RAW_PATH.name,
                                         "sample": True}
        st.rerun()
    if st.session_state.pop("batch_sheet", None) == "templates":
        templates_sheet()
    st.stop()

# ---------------------------------------------------------------------------
# Read, validate, clean and score (once per file)
# ---------------------------------------------------------------------------

data, name = src["data"], src["name"]
try:
    df = read_csv(data)
except Exception as exc:  # noqa: BLE001
    fail(f"The file could not be read as CSV ({type(exc).__name__}). Check that it is a comma-separated file.")
df.columns = [str(c).strip() for c in df.columns]
if df.empty:
    fail("The file has no rows. Add one loan per row and upload it again.")

layout = U.detect_layout(df)
if layout is None:
    miss = U.missing_columns(df)
    fail("The columns do not match either layout. Missing for the raw layout: " + ", ".join(miss["raw"])
         + ". Missing for the form layout: " + ", ".join(miss["form"]) + ".")

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
        fail(f"The file could not be cleaned ({exc}). Compare it with the templates.")

    reasons = U.validate_clean(clean)
    valid = (reasons == "").to_numpy()
    if not valid.any():
        progress.empty()
        fail("No row passed validation. Compare the file with the templates.")

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

approved = (out["decision"] == "Approve").to_numpy()
counts = out["decision"].value_counts().reindex(["Approve", "Refer", "Decline"], fill_value=0)
pd_range = st.session_state.get("batch_pd_range")
BIN = 0.025
edges = np.arange(0, 1 + BIN, BIN)
COLUMN_ORDER = ["loan_id", "probability_of_default", "risk_band", "decision", "expected_net_value_usd",
                "expected_loss_usd", "GrAppv", "Portion", "DisbursementGross", "Term", "Sector", "State", "Bank"]
sectors = sorted(set(U.SECTOR_TO_NAICS) | set(out["Sector"]))
COLUMN_CONFIG = {
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
}


# ---------------------------------------------------------------------------
# Middle pictures
# ---------------------------------------------------------------------------

def picture_histogram() -> None:
    """PD distribution as clickable bars: a click filters the table to that PD range."""
    hist, _ = np.histogram(out["probability_of_default"].clip(0, 0.99999), bins=edges)
    colours = [ui.CHART["accent"] if pd_range is None or pd_range[0] <= lo < pd_range[1] else ui.CHART["accent_soft"]
               for lo in edges[:-1]]
    fig = go.Figure(go.Bar(x=(edges[:-1] + BIN / 2) * 100, y=hist, width=BIN * 100 * 0.92, marker=dict(color=colours),
                           customdata=np.stack([edges[:-1] * 100, edges[1:] * 100], axis=1),
                           hovertemplate="PD %{customdata[0]:.1f}% to %{customdata[1]:.1f}%<br>%{y} loans"
                                         "<br>Click to filter the table<extra></extra>"))
    fig.add_vline(x=t * 100, line_color=ui.CHART["ink"], line_dash="dot", line_width=1.5)
    fig.add_annotation(x=t * 100, y=1.0, yref="paper", text=f"approve below {t:.0%}", showarrow=False, xanchor="left",
                       xshift=6, yshift=-6, font=dict(size=12, color=ui.CHART["muted"]))
    ui.chart_layout(fig, "PD distribution: click a bar to filter the table")
    fig.update_layout(xaxis=dict(title="Probability of default (%)", range=[0, 100]), yaxis=dict(title="Loans"),
                      showlegend=False)
    with st.container(key="picture_fixed"):
        event = U.chart(fig, height=400, key=f"hist_{st.session_state.get('batch_hist_v', 0)}", on_select="rerun")
    pts = list(event.selection.points) if event is not None and hasattr(event, "selection") else []
    sig = repr(pts)
    if pts and st.session_state.get("_batch_hist_seen") != sig:
        st.session_state["_batch_hist_seen"] = sig
        k = int(pts[0].get("point_index", pts[0].get("point_number", 0)))
        st.session_state["batch_pd_range"] = (float(edges[k]), float(edges[k + 1]))
        st.rerun()


def picture_sectors(focus: str) -> None:
    """Decisions by sector, stacked, with the chosen decision in its tone and the others quiet."""
    by = (out.groupby(["Sector", "decision"]).size().unstack(fill_value=0)
          .reindex(columns=["Approve", "Refer", "Decline"], fill_value=0))
    by = by.loc[by.sort_values([focus, "Approve"]).index].tail(10)
    word = {"Approve": "Approved", "Refer": "Referred", "Decline": "Declined"}
    quiet = {"Approve": ui.CHART["hairline"], "Refer": ui.CHART["canvas"], "Decline": ui.CHART["accent_soft"]}
    fig = go.Figure()
    for k in [focus] + [x for x in ["Approve", "Refer", "Decline"] if x != focus]:
        fig.add_trace(go.Bar(y=[str(x)[:30] for x in by.index], x=by[k], name=word[k], orientation="h",
                             marker=dict(color=ui.tone_colour(word[k]) if k == focus else quiet[k],
                                         line=dict(color=ui.CHART["hairline"], width=0 if k == focus else 1)),
                             hovertemplate=f"%{{y}}<br>{word[k]}: %{{x}} loans<extra></extra>"))
    ui.chart_layout(fig, f"Decisions by sector: {word[focus].lower()} loans")
    fig.update_layout(barmode="stack", xaxis=dict(title="Loans"), yaxis=dict(title=""), margin=dict(l=12))
    with st.container(key="picture_fixed"):
        U.chart(fig, height=400, key="batch_picture")


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------

chosen = st.session_state.get("batch_pick", "loans")
if chosen not in ("loans", "approved", "referred", "declined"):
    chosen = "loans"
editing = bool(st.session_state.get("batch_edit"))
n_edited = len(batch["edited"])
PICKS = [("loans", "PD distribution"), ("approved", "Approved by sector"), ("referred", "Referred by sector"),
         ("declined", "Declined by sector")]


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
<style>body{{font-family:Arial,sans-serif;color:#142f43;margin:0;background:#f1f5f7}}
.page{{max-width:820px;margin:24px auto;background:#fff;border:1px solid #d9e3e8;border-radius:16px;padding:32px 40px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#5f7180;margin:24px 0 8px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .k{{border:1px solid #d9e3e8;border-radius:10px;padding:10px}}
.k span{{display:block;font-size:11px;color:#5f7180;text-transform:uppercase}} .k b{{font-size:18px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #d9e3e8}}
th{{color:#5f7180}} footer{{margin-top:22px;font-size:11px;color:#5f7180}}
@page{{size:A4;margin:16mm}} @media print{{body{{background:#fff}} .page{{border:none;margin:0}}}}</style></head><body><div class="page">
<h1>Portfolio memo: {_html.escape(name)}</h1><div style="color:#5f7180;font-size:13px">{_dt.date.today().strftime('%d %B %Y')},
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


ui.kpis([
    ("Loans scored", f"{len(out):,}", f"Of {batch['n_rows']:,} rows" + (f", {n_edited:,} edited" if n_edited else "")),
    ("Approval rate", ui.pct(approved.mean()), f"{counts['Approve']:,} approved"),
    ("Referred", f"{counts['Refer']:,}", f"PD {t:.0%} to {d:.0%}"),
    ("Declined", f"{counts['Decline']:,}", f"PD {d:.0%} and above"),
])

with st.container(key="two"):
    left, right = st.columns([1.2, 1], gap="medium")

    # The right column first: its filters decide the table that the picture, the editor and the sheets show
    with right:
        with st.container(key="batch_controls"):
            U.html(ui.card_html("File", "Filter the table of scored loans.", ""))
            if ui.row_button({"a": "Sample, 50 loans" if src.get("sample") else name[:26],
                              "a_small": "Load another file"}, key="batch_reset"):
                for k in ["batch_src", "batch", "batch_pd_range", "batch_pick", "batch_edit"]:
                    st.session_state.pop(k, None)
                st.rerun()
            passed = {"a": "Rows passed validation", "b": f"{len(out):,} of {batch['n_rows']:,}"}
            if len(batch["bad"]):
                if ui.row_button({**passed, "a_small": f"{len(batch['bad']):,} left out: see why"}, key="batch_bad"):
                    st.session_state["batch_sheet"] = "bad"
                U.html(ui.rows([{"a": "Layout detected", "b": LAYOUT[layout]}]))
            else:
                U.html(ui.rows([passed, {"a": "Layout detected", "b": LAYOUT[layout]}]))
            U.html(ui.eyebrow("Decisions in the table"))
            f1, f2 = st.columns(2)
            dec_filter = f1.multiselect("Decision", ["Approve", "Refer", "Decline"], key="batch_dec",
                                        placeholder="All decisions", label_visibility="collapsed")
            band_filter = f2.multiselect("Risk band", [b["name"] for b in U.risk_bands()], key="batch_band",
                                         placeholder="All bands", label_visibility="collapsed")
            if pd_range is not None:
                s1, s2 = st.columns([1.4, 1])
                search = s1.text_input("Search", key="batch_search", placeholder="Sector, state or bank",
                                       label_visibility="collapsed")
                if s2.button(f"Clear PD {pd_range[0]:.1%} to {pd_range[1]:.1%}", key="batch_clear_range",
                             use_container_width=True, help="Remove the PD range chosen on the histogram."):
                    st.session_state.pop("batch_pd_range", None)
                    st.session_state["batch_hist_v"] = st.session_state.get("batch_hist_v", 0) + 1
                    st.rerun()
            else:
                search = st.text_input("Search", key="batch_search", placeholder="Sector, state or bank",
                                       label_visibility="collapsed")
            mask = pd.Series(True, index=out.index)
            if dec_filter:
                mask &= out["decision"].isin(dec_filter)
            if band_filter:
                mask &= out["risk_band"].isin(band_filter)
            if search:
                q = search.strip().lower()
                mask &= (out["Sector"].str.lower().str.contains(q, regex=False)
                         | out["State"].str.lower().str.contains(q, regex=False)
                         | out["Bank"].str.lower().str.contains(q, regex=False))
            if pd_range is not None:
                mask &= (out["probability_of_default"] >= pd_range[0]) & (out["probability_of_default"] < pd_range[1])
            view = out[mask]
            st.session_state["batch_view_index"] = list(view.index)
            st.caption(f"Showing {len(view):,} of {len(out):,} loans in the table.")

        ui.card("Portfolio summary", f"{counts['Approve']:,} approved, {counts['Refer']:,} referred, "
                f"{counts['Decline']:,} declined.", ui.rows([
                    {"a": "Expected net value", "a_small": "Approved loans", "b": ui.money(out.loc[approved, "expected_net_value_usd"].sum())},
                    {"a": "Mean PD", "a_small": "All scored loans", "b": ui.pct(out["probability_of_default"].mean())},
                    {"a": "Expected loss avoided", "a_small": "Referred and declined", "b": ui.money(out.loc[~approved, "expected_loss_usd"].sum())},
                ]), kind="panel")

        with st.container(key="sheet_rows"):
            U.html(ui.card_html("Downloads and sheets", None, ""))
            with st.container(key="row_dl_scored"):
                st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"),
                                   file_name=f"creditlens_scored_{Path(name).stem}.csv", mime="text/csv", type="primary",
                                   key="dl_scored", use_container_width=True)
            with st.container(key="row_dl_memo"):
                st.download_button("Download portfolio memo", portfolio_memo().encode("utf-8"),
                                   file_name=f"creditlens_portfolio_memo_{Path(name).stem}.html", mime="text/html",
                                   key="dl_memo", use_container_width=True)
            if ui.row_button({"a": "See all loans", "a_small": f"{len(view):,} in the table"}, key="batch_all"):
                st.session_state["batch_sheet"] = "all"
            if ui.row_button({"a": "Show the chart" if editing else "Edit rows",
                              "a_small": "Back to the picture" if editing else "Amount, guarantee, term, sector"},
                             key="batch_edit_rows", chosen=editing):
                st.session_state["batch_edit"] = not editing
                st.rerun()
            if ui.row_button({"a": "Templates and layouts", "a_small": "Four CSV files"}, key="batch_templates"):
                st.session_state["batch_sheet"] = "templates"

    with left:
        if editing:
            with st.container(key="picture_fixed"):
                st.data_editor(view, key=editor_key, hide_index=True, use_container_width=True, num_rows="fixed",
                               height=400, disabled=[c for c in view.columns if c not in EDITABLE],
                               column_order=COLUMN_ORDER, column_config=COLUMN_CONFIG)
        elif chosen == "loans":
            picture_histogram()
        else:
            picture_sectors({"approved": "Approve", "referred": "Refer", "declined": "Decline"}[chosen])

        with st.container(key="pick_buttons"):
            cols = st.columns(len(PICKS))
            for col, (key, label) in zip(cols, PICKS):
                if col.button(label, key=f"btn_figures_{key}",
                              type="primary" if key == chosen and not editing else "secondary",
                              use_container_width=True):
                    st.session_state["batch_pick"] = key
                    st.session_state.pop("batch_edit", None)
                    st.rerun()

ui.strip([
    ui.strip_title("File"),
    ui.strip_value("rows passed validation", f"{len(out):,} of {batch['n_rows']:,}"),
    ui.strip_value("edited", f"{n_edited:,}"),
    f'<span class="l">{ui.esc("Sample, 50 loans" if src.get("sample") else name)} · {LAYOUT[layout]} · '
    f'approve below {t:.1%} · decline from {d:.1%}</span>',
])


# ---------------------------------------------------------------------------
# Sheets on the loaded file
# ---------------------------------------------------------------------------

@st.dialog("Scored loans", width="large")
def all_sheet() -> None:
    U.html(ui.note(f"Showing {len(view):,} of {len(out):,} loans with the filters of the File card. "
                   "Use Edit rows to change a loan and rescore it.", "accent"))
    st.dataframe(view, hide_index=True, use_container_width=True, height=440, column_order=COLUMN_ORDER,
                 column_config=COLUMN_CONFIG)


@st.dialog("Rows left out", width="large")
def bad_sheet() -> None:
    U.html(ui.note(f"{len(batch['bad']):,} rows cannot be scored and are left out. Fix them and upload the file again.", "warning"))
    st.dataframe(batch["bad"], hide_index=True, use_container_width=True, height=360)


sheet = st.session_state.pop("batch_sheet", None)
if sheet == "all":
    all_sheet()
elif sheet == "bad":
    bad_sheet()
elif sheet == "templates":
    templates_sheet()
