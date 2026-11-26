"""CreditLens component kit: the only parts a page may place on its board.

Each function emits one agreed piece of markup (or one Streamlit widget
arrangement) styled by app/assets/theme.css from the values in
app/assets/tokens.css. A page composes these; it never restyles them and
never writes a colour, a size or a status word of its own. See
design-system/MASTER.md.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import json

import streamlit as st

import utils as U

# ---------------------------------------------------------------------------
# One status map: every status word on screen maps to one of four tones
# ---------------------------------------------------------------------------

STATUS = {
    "approve": "good", "approved": "good", "repaid": "good", "gain": "good", "low": "good", "pass": "good",
    "paid in full": "good", "on target": "good", "met": "good",
    "refer": "warning", "referred": "warning", "review": "warning", "medium": "warning", "caution": "warning",
    "below minimum": "warning", "stale": "warning",
    "decline": "bad", "declined": "bad", "default": "bad", "defaulted": "bad", "charged off": "bad", "loss": "bad",
    "high": "bad", "very high": "bad", "missed": "bad", "fail": "bad", "raises risk": "bad", "lowers risk": "good",
    "repaid loan": "good", "charged off loan": "bad",
    "none": "neutral", "no comparison": "neutral", "unknown": "neutral", "not recorded": "neutral", "n/a": "neutral",
    "pending": "neutral", "base case": "neutral", "model": "neutral", "officer": "neutral",
}
TONE_COLOUR = {"good": "#0e6b52", "warning": "#8a5a00", "bad": "#b23a2e", "neutral": "#10302b"}
CHART = {
    "accent": "#0f8b6c", "accent_deep": "#0a5f4b", "accent_light": "#7fcdb6", "accent_soft": "#ddf6ee",
    "ink": "#10302b", "muted": "#6b7c7e", "hairline": "rgba(16, 48, 43, 0.12)", "canvas": "#eef3f3",
    "good": "#0e6b52", "warning": "#c58a1a", "bad": "#d2503f", "series2": "#2563eb", "series3": "#9aa8ad",
}
SERIES = ["#0f8b6c", "#2563eb", "#9aa8ad", "#7fcdb6", "#c58a1a"]
SEQUENTIAL = [[0, "#ddf6ee"], [0.5, "#7fcdb6"], [1, "#0a5f4b"]]
DIVERGING = [[0, "#0e6b52"], [0.5, "#eef3f3"], [1, "#b23a2e"]]


def tone(word: str) -> str:
    return STATUS.get(str(word).strip().lower(), "neutral")


def tone_colour(word: str) -> str:
    return TONE_COLOUR[tone(word)]


# ---------------------------------------------------------------------------
# Formatting: money, dates and units in one place
# ---------------------------------------------------------------------------

def money(x: float, unit: str = "USD") -> str:
    """USD 1,250 / USD 1.3 m / USD 2.1 bn; always with the unit."""
    x = float(x)
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e9:
        return f"{sign}{unit} {a / 1e9:,.1f} bn"
    if a >= 1e6:
        return f"{sign}{unit} {a / 1e6:,.1f} m"
    return f"{sign}{unit} {a:,.0f}"


def pct(x: float, decimals: int = 1) -> str:
    return f"{float(x) * 100:,.{decimals}f}%"


def num(x: float, decimals: int = 0) -> str:
    return f"{float(x):,.{decimals}f}"


def months(n: int) -> str:
    n = int(n)
    return f"{n} month" if n == 1 else f"{n} months"


def day(d) -> str:
    """Dates in words: Mon 3 Nov."""
    if isinstance(d, str):
        d = _dt.date.fromisoformat(d[:10])
    return d.strftime("%a %-d %b")


def esc(text) -> str:
    return _html.escape(str(text))


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

_HOME_SVG = ('<svg class="home" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
             'stroke-linejoin="round" aria-hidden="true"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h5v-6h4v6h5V10"/></svg>')


def status_pills() -> str:
    """The three status pills of the page bar: model, held-out loans, approve-below."""
    meta = U.load_metadata()
    name = str(meta.get("model_name", "Model"))
    short = "LightGBM" if "lightgbm" in name.lower() else name.split("(")[0].strip()
    pills = [f'<span class="ui-pill solid">{esc(short)}</span>']
    if isinstance(meta.get("n_test"), int):
        pills.append(f'<span class="ui-pill quiet">{meta["n_test"]:,} held-out loans</span>')
    pills.append(f'<span class="ui-pill">Approve below {U.threshold():.1%}</span>')
    return "".join(pills)


def bar(title: str, line: str | None = None, meta: str | None = None) -> None:
    """The glass page bar: home, page name, and the status pills on the right."""
    pills = status_pills() if meta is None else f'<span class="ui-pill">{meta}</span>'
    U.html(f'<div class="ui-bar ui-arrive"><div class="crumb">{_HOME_SVG}<span class="sep">&rsaquo;</span><span>{esc(title)}</span></div>'
           f'<div class="pills">{pills}</div></div>')


def title(eyebrow: str, heading: str, sub: str, art: bool = True, small: bool = False) -> None:
    """The editorial title block: eyebrow with rule, display heading, one-line subtitle, ring art on the right."""
    ring = '<div class="ui-art" aria-hidden="true"><div class="base"></div><div class="ring2"></div><div class="ring"></div></div>' if art else ""
    U.html(f'<div class="ui-title ui-arrive"><div><div class="eyebrow">{esc(eyebrow)}</div>'
           f'<div class="h{" sm" if small else ""}">{esc(heading)}</div><div class="s">{esc(sub)}</div></div>'
           f'<div class="right">{ring}</div></div>')


def kpis(items: list[tuple[str, str, str | None]], small: bool = False) -> None:
    """The figure strip: (label, value, meta) side by side, separated by hairlines."""
    cells = "".join(f'<div><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div>' + (f'<div class="m">{esc(m)}</div>' if m else "") + "</div>"
                    for l, v, m in items)
    U.html(f'<div class="ui-kpis ui-arrive{" small" if small else ""}" style="--n:{len(items)}">{cells}</div>')


def strip(parts: list[str]) -> None:
    """A pill-shaped strip of parts (markup strings), separated by hairlines."""
    U.html(f'<div class="ui-strip ui-arrive">{"".join(f"<div>{p}</div>" for p in parts)}</div>')


def strip_value(label: str, value: str) -> str:
    return f'<span class="v">{esc(value)}</span><span class="l">{esc(label)}</span>'


def strip_title(text: str) -> str:
    return f'<span class="t">{esc(text)}</span>'


def steps(n: int, done: int, current: int | None) -> str:
    """A stepper 01 to n: done steps filled, the current one highlighted."""
    out = []
    for i in range(1, n + 1):
        cls = "done" if i <= done else ("now" if i == current else "")
        if i > 1:
            out.append(f'<i class="{"done" if i <= done else ""}"></i>')
        out.append(f'<span class="{cls}">{i:02d}</span>')
    return f'<div class="ui-steps">{"".join(out)}</div>'


def facts(items: list[tuple[str, str]], columns: int = 4) -> str:
    """A grid of labelled facts (label, value)."""
    cells = "".join(f'<div><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div></div>' for l, v in items)
    return f'<div class="ui-facts{" n3" if columns == 3 else ""}" style="--n:{columns}">{cells}</div>'


def stack(parts: list[tuple[str, float, str]]) -> str:
    """A stacked share bar: (tone, share, label)."""
    spans = "".join(f'<span class="{tone_}" style="flex:{max(share, 0.001):.4f}">{esc(label) if share > 0.06 else ""}</span>'
                    for tone_, share, label in parts)
    return f'<div class="ui-stack">{spans}</div>'


def card(title: str, line: str | None = None, body: str = "", key: str | None = None, kind: str = "") -> None:
    """A card: title says what it is, first line says what to do, then the body markup.

    ``kind`` is "" (glass), "panel" (dark emerald) or "tint" (mint glass).
    """
    U.html(card_html(title, line, body, kind))


def card_html(title: str, line: str | None, body: str, kind: str = "") -> str:
    d = f'<div class="d">{esc(line)}</div>' if line else ""
    h = f'<div class="h">{esc(title)}</div>' if title else ""
    return f'<div class="ui-card ui-arrive {kind}">{h}{d}{body}</div>'


def badge(word: str, large: bool = False, tone_: str | None = None) -> str:
    """A tinted pill with a dot and the status word; the tone comes from STATUS unless ``tone_`` is given."""
    return f'<span class="ui-badge {tone_ or tone(word)}{" lg" if large else ""}">{esc(word)}</span>'


def note(text: str, kind: str = "accent") -> str:
    """A note with a 3 px tone bar (accent, good, warning or bad)."""
    return f'<div class="ui-note {kind}">{esc(text)}</div>'


def rows(items: list[dict], chosen: str | None = None) -> str:
    """Rows: {key, a, a_small, b, b_small, badge}. The chosen row gets the accent bar."""
    out = []
    for it in items:
        on = " on" if chosen is not None and it.get("key") == chosen else ""
        a_small = f'<small>{esc(it["a_small"])}</small>' if it.get("a_small") else ""
        b_small = f'<small>{esc(it["b_small"])}</small>' if it.get("b_small") else ""
        b = badge(it["badge"]) if it.get("badge") else esc(it.get("b", ""))
        out.append(f'<div class="ui-row{on}"><div class="a">{esc(it["a"])}{a_small}</div><div class="b">{b}{b_small}</div></div>')
    return f'<div class="ui-rows">{"".join(out)}</div>'


def pair(items: list[tuple[str, str, str | None]]) -> str:
    """Two figures side by side separated by a hairline: (label, value, small)."""
    cells = []
    for label, value, small in items:
        sm = f"<small>{esc(small)}</small>" if small else ""
        cells.append(f'<div><div class="l">{esc(label)}</div><div class="v">{esc(value)}{sm}</div></div>')
    return f'<div class="ui-pair" style="--n:{len(items)}">{"".join(cells)}</div>'


def eyebrow(text: str) -> str:
    return f'<div class="ui-eyebrow">{esc(text)}</div>'


def empty(sentence: str) -> str:
    """An empty state or an error: one plain sentence saying what to do."""
    return f'<div class="ui-empty">{esc(sentence)}</div>'


def figure_html(label: str, value: str, meta: str, on: bool = False, hero: bool = False,
                count_to: float | None = None, decimals: int = 0, prefix: str = "", suffix: str = "",
                fid: str | None = None) -> str:
    cls = "ui-figure ui-arrive" + (" on" if on else "") + (" hero" if hero else "")
    attrs = ""
    if count_to is not None:
        attrs = (f' class="v ui-count" data-to="{count_to}" data-dec="{decimals}" data-pre="{esc(prefix)}" '
                 f'data-suf="{esc(suffix)}"' + (f' data-id="{fid}"' if fid else ""))
    else:
        attrs = ' class="v"'
    return (f'<div class="{cls}"><div class="l">{esc(label)}</div><div{attrs}>{esc(value)}</div>'
            f'<div class="m">{esc(meta)}</div></div>')


def figure_cards(items: list[dict], chosen: str, key: str = "figures") -> str:
    """Four figure cards (2 x 2) that are the filters of the middle picture.

    Each item: key, label, value, meta, help (what it is and how it is
    worked out), and optionally count_to / decimals / prefix / suffix for
    the count-up. The card is an HTML figure with a full-size button over
    it, so it is a real button for the keyboard and a screen reader.
    Returns the chosen key (the pressed card, or the current one).
    """
    picked = chosen
    pairs = [items[i:i + 2] for i in range(0, len(items), 2)]
    with st.container(key=key):
        for row in pairs:
            cols = st.columns(len(row))
            for col, it in zip(cols, row):
                with col:
                    with st.container(key=f"fig_{it['key']}"):
                        U.html(figure_html(it["label"], it["value"], it["meta"], on=(it["key"] == chosen),
                                           count_to=it.get("count_to"), decimals=it.get("decimals", 0),
                                           prefix=it.get("prefix", ""), suffix=it.get("suffix", ""), fid=it["key"]))
                        if st.button(f"Show {it['label']}", key=f"btn_{key}_{it['key']}", help=it.get("help"),
                                     use_container_width=True):
                            picked = it["key"]
    return picked


def picture(fig, height: int | None = None, key: str = "picture", on_select=None):
    """The big picture in the middle: the chart is the card.

    Without ``height`` the chart fills the board's height (the CSS sizes the
    card from the viewport and Plotly resizes to it); pass a height only for
    a picture that must stay small.
    """
    with st.container(key=key if height is None else f"{key}_fixed"):
        if height is None:
            fig.update_layout(autosize=True, height=None)
        return U.chart(fig, height=height, key=f"{key}_chart", on_select=on_select)


def chart_layout(fig, title: str | None = None):
    """House layout for a picture: hairline grid, muted axis text, no legend box."""
    fig.update_layout(
        title=dict(text=title or "", font=dict(size=17, color=CHART["ink"]), x=0, xanchor="left"),
        font=dict(family=U.FONT_STACK, size=13, color=CHART["ink"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=CHART["hairline"], linecolor=CHART["hairline"], tickfont=dict(color=CHART["muted"], size=12),
                   title=dict(font=dict(color=CHART["muted"], size=12))),
        yaxis=dict(gridcolor=CHART["hairline"], linecolor=CHART["hairline"], tickfont=dict(color=CHART["muted"], size=12),
                   title=dict(font=dict(color=CHART["muted"], size=12))),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=12, color=CHART["muted"]), bgcolor="rgba(0,0,0,0)"),
        colorway=SERIES,
        margin=dict(l=12, r=16, t=56 if title else 24, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Runtime: arrival on first mount, count-up, count from old to new, wash
# ---------------------------------------------------------------------------

_RUNTIME_JS = """
<script>
(function () {
  var w = window.parent, doc = w && w.document;
  if (!doc) return;
  var root = doc.documentElement;
  var reduce = w.matchMedia && w.matchMedia('(prefers-reduced-motion: reduce)').matches;
  root.classList.add('ui-js');
  w.__uiSeen = w.__uiSeen || {};
  w.__uiValues = w.__uiValues || {};
  var page = w.location.pathname;
  function fmt(v, dec, pre, suf) {
    return pre + v.toLocaleString('en-GB', {minimumFractionDigits: dec, maximumFractionDigits: dec}) + suf;
  }
  function count(el, from, to, dec, pre, suf, wash) {
    if (reduce || from === to) { el.textContent = fmt(to, dec, pre, suf); return; }
    var start = w.performance.now(), dur = 900;
    if (wash) { var card = el.closest('.ui-figure'); if (card) { card.classList.remove('wash'); void card.offsetWidth; card.classList.add('wash'); } }
    function step(now) {
      var k = Math.min(1, (now - start) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(from + (to - from) * e, dec, pre, suf);
      if (k < 1) w.requestAnimationFrame(step); else el.textContent = fmt(to, dec, pre, suf);
    }
    w.requestAnimationFrame(step);
  }
  function scan() {
    var first = !w.__uiSeen[page];
    var parts = doc.querySelectorAll('.ui-arrive:not(.in)');
    parts.forEach(function (el, i) {
      if (first) { el.style.setProperty('--delay', Math.min(i, 10) * 70 + 'ms'); }
      else { el.style.setProperty('--delay', '0ms'); el.style.transition = 'none'; }
      w.requestAnimationFrame(function () { el.classList.add('in'); });
    });
    doc.querySelectorAll('.ui-count:not([data-done])').forEach(function (el) {
      el.setAttribute('data-done', '1');
      var to = parseFloat(el.dataset.to), dec = parseInt(el.dataset.dec || '0', 10);
      var pre = el.dataset.pre || '', suf = el.dataset.suf || '', id = el.dataset.id || el.textContent;
      if (isNaN(to)) return;
      var key = page + '|' + id, old = w.__uiValues[key];
      if (first || old === undefined) count(el, 0, to, dec, pre, suf, false);
      else if (old !== to) count(el, old, to, dec, pre, suf, true);
      w.__uiValues[key] = to;
    });
    if (parts.length) w.__uiSeen[page] = true;
    var b = doc.querySelector('[data-testid="stSidebarNavViewButton"]');
    if (b && /more/i.test(b.innerText) && !b.__done) { b.__done = true; b.click(); }
  }
  if (!w.__uiRuntime) {
    w.__uiRuntime = {scan: scan};
    new w.MutationObserver(function () { scan(); }).observe(doc.body, {childList: true, subtree: true});
  }
  scan();
})();
</script>
"""


def runtime() -> None:
    """Install the motion runtime (zero-height component, hidden by CSS)."""
    try:
        import streamlit.components.v1 as components
        components.html(_RUNTIME_JS, height=0)
    except Exception:  # noqa: BLE001
        pass


def page(name: str, line: str | None = None, meta: str | None = None, eyebrow: str | None = None,
         heading: str | None = None, art: bool = True) -> None:
    """Start of every page: runtime, the page bar and, when ``heading`` is given, the title block."""
    runtime()
    bar(name, line, meta)
    if heading:
        title(eyebrow or name, heading, line or "", art=art)


def row_button(item: dict, key: str, chosen: bool = False) -> bool:
    """One row that opens a sheet: the row markup with a full-size button over it. Returns True when pressed."""
    on = " on" if chosen else ""
    a_small = f'<small>{esc(item["a_small"])}</small>' if item.get("a_small") else ""
    b = badge(item["badge"]) if item.get("badge") else esc(item.get("b", ""))
    b_small = f'<small>{esc(item["b_small"])}</small>' if item.get("b_small") else ""
    with st.container(key=f"row_{key}"):
        U.html(f'<div class="ui-row{on}"><div class="a">{esc(item["a"])}{a_small}</div>'
               f'<div class="b">{b}{b_small}</div><span class="arrow">&#8250;</span></div>')
        return st.button(f"Open {item['a']}", key=f"btn_row_{key}", use_container_width=True)
