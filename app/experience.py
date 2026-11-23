"""Presentation layer of CreditLens: dark mode, hero animation, auto-play strip and the guided tour.

Every effect here degrades gracefully: the hero canvas, the scroll reveal
and the count-up are added by small vanilla scripts that run in the
same-origin component iframe and act on the parent document; without
JavaScript the page shows the same content, fully visible, with final values.
"""

from __future__ import annotations

import json

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
# Hero: animated loan flow
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


_HERO_JS = """
<script>
(function () {
  var w = window.parent, doc = w && w.document;
  if (!doc) return;
  var MIX = __MIX__;
  var COL = {pre: [170, 196, 226], a: [42, 157, 143], r: [233, 196, 106], d: [231, 111, 81]};
  function start(c, hero) {
    var ctx = c.getContext('2d');
    var reduce = w.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var phone = w.matchMedia('(max-width: 768px)').matches;
    var N = phone ? 46 : 150, dpr = Math.min(w.devicePixelRatio || 1, 2), W = 0, H = 0, running = false;
    function resize() {
      var r = hero.getBoundingClientRect(); W = r.width; H = r.height;
      c.width = W * dpr; c.height = H * dpr; c.style.width = W + 'px'; c.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    if ('ResizeObserver' in w) new w.ResizeObserver(resize).observe(hero);
    function gate() { return W * (phone ? 0.62 : 0.66); }
    function lane(s) { return H * ({a: 0.26, r: 0.55, d: 0.82}[s]); }
    function spawn(p, anywhere) {
      p.x = anywhere ? Math.random() * W : -Math.random() * W * 0.25;
      p.y = H * (0.12 + Math.random() * 0.76); p.v = 0.35 + Math.random() * 0.85;
      p.r = 1.1 + Math.random() * 2.1; p.s = 'pre'; p.j = (Math.random() - 0.5) * H * 0.12; p.f = 1;
      if (anywhere && p.x > gate()) decide(p);
      return p;
    }
    function decide(p) {
      var u = Math.random();
      p.s = u < MIX.approve ? 'a' : (u < MIX.approve + MIX.refer ? 'r' : 'd');
    }
    var ps = []; for (var i = 0; i < N; i++) ps.push(spawn({}, true));
    function draw() {
      ctx.clearRect(0, 0, W, H);
      var gx = gate();
      var g = ctx.createLinearGradient(gx - 40, 0, gx + 40, 0);
      g.addColorStop(0, 'rgba(127,209,199,0)'); g.addColorStop(0.5, 'rgba(127,209,199,0.20)'); g.addColorStop(1, 'rgba(127,209,199,0)');
      ctx.fillStyle = g; ctx.fillRect(gx - 40, H * 0.06, 80, H * 0.88);
      ctx.fillStyle = 'rgba(200,235,230,0.55)'; ctx.fillRect(gx - 1, H * 0.08, 2, H * 0.84);
      for (var i = 0; i < ps.length; i++) {
        var p = ps[i], c0 = COL[p.s];
        var fade = Math.min(1, 0.25 + p.x / (W * 0.7)) * p.f;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r * 3.2, 0, 6.283);
        ctx.fillStyle = 'rgba(' + c0 + ',' + (0.10 * fade) + ')'; ctx.fill();
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.283);
        ctx.fillStyle = 'rgba(' + c0 + ',' + (0.85 * fade) + ')'; ctx.fill();
      }
    }
    function step() {
      var gx = gate();
      for (var i = 0; i < ps.length; i++) {
        var p = ps[i];
        if (p.s === 'pre') {
          p.x += p.v * 1.6;
          p.y += (H * 0.5 - p.y) * 0.004;
          if (p.x >= gx) decide(p);
        } else {
          var speed = p.s === 'a' ? 1.9 : (p.s === 'r' ? 1.1 : 0.7);
          p.x += p.v * speed;
          p.y += (lane(p.s) + p.j - p.y) * 0.03;
          if (p.s === 'd') p.f = Math.max(0, p.f - 0.004);
        }
        if (p.x > W + 12 || p.f <= 0) spawn(p, false);
      }
    }
    function loop() {
      if (!c.isConnected) { running = false; return; }
      if (doc.hidden) { running = false; return; }
      step(); draw(); w.requestAnimationFrame(loop);
    }
    function go() { if (!running && !reduce) { running = true; w.requestAnimationFrame(loop); } }
    if (reduce) { for (var k = 0; k < 400; k++) step(); draw(); return; }
    doc.addEventListener('visibilitychange', function () { if (!doc.hidden && c.isConnected) go(); });
    go();
  }
  function mount() {
    var hero = doc.querySelector('.st-key-hero');
    if (!hero) return false;
    if (hero.querySelector('canvas.cl-flow')) return true;
    var c = doc.createElement('canvas');
    c.className = 'cl-flow'; c.setAttribute('aria-hidden', 'true');
    hero.insertBefore(c, hero.firstChild);
    start(c, hero);
    return true;
  }
  var tries = 0, t = setInterval(function () { if (mount() || ++tries > 60) clearInterval(t); }, 100);
})();
</script>
"""


def hero_canvas(mix: dict) -> None:
    """Animated loans drifting through the scoring gate, coloured by decision in the given proportions."""
    try:
        import streamlit.components.v1 as components
        payload = json.dumps({k: round(float(v), 4) for k, v in mix.items()})
        components.html(_HERO_JS.replace("__MIX__", payload), height=0)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Auto-play strip and scoring sequence (CSS animation only)
# ---------------------------------------------------------------------------

def watch_strip(examples) -> str:
    """Three example applicants that take turns being scored (CSS keyframes, no reruns)."""
    cards = []
    n = len(examples)
    for i, (key, r) in enumerate(examples.iterrows()):
        colour = U.DECISION_COLOURS[r["decision"]]
        cards.append(
            f'<div class="cl-watch-card">'
            f'<div class="cl-watch-who"><span class="cl-watch-tag">{U.EXAMPLE_LABELS.get(key, key)}</span>'
            f'<b>{r["Sector"]}</b><span>{U.state_name(r["State"])}, {int(r["NoEmp"])} employees, '
            f'{"new" if r["NewExist"] == "New" else "established"} business</span>'
            f'<span>USD {r["DisbursementGross"]:,.0f} over {int(r["Term"])} months, {r["Portion"]:.0%} guaranteed</span></div>'
            f'<div class="cl-watch-score"><div class="cl-watch-steps"><i>Cleaned</i><i>Features</i><i>Scored</i></div>'
            f'<div class="cl-watch-bar"><div class="cl-watch-fill" style="width:{max(min(r["p"], 1) * 100, 2):.1f}%;'
            f'background:linear-gradient(90deg,#7FD1C7,{colour})"></div></div>'
            f'<div class="cl-watch-pd"><span>PD</span><b>{r["p"]:.1%}</b>'
            f'<span class="cl-watch-dec" style="background:{colour}">{r["decision"]}</span></div></div></div>'
        )
    return f'<div class="cl-watch">{"".join(cards)}</div>'


def scoring_sequence(run_id: str) -> str:
    """Four steps that light up in turn over about one second after a loan is scored."""
    steps = ["Cleaned", "Features built", "Scored", "Explained"]
    items = "".join(f'<li><span class="dot"></span>{s}</li>' for k, s in enumerate(steps))
    return f'<ol class="cl-seq" data-run="{run_id}">{items}</ol>'


# ---------------------------------------------------------------------------
# Guided tour
# ---------------------------------------------------------------------------

TOUR = [
    {"page": "home", "path": "pages/home.py", "title": "The headline numbers",
     "text": "These tiles come straight from the model metadata: how well the model ranks risk on held-out loans, "
             "how many defaults it stops and how much charge-off loss it avoids."},
    {"page": "portfolio", "path": "pages/portfolio_threshold.py", "title": "Set the risk appetite",
     "text": "Move the threshold or pick an objective chip. Every tile, the curve and the confusion matrix show what "
             "would have happened to the held-out loans."},
    {"page": "market", "path": "pages/market_explorer.py", "title": "Explore the market",
     "text": "Press play to watch default rates move across states and years, then click a state or a sector to "
             "filter every other chart."},
    {"page": "lab", "path": "pages/model_lab.py", "title": "Look inside the model",
     "text": "Compare the final model with the comparison model on ROC and precision-recall curves. Click a point on "
             "the ROC curve to move the threshold."},
    {"page": "score", "path": "pages/score_loan.py", "title": "Score a loan",
     "text": "The borderline example is already scored. Change any input and score again, or use the what-if "
             "sliders and the similar-loans card to see why."},
]


def start_tour() -> None:
    st.session_state["tour_step"] = 0


def _go(step: int) -> None:
    if step >= len(TOUR):
        st.session_state.pop("tour_step", None)
        return
    st.session_state["tour_step"] = step
    if TOUR[step]["page"] == "score":
        st.session_state["example"] = "borderline"
        st.session_state["auto_score"] = True
    U.switch_page(TOUR[step]["path"])


def tour_card(page: str) -> None:
    """Highlighted tour card with Back and Next, shown when the tour is on this page's step."""
    step = st.session_state.get("tour_step")
    if step is None or not 0 <= step < len(TOUR):
        return
    item = TOUR[step]
    if item["page"] != page:
        with st.container(key="tour_resume"):
            c1, c2 = st.columns([4, 1], vertical_alignment="center")
            c1.markdown(f'<div class="cl-tour-mini">Tour paused at step {step + 1} of {len(TOUR)}: {item["title"]}</div>',
                        unsafe_allow_html=True)
            if c2.button("Resume", key=f"tour_resume_{page}", use_container_width=True):
                U.switch_page(item["path"])
        return
    with st.container(key="tour_card"):
        dots = "".join(f'<span class="{"on" if i == step else ""}"></span>' for i in range(len(TOUR)))
        st.markdown(
            f'<div class="cl-tour"><div class="cl-tour-head"><span class="cl-tour-step">Step {step + 1} of {len(TOUR)}</span>'
            f'<span class="cl-tour-dots">{dots}</span></div><h4>{item["title"]}</h4><p>{item["text"]}</p></div>',
            unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        if c1.button("Back", key="tour_back", disabled=step == 0, use_container_width=True):
            _go(step - 1)
        last = step == len(TOUR) - 1
        if c2.button("Finish" if last else "Next", key="tour_next", type="primary", use_container_width=True):
            if last:
                st.session_state.pop("tour_step", None)
                st.rerun()
            _go(step + 1)
        if c3.button("End tour", key="tour_end", use_container_width=True):
            st.session_state.pop("tour_step", None)
            st.rerun()


def page_start(page: str) -> None:
    """Common start of every page: runtime script and the tour card when it applies."""
    U.runtime()
    tour_card(page)
