// CreditLens page checks: run in the app's browser pane on a loaded page. Returns one object of numbers.
(function () {
  var out = {};
  var se = document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
  out.innerHeight = innerHeight; out.scrollHeight = se.scrollHeight; out.clientHeight = se.clientHeight; out.pageScroll = se.scrollHeight > se.clientHeight + 1 || document.scrollingElement.scrollHeight > innerHeight;
  var main = document.querySelector('[data-testid="stMain"]') || document.body;
  var overflow = [];
  main.querySelectorAll('*').forEach(function (el) {
    var cs = getComputedStyle(el);
    if (cs.display === 'none' || el.closest('[data-testid="stSidebar"]')) return;
    if (el.tagName === 'IFRAME' || el.closest('.js-plotly-plot') || el.closest('[data-testid="stDataFrame"]') || el.closest('[data-testid="stDataEditor"]')) return;
    if ((el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1) && (cs.overflowX !== 'visible' || cs.overflowY !== 'visible')) {
      if (el.classList.contains('cl-table-wrap') && el.scrollHeight <= el.clientHeight + 1) return;
      overflow.push(el.tagName + '.' + (el.className && el.className.baseVal === undefined ? String(el.className).slice(0, 40) : ''));
    }
  });
  out.overflow = overflow.slice(0, 15); out.overflowCount = overflow.length;
  var wraps = [];
  main.querySelectorAll('.ui-figure .l, .ui-figure .m, .ui-pair .l, .ui-card > .h').forEach(function (el) {
    var lh = parseFloat(getComputedStyle(el).lineHeight) || 16;
    if (el.getBoundingClientRect().height > lh * 1.6) wraps.push(el.textContent.trim().slice(0, 40));
  });
  out.labelWraps = wraps;
  function isBox(el) {
    var cs = getComputedStyle(el);
    if (cs.display === 'none') return false;
    var bg = cs.backgroundColor, sh = cs.boxShadow;
    var fill = bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent';
    var sides = ['Top', 'Right', 'Bottom', 'Left'].filter(function (s) { return parseFloat(cs['border' + s + 'Width']) > 0 && cs['border' + s + 'Style'] !== 'none'; }).length;
    return (fill && el.getBoundingClientRect().width > 40 && el.getBoundingClientRect().height > 24) || sides >= 3 || (sh && sh !== 'none');
  }
  var nested = [];
  main.querySelectorAll('.ui-card, .ui-figure, .cl-card, .cl-tiles, [data-testid="stPlotlyChart"], [data-testid="stForm"], [data-testid="stExpander"] details, .cl-table-wrap').forEach(function (el) {
    var p = el.parentElement, found = null;
    while (p && p !== main) { if (p.matches('.ui-card, .ui-figure, .cl-card, .cl-tiles, [data-testid="stPlotlyChart"], [data-testid="stForm"], .cl-table-wrap') && isBox(p)) { found = p; break; } p = p.parentElement; }
    if (found) nested.push(el.className.toString().slice(0, 30) + ' in ' + found.className.toString().slice(0, 30));
  });
  out.nestedBoxes = nested.slice(0, 10); out.nestedCount = nested.length;
  out.buttons = main.querySelectorAll('button').length;
  var small = 0; main.querySelectorAll('button').forEach(function (b) { var r = b.getBoundingClientRect(); if (r.height > 0 && r.height < 32) small++; });
  out.smallButtons = small;
  return out;
})();
