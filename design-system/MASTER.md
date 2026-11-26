# CreditLens design system

CreditLens is the credit desk's default-risk workstation: a credit officer at a bank that lends under a
government guarantee opens it to score an application, compare applicants, run the held-out portfolio
under a threshold or a stress scenario, and read the model's documentation. The front end is Streamlit
(the stack already in the repository), so the "component kit" is `app/ui.py` (Python functions that emit
one agreed piece of markup each) plus `app/assets/tokens.css` (every colour, size, duration) and
`app/assets/theme.css` (the parts and their states). A page composes kit parts on a board; it never
restyles a part, never writes a colour, a size or a word that the kit does not provide.

Target screens: a laptop at 1440 x 760 and 1680 x 930; a tablet at 1138 x 690. A phone (390 wide) stacks
the board in one column and may scroll; nothing else may.

## Layout

The concept images decide each page's composition; the board below is the default for pages that have no concept image.

- Every page is one screen at the target sizes: no page scroll, and no card scrolls inside itself. A list
  shows the whole rows that fit, then "See all" (opens a sheet) or a pager. Only a sheet's form may scroll,
  with its buttons always visible.
- A page is a board: a left column of four small figure cards (2 x 2) and one or two cards; one big picture
  in the middle that tells the page's story (a chart, a map, a list of bars, a calendar); a right column of
  two or three cards. Figure cards are all 88 px high with the label on one line.
- The cards are the filters: clicking a figure card or a row in a side card changes what the middle shows.
  A row, a tile or a figure opens a sheet (`st.dialog`, a fixed portal) over the page; nothing swaps the page.
- 8-point spacing: 8 / 12 / 16 / 24 / 32 / 48. Page gutters 32 at the sides, 24 top and bottom. Cards pad
  12 x 20. Radii: 10 controls, 16 cards, full pills. Two shadows only: rest and hover. Depth order: canvas,
  cards, the open card, the hero (top bar and sheet).
- No box inside a box. A card is the only box on its part of the page. On it: a badge is a coloured word with
  a 6 px dot before it (never a filled pill); an icon is bare (never on a tile or circle); figures side by
  side are separated by a hairline; a chosen row gets a 3 px accent bar on its left edge; a note gets a 3 px
  tone bar; a picture is the thing itself with a soft shadow (the chart is the card).

## Visual direction (from the four concept images in design-system/concepts)

The app is a pearlescent glass workstation. The canvas is a light iridescent wash (mint, lavender and silver
gradients on #EEF3F3); every surface is frosted glass (white at about 65% with a soft white edge, a blur and a
quiet shadow, radius 22 px); the one accent is emerald. Key results sit on a dark emerald panel (gradient
#0E7A5F to #0A4F3F, white text, glossy highlight). Page titles are editorial: a green eyebrow with a short rule,
a 44 px display title with a full stop ("Credit risk, in focus."), a one-line grey subtitle, and the glass ring
art at the right of the title block. The top bar is a glass breadcrumb bar (home, page name) with status pills on
the right (model, held-out loans, approve-below). The sidebar is light glass with the active page as an emerald
pill. Badges are tinted pills with a dot (mint for good, amber for warning, rose for bad, grey for neutral).

## Colour

| Token | Value | Use |
|---|---|---|
| ink | #10302B | text, figures |
| text-2 | #4F6366 | secondary text, subtitles |
| muted | #6B7C7E | eyebrows of cards, captions, axis text |
| accent | #0F8B6C | eyebrow, primary actions, chosen state, primary series |
| accent-deep | #0A5F4B | gradient end, pressed |
| accent-soft | #DDF6EE | mint tint: good badges, hover fill |
| accent-light | #7FCDB6 | secondary series, hover borders |
| panel | #0E7A5F to #0A4F3F | dark emerald panels |
| hairline | rgba(16, 48, 43, 0.10) | separators |
| canvas | #EEF3F3 with the iridescent wash | page background |
| glass | rgba(255, 255, 255, 0.66) | cards, bars, sidebar |
| good | #0E6B52 on #DDF6EE | approved, repaid, gain |
| warning | #8A5A00 on #FBEBC8 | referred, caution |
| bad | #B23A2E on #FBDAD5 | declined, default, loss |
| neutral | ink on #E6ECEF | none, unknown, pending |

Charts use accent, accent-light, a cool blue #2563EB for a second applicant and grey #9AA8AD for a third; the
sequential scale runs accent-soft to accent-deep. One status map (`ui.STATUS`) gives every badge its tone.

## Type

Inter, tabular figures everywhere. 12 px uppercase tracked eyebrows; 13 px labels and buttons (500 to 600);
15 px body; 17 px card titles (600); 26 to 34 px figures (600, tracked -0.03em); 44 px for the one hero
figure. Rows 14 / 12 (label / meta). Sentence case everywhere. No text ever ends in "..." and nothing is
clipped: a sentence gets the room it needs or the layout changes. Loading words have no dots.

## Components and their five states (rest, hover, active, focus, disabled)

- Button: white, hairline, 10 px radius; hover accent-soft fill and accent-light border; active moves down
  1 px; focus a 2 px accent outline; disabled 55% opacity. Primary button: accent to accent-deep gradient,
  white text.
- Row: hairline under, 36 px or more, hover canvas fill with the arrow sliding 4 px.
- Input: 32 px tall (44 px on touch).
- Tabs: quiet text in a pill, the chosen one white with a hairline.
- Figure card: 88 px, eyebrow label on one line, figure 26 to 34 px, one meta line; chosen = 3 px accent bar
  on the left edge; it is a button (`ui.figure_card`).
- Badge: dot + word (`ui.badge`). Note: 3 px tone bar (`ui.note`). Sheet: `st.dialog`.
- Every button a finger can press is 44 px or more on a tablet. Hover styles only under
  `@media (hover: hover)`, so a touch screen never keeps a card looking pressed.

## Motion

One ease: cubic-bezier(.2,.7,.2,1). Three durations: 180 ms hover, 260 ms surfaces, 320 ms arrivals;
stagger 70 ms. On a page's first mount the parts arrive top to bottom, rising 12 px and fading in; figures
count up over 900 ms; the whole arrival is over in about 1 s. A rerun never replays the arrival: a changed
figure counts from its old value to the new one and its card washes for 900 ms. Animate only transform,
opacity and stroke; `prefers-reduced-motion` makes the page simply appear in its end state. Forbidden:
bounce, parallax, scale on hover, spinners, shimmer loops, layout shift, tweens of width / height / top /
left / margin. The runtime is `ui.runtime()` (one small script on the parent document).

## Data

- Every figure and word on screen comes from the model artefacts and the scored data through `app/utils.py`;
  a page never adds up, rounds or invents a label. Money, dates and units are formatted in one place
  (`ui.fmt`).
- Stale while revalidating: skeletons only on the first load; on a filter change the old figures stay,
  dimmed to 60%, until the new ones land, then count to the new value. Figures never blank.
- What is not recorded is said in words, never shown as 0. Every empty state and every error is one plain
  sentence saying what to do.
- Every figure has an explain control (what it is, how it was worked out, the numbers): `ui.figure_card(help=)`.
- A dialog is `st.dialog` (fixed portal), never held by the card it opens from.
- A control the model cannot honour (no artefact, no data) is not drawn at all.
- A refresh button visibly answers the press even when nothing changed (the card washes).

## Writing on screen

Short plain sentences, sentence case, no jargon, every figure with its unit, dates in words ("Mon 3 Nov").
A card's title says what it is; its first line says what to do.

## Checks before a page is done

At each target size: `document.scrollingElement.scrollHeight === innerHeight`; no element has
`scrollWidth > clientWidth` or `scrollHeight > clientHeight` (except the sidebar's own list); no card label
wraps to two lines; no box inside a box (every element with a fill, a border on three or more sides, or a
shadow whose ancestor is also one is listed and fixed); axe-core WCAG 2.1 A and AA with zero violations; every
button pressed and every sheet opened by mouse and by touch, each view screenshotted and looked at. The
numbers (views, violations, overflows) are reported when the page is done. The checks run as
`python tooling/checks.py` is not available in this environment, so they run in the app's browser pane
with `design-system/checks.js`.

## Order of work

MASTER.md, tokens.css, the kit (app/ui.py) with `app/pages/kit.py` showing every part in its five states,
the reference screen (Overview), the checks, then the other screens one at a time, each gated by the checks.
