"""Gradio theme for the NeoMME demo — a port of the editorial-academic look from
`late-interaction-kernels/docs/how-it-works.html`.

Warm-tinted paper neutrals (hue ~75° in OKLCH), one confident oxblood accent, moss for the
"good"/answer path, and the same three faces (General Sans display, Source Serif 4 body, JetBrains
Mono). We reskin Gradio by overriding its semantic CSS variables in both light and `.dark`, so the
whole app reads as one piece rather than default Gradio chrome.
"""

import gradio as gr

# Fontshare (General Sans) and Google Fonts (Source Serif 4, JetBrains Mono). General Sans is not on
# Google Fonts, so it comes in via an @import in the CSS below; the other two ride the theme object.
_FONT_BODY = gr.themes.GoogleFont("Source Serif 4")
_FONT_MONO = gr.themes.GoogleFont("JetBrains Mono")


def build_theme() -> gr.themes.Base:
    """A neutral Gradio base; the real look is applied by NEOMME_CSS overriding Gradio's CSS vars."""
    return gr.themes.Base(
        font=[_FONT_BODY, "Source Serif Pro", "Georgia", "serif"],
        font_mono=[_FONT_MONO, "ui-monospace", "SF Mono", "monospace"],
        radius_size=gr.themes.sizes.radius_md,
        spacing_size=gr.themes.sizes.spacing_md,
    )


# Everything the reskin needs lives here. Gradio's own tokens (--body-*, --block-*, --button-*, …)
# are re-pointed at the how-it-works palette for light and dark; a few custom classes add the
# eyebrow/hero styling that gives the page its editorial feel.
NEOMME_CSS = """
@import url('https://api.fontshare.com/v2/css?f[]=general-sans@500,600,700&display=swap');

:root, .gradio-container {
  /* how-it-works.html light palette */
  --neo-bg:        oklch(98.2% 0.008 75);
  --neo-bg-soft:   oklch(96.4% 0.010 75);
  --neo-panel:     oklch(99.4% 0.005 75);
  --neo-line:      oklch(89%   0.012 75);
  --neo-ink:       oklch(22%   0.018 70);
  --neo-ink-mid:   oklch(42%   0.018 70);
  --neo-ink-dim:   oklch(58%   0.016 70);
  --neo-accent:    oklch(42%   0.13  25);   /* oxblood */
  --neo-accent-2:  oklch(46%   0.10 150);   /* moss */
  --neo-code-bg:   oklch(95%   0.018 75);

  --neo-font-display: "General Sans", -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif;
}

.dark, .dark .gradio-container {
  --neo-bg:        oklch(15%   0.014 65);
  --neo-bg-soft:   oklch(18%   0.014 65);
  --neo-panel:     oklch(20%   0.015 65);
  --neo-line:      oklch(28%   0.014 65);
  --neo-ink:       oklch(92%   0.012 75);
  --neo-ink-mid:   oklch(72%   0.015 75);
  --neo-ink-dim:   oklch(55%   0.018 75);
  --neo-accent:    oklch(72%   0.13  25);
  --neo-accent-2:  oklch(68%   0.10 150);
  --neo-code-bg:   oklch(22%   0.016 65);
}

/* Re-point Gradio's semantic variables at the palette (light + dark share the same names). */
:root, .gradio-container, .dark, .dark .gradio-container {
  --body-background-fill:        var(--neo-bg);
  --background-fill-primary:     var(--neo-bg);
  --background-fill-secondary:   var(--neo-bg-soft);
  --block-background-fill:       var(--neo-panel);
  --block-border-color:          var(--neo-line);
  --block-label-background-fill: var(--neo-bg-soft);
  --border-color-primary:        var(--neo-line);
  --border-color-accent:         var(--neo-accent);
  --body-text-color:             var(--neo-ink);
  --body-text-color-subdued:     var(--neo-ink-dim);
  --block-label-text-color:      var(--neo-ink-dim);
  --block-title-text-color:      var(--neo-ink-mid);
  --input-background-fill:       var(--neo-panel);
  --input-border-color:          var(--neo-line);
  --color-accent:                var(--neo-accent);
  --color-accent-soft:           color-mix(in oklch, var(--neo-accent) 12%, var(--neo-bg));
  --link-text-color:             var(--neo-accent);
  --link-text-color-hover:       var(--neo-accent);
  --code-background-fill:        var(--neo-code-bg);

  --button-primary-background-fill:        var(--neo-accent);
  --button-primary-background-fill-hover:  color-mix(in oklch, var(--neo-accent) 85%, black);
  --button-primary-text-color:             var(--neo-bg);
  --button-primary-border-color:           var(--neo-accent);
  --button-secondary-background-fill:      var(--neo-bg-soft);
  --button-secondary-border-color:         var(--neo-line);
  --button-secondary-text-color:           var(--neo-ink-mid);
  --slider-color:                          var(--neo-accent);
}

body, .gradio-container { background: var(--neo-bg); color: var(--neo-ink); }

/* Fit the app to the window when there is room for it, and scroll inside the container when there is not. On a
   Space the app runs in an iframe whose own document does not scroll, so the container has to own the scrolling
   or the bottom of a tall page becomes unreachable. */
html, body { height: 100%; }
.gradio-container { max-height: 100dvh; overflow-y: auto; }
.contain > .column { gap: 10px; padding-bottom: 22px; }   /* 16px between sections is more than this page needs */

/* Editorial headings for the Markdown blocks (title + numbered steps). */
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  font-family: var(--neo-font-display);
  color: var(--neo-ink);
  letter-spacing: -0.015em;
  font-weight: 600;
}
.gradio-container h1 { font-size: 1.85rem; line-height: 1.1; }
.gradio-container h2 { font-size: 1.15rem; }

/* Compact hero: glyph + wordmark + subtitle on a single line, minimal vertical footprint. */
.neo-hero-bar { display: flex; align-items: center; gap: 12px; margin: 2px 0 10px; }
.neo-hero-bar h1 { margin: 0; }
.neo-glyph { height: 1.9rem; width: auto; display: block; }
.neo-subtitle {
  font-family: var(--neo-mono, "JetBrains Mono", monospace);
  font-size: 10.5px;
  letter-spacing: 0.22em;
  color: var(--neo-accent);
  text-transform: uppercase;
  font-weight: 500;
}

/* Compact control band + section headings: keep the whole control area above the fold. */
.neo-controls { gap: 18px; }
.neo-status { color: var(--neo-ink-mid); font-size: 0.92rem; margin: 2px 0 0; }
.neo-step h2 { margin: 4px 0 8px; }
.neo-step h3 { margin: 4px 0 6px; }

/* "How it works" is always open above the columns. h3 has no size rule of its own and would otherwise inherit
   a browser default LARGER than the 1.15rem column headings, so pin it just below them. */
.neo-about { margin: 0 0 2px; font-size: 0.9rem; line-height: 1.4; }
.neo-about h3 { font-size: 1.02rem; margin: 2px 0 4px; }
.neo-about p { margin: 0; }

/* Equal-height control columns. equal_height makes Gradio flex-grow EVERY block, which splits the slack
   across section 3's fields and spreads them apart. Reset that, then let only the single big boxes (upload,
   query) absorb the extra height; everything else packs to the top with the slack falling at the bottom. */
.neo-controls > .column { justify-content: flex-start; gap: 8px; }   /* the 16px default is most of the slack */
/* Direct children only — so the Provider/Model blocks INSIDE their row keep growing side by side. */
.neo-controls > .column > .block,
.neo-controls > .column > .form,
.neo-controls > .column > .row,
.neo-controls > .column > button { flex-grow: 0 !important; }   /* a Button is a bare <button>, not a .block */
.neo-controls > .column > .neo-upload,
.neo-controls > .column > .neo-query { flex-grow: 1 !important; }
/* Vertical sizes track the window height, so a large window fits the whole app on one screen while a short one
   keeps every box usable and scrolls instead. The upload and query boxes are sized so the three control columns
   come out roughly level with column 3, which ends at its Submit button. */
.neo-upload { min-height: clamp(110px, 13dvh, 180px); }   /* also its constant height when a file is dropped */
.neo-query textarea { min-height: clamp(64px, 7.5dvh, 130px); height: 100%; }

/* The pages and the answer take whatever height the window has left. Everything above them measures 712px at this
   width, so subtracting that fills the screen exactly, and the floor keeps both usable on a short window where the
   container scrolls instead. A definite height matters here: sized by flex-grow, the gallery's grid escapes its
   column and lands on top of the answer. The answer sits under a 40px tab bar, so it gets 40px less than the
   gallery and the two columns still end level. */
.neo-gallery { height: clamp(150px, calc(100dvh - 712px), 700px); }
#neo-answer,
.neo-answer textarea { height: clamp(110px, calc(100dvh - 752px), 660px); }
#neo-answer { overflow-y: auto; }
/* Past 1600px the "How it works" paragraph wraps to two lines instead of three, freeing 22px. */
@media (min-width: 1600px) {
  .neo-gallery { height: clamp(150px, calc(100dvh - 690px), 700px); }
  #neo-answer,
  .neo-answer textarea { height: clamp(110px, calc(100dvh - 730px), 660px); }
}
/* One line of example buttons rather than three wrapped ones, and one line of scoring options. */
.neo-examples { flex-wrap: nowrap; gap: 6px; }
/* min-width lets the three buttons share the column instead of overflowing it, since nowrap otherwise pins each
   one to its text width. */
.neo-examples button {
  white-space: nowrap;
  font-size: 0.8rem;
  padding-left: 6px;
  padding-right: 6px;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}
.neo-step h2 + * { margin-top: 0; }
/* Hints belong to the control right above them, so they get no padding of their own and almost no margin. */
.neo-hint { color: var(--neo-ink-mid); font-size: 0.9rem; font-style: italic; margin: -4px 0 0; padding: 0; }

/* Primary action buttons get the confident accent; keep the label calm. */
.gradio-container button.primary,
.gradio-container .gr-button-primary { font-weight: 600; letter-spacing: 0.01em; }

/* Answer panel — moss-tinted like the "streamed/good" path in the source. */
#neo-answer textarea {
  background: color-mix(in oklch, var(--neo-accent-2) 6%, var(--neo-panel));
  border-color: color-mix(in oklch, var(--neo-accent-2) 22%, var(--neo-line));
  font-size: 1rem;
  line-height: 1.6;
}

footer { display: none !important; }
"""
