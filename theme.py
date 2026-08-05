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
  --neo-ink-dim:   oklch(50%   0.016 70);
  --neo-accent:    oklch(42%   0.13  25);   /* oxblood */
  --neo-accent-2:  oklch(46%   0.10 150);   /* moss */
  --neo-code-bg:   oklch(95%   0.018 75);

  --neo-font-display: "General Sans", "Avenir Next", "Helvetica Neue", sans-serif;
}

.dark, .dark .gradio-container {
  --neo-bg:        oklch(15%   0.014 65);
  --neo-bg-soft:   oklch(18%   0.014 65);
  --neo-panel:     oklch(20%   0.015 65);
  --neo-line:      oklch(28%   0.014 65);
  --neo-ink:       oklch(92%   0.012 75);
  --neo-ink-mid:   oklch(72%   0.015 75);
  --neo-ink-dim:   oklch(68%   0.018 75);
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

/* The setup is a real two-step sequence. Settings stay available without competing with the task. */
.neo-settings { margin: 4px 0 10px; }
.neo-settings > .label-wrap,
.neo-cite > .label-wrap {
  color: var(--neo-ink-mid);
  font-family: var(--neo-font-display);
  font-size: 0.88rem;
  font-weight: 600;
}
/* Gradio uses a rotating black triangle for accordions. Replace it with a quiet plus/minus control. */
.block.gr-accordion.neo-settings > button.label-wrap .icon,
.block.gr-accordion.neo-cite > button.label-wrap .icon {
  position: relative;
  display: inline-flex;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  font-size: 0;
  transform: none !important;
}
.block.gr-accordion.neo-settings > button.label-wrap .icon::before,
.block.gr-accordion.neo-settings > button.label-wrap .icon::after,
.block.gr-accordion.neo-cite > button.label-wrap .icon::before,
.block.gr-accordion.neo-cite > button.label-wrap .icon::after {
  content: "";
  position: absolute;
  background: currentColor;
  transition: opacity 150ms ease, transform 150ms ease;
}
.block.gr-accordion.neo-settings > button.label-wrap .icon::before,
.block.gr-accordion.neo-cite > button.label-wrap .icon::before {
  width: 10px;
  height: 1.5px;
}
.block.gr-accordion.neo-settings > button.label-wrap .icon::after,
.block.gr-accordion.neo-cite > button.label-wrap .icon::after {
  width: 1.5px;
  height: 10px;
}
.block.gr-accordion.neo-settings > button.label-wrap.open .icon::after,
.block.gr-accordion.neo-cite > button.label-wrap.open .icon::after {
  opacity: 0;
  transform: rotate(90deg);
}
.neo-settings > .label-wrap:focus-visible,
.neo-cite > .label-wrap:focus-visible {
  outline: 1px solid color-mix(in oklch, var(--neo-accent) 62%, var(--neo-line));
  outline-offset: 1px;
}
.neo-workspace { align-items: stretch; gap: clamp(20px, 3vw, 40px); }
.neo-workspace > .column { justify-content: flex-start; gap: 10px; min-width: 0; }
.neo-index-panel,
.neo-query-panel { padding-top: 2px; }
.neo-status {
  color: var(--neo-ink-mid);
  font-family: var(--neo-font-display);
  font-size: 0.88rem;
  line-height: 1.45;
  margin: 2px 0 0;
}
.neo-run-meta { font-variant-numeric: tabular-nums; }
.neo-step h2 { margin: 4px 0 8px; }
.neo-step h3 { margin: 4px 0 6px; }
.neo-results-heading { margin-top: 12px; }
.neo-model-links {
  display: flex;
  flex-wrap: nowrap;
  align-items: baseline;
  gap: 10px;
  margin: -3px 0 0;
  color: var(--neo-ink-dim);
  font-family: var(--neo-font-display);
  font-size: clamp(0.72rem, 1.15vw, 0.82rem);
  white-space: nowrap;
}
.neo-model-links a {
  color: var(--neo-accent);
  font-weight: 600;
  text-decoration: underline;
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}
.neo-model-links a:hover { color: color-mix(in oklch, var(--neo-accent) 78%, var(--neo-ink)); }

/* The introduction states the job once, then gets out of the way. */
.neo-about { max-width: 72ch; margin: 0 0 4px; font-size: 0.98rem; line-height: 1.5; }
.neo-about p { margin: 0; }

/* Natural heights survive new controls and narrow viewports without viewport-subtraction arithmetic. */
.neo-upload { min-height: 220px; }
.neo-query textarea { min-height: 174px; }
.neo-gallery { min-height: clamp(300px, 46dvh, 620px); height: auto; }
#neo-answer { min-height: 180px; max-height: 520px; overflow-y: auto; }
.neo-answer textarea { min-height: 180px; max-height: 520px; }

/* Examples wrap instead of truncating, so every question remains readable and tappable. */
.neo-examples { flex-wrap: wrap; gap: 8px; }
.neo-examples button {
  flex: 1 1 150px;
  min-height: 44px;
  white-space: normal;
  font-size: 0.82rem;
  line-height: 1.2;
}
.neo-step h2 + * { margin-top: 0; }
/* Hints are explanatory prose. Italics are reserved for emphasis inside that prose. */
.neo-hint { color: var(--neo-ink-mid); font-size: 0.9rem; line-height: 1.45; margin: -2px 0 0; padding: 0; }
.neo-hint p { margin: 0; }

/* Primary action buttons get the confident accent; keep the label calm. */
.gradio-container button.primary,
.gradio-container .gr-button-primary { font-weight: 600; letter-spacing: 0.01em; }
.gradio-container button { min-height: 44px; }

/* The answer section gains its own restrained color only after retrieval. */
.neo-answer-heading { margin-top: 16px; }
.neo-answer-section { gap: 10px; }
.neo-answer-controls { align-items: stretch; gap: 12px; }
#neo-answer,
.neo-answer textarea {
  background: color-mix(in oklch, var(--neo-accent-2) 6%, var(--neo-panel));
  border-color: color-mix(in oklch, var(--neo-accent-2) 22%, var(--neo-line));
  font-size: 1rem;
  line-height: 1.6;
}

/* Keyboard focus must remain visible after reskinning Gradio controls. */
.gradio-container :is(a, button, input, textarea, [role="radio"], [role="slider"]):focus-visible {
  outline: 2px solid var(--neo-accent);
  outline-offset: 2px;
}

@media (max-width: 760px) {
  .contain > .column { gap: 14px; padding-inline: 12px; }
  .neo-workspace { flex-direction: column; gap: 24px; }
  .neo-upload { min-height: 180px; }
  .neo-query textarea { min-height: 140px; }
  .neo-gallery { min-height: 280px; }
  .neo-examples button { flex-basis: 100%; }
  .neo-answer-controls { flex-direction: column; }
  .neo-model-links { flex-wrap: wrap; white-space: normal; }
}

footer { display: none !important; }
"""
