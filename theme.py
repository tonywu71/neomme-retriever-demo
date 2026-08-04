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

/* Editorial headings for the Markdown blocks (title + numbered steps). */
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  font-family: var(--neo-font-display);
  color: var(--neo-ink);
  letter-spacing: -0.015em;
  font-weight: 600;
}
.gradio-container h1 { font-size: 2.1rem; line-height: 1.1; }
.gradio-container h2 { font-size: 1.15rem; }

/* Compact hero: glyph + wordmark + subtitle on a single line, minimal vertical footprint. */
.neo-hero-bar { display: flex; align-items: center; gap: 12px; margin: 2px 0 10px; }
.neo-hero-bar h1 { margin: 0; }
.neo-glyph { height: 2.2rem; width: auto; display: block; }
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
.neo-about { margin: 0 0 6px; }
.neo-about h3 { font-size: 1.02rem; margin: 2px 0 6px; }

/* Equal-height control columns. equal_height makes Gradio flex-grow EVERY block, which splits the slack
   across section 3's fields and spreads them apart. Reset that, then let only the single big boxes (upload,
   query) absorb the extra height; everything else packs to the top with the slack falling at the bottom. */
.neo-controls > .column { justify-content: flex-start; }
/* Direct children only — so the Provider/Model blocks INSIDE their row keep growing side by side. */
.neo-controls > .column > .block,
.neo-controls > .column > .form,
.neo-controls > .column > .row,
.neo-controls > .column > button { flex-grow: 0 !important; }   /* a Button is a bare <button>, not a .block */
.neo-controls > .column > .neo-upload,
.neo-controls > .column > .neo-query { flex-grow: 1 !important; }
.neo-upload { min-height: 250px; }        /* also its constant height when a file is dropped */
.neo-query textarea { min-height: 202px; height: 100%; }
/* The "optional …" hint sits tight under the section 3 heading rather than floating. */
.neo-hint { color: var(--neo-ink-mid); font-size: 0.9rem; font-style: italic; margin: -2px 0 10px; }

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
