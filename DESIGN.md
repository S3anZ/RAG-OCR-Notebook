# Design System & Aesthetics Guidelines (Warm Editorial & Notebook Theme)

## Core Aesthetics & Anti-Patterns

- **Typography**: 
  - Headings, answers, and editorial content use `Fraunces` (warm, high-character serif).
  - Navigation, sidebar, badges, inputs, and citation footnotes use `Public Sans` (clean, functional sans-serif).
  - *No default system fonts or plain Inter.*
- **Color Palette**:
  - Light Parchment Mode: `--bg: #FAF5EC`, `--surface: #FFFFFF`, `--text: #2B2118`, `--text2: #8A7A68`, `--border: #E8DFD0`, `--accent: #C1592B`, `--accentbg: #F5DFCB`.
  - Dark Espresso Mode: `--bg: #221A14`, `--surface: #2E241C`, `--text: #F0E6D8`, `--text2: #B0A08C`, `--border: #42362A`, `--accent: #E88A4C`, `--accentbg: #4A3221`.
  - *No stark pure black or pure white backgrounds. Warm earth tones only.*
- **Citations & Sources**:
  - Footnote numbers (`[1]`, `[2]`) embedded inline in generated responses.
  - Footnote sources listed cleanly at the bottom of each answer stream with interactive line expansion.
  - *No clunky chat-bubble cards or generic source chips.*
- **Layout & Micro-Interactions**:
  - Collapsible sidebar for document library management and file uploads.
  - Plain colored status dots (`● parsing`, `● chunking`, `● ready`).
  - Subtle hover animations and responsive theme switching.
