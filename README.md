
# Basketball Game Strategy — PSPI (Inception)

This is a **UI-focused, evolutionary prototype** to satisfy the "Potentially Shippable Product Increment (PSPI)"
requirement in the Inception iteration. It demonstrates the main page and the three functional areas with
**hard-coded data** and **mock flows**.

## What’s included
- **Home**: overview and navigation
- **Data Explorer**: browse mock team/season offense PPP by play-type
- **Matchup Console (Baseline)**: compute Top-K ranked plays with a transparent baseline rule
- **Context Simulator**: UI-only preview of context-driven ML recommendations
- **Glossary**: key terms one click away

## Tech
Minimal Next.js (App Router), React 18, no styling frameworks (basic CSS only).

## Run
```bash
npm install
npm run dev
# open http://localhost:3000
```

## Notes
- Data are **mock** and hard-coded.
- Baseline formula uses `0.7*Off + 0.3*(1 - DefAllowed)` (illustrative) and produces a rationale string.
- This satisfies the **UI-only** PSPI requirement for Inception; no back-end or model is required at this stage.
