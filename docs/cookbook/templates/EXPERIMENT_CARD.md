# E__ · `<experiment_name>` (pp. __)

<!--
One card per experiment, in ten sections (the cookbook's recipe-card format; this repository's
cards are in ../experiments/). Fill sections 1-5 BEFORE the first graded run and commit them;
fill 6-10 afterwards. Label each step about the past [Documented], [Reconstructed] (say from
what) or [Recommended]. Terms: ../../GLOSSARY.md.
-->

## 1. Claim and pages

- **The paper says** (paraphrase, with page): ____ (p. __)
- **Figure or table it rests on:** ____ (p. __)

## 2. Prerequisites

- Modules: ____
- Earlier experiments or results it depends on: ____
- Decisions it relies on (D-___): ____

## 3. Config and commands

- Config file: `configs/<experiment>.yaml`
- Profiles: `quick` = ____ ; `full` = ____ ; others = ____

```bash
uv run <your-cli> run <experiment> --profile quick --seed 0
```

[Not verified: template placeholder; replace with your project's real command.]

## 4. Procedure

<!-- Numbered steps. Each: action → command → expected result → how to check → if it fails. -->

1. **Action:** ____
   - Command: ____
   - Expected result: ____
   - Check: ____
   - If it fails: ____

## 5. Acceptance

<!-- Written and committed before the graded run. Source = paper page, or "proposal". -->

| # | Criterion (operational) | Source | Threshold and why | Status | Evidence (run ID) |
|---|---|---|---|---|---|
| 1 | | p. __ / proposal | | PASS / PARTIAL / FAIL / INFO | |

- Committed at: `<hash>` on `<date>` (before any graded run: yes / no)
- Already seen before writing these (disclose): ____

## 6. What actually happened

- Issues, diagnosis, decisions (D-___): ____
- **Changed after results were seen** (flag each, with its DECISIONS entry): ____
- Runs outside the provenance system (exploratory, uncited): ____

## 7. Compute

| Run ID | Profile | Device | Wall-clock (measured) |
|---|---|---|---|
| | | | |

Estimates (label them as estimates, show the arithmetic): ____

## 8. What to look at

- Figures: `docs/figures/<experiment>/____.png`
- Notebook or app page: ____

## 9. Explain it back

1. ____
2. ____
3. ____

- **Drill:** ____ · Self-check: ____

## 10. Answer key

<!-- Files and functions to open after attempting, never before. -->

- `src/<package>/<module>.py`: `<function>`
- `tests/<path>::<test>`
