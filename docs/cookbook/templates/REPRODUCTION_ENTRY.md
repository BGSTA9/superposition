# E__ `<experiment_name>` (pp. __)

<!--
One section of docs/REPRODUCTION.md per experiment (SPEC §10): the paper's claim with page, our
figure, status, evidence, discrepancies with hypotheses, run IDs. The E00 section of
../../REPRODUCTION.md is a worked example. In REPRODUCTION.md this becomes an H2 section.
Every number must appear in a run record you can name. Write in your own words.
-->

**Paper's claim (p. __).** {What the paper says happens, paraphrased, with the page of each
part. Name anything the paper leaves unstated and how you filled it (D-___).}

**Our figures** ({profile}, {selection rule, e.g. best of 8 restarts}):

<!-- Embed each published figure, e.g. `![E00 arrows](figures/<experiment>/<file>.png)`. -->

**Status: {PASS | PARTIAL | FAIL | INFO} ({k}/{n} graded criteria).** {Two to four sentences:
what reproduced, with the key numbers; what did not.}

**Evidence.** {The criteria table, generated from the cited run's acceptance record if you have
a report command; otherwise copied from it with the run ID.}

| Criterion | Paper | Status | Value | Threshold |
|---|---|---|---|---|
| | p. __ | | | |

**Robustness.** {Other profiles, seeds or devices, and whether they agree. Held-out seeds for any
result near its threshold.}

**Discrepancies, and hypotheses for them.**

1. *{What differs.}* {Measured how, from which run.} **Hypothesis:** {labeled as a hypothesis,
   with what would test it and, if you can, how likely it is.}

**Changes since first reported.** {Criteria or statuses changed, with D-___ and review R_-__;
keep the original result visible.}

**Run IDs.**

- Cited ({profile}, {device}, seed {s}, commit `<hash>`): `<run_id>` ({wall-clock} s)
- Supplementary ({label}): `<run_id>`
- Not cited but relevant: `<run_id>` ({why})
