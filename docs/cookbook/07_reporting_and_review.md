# 07 · Reporting and review

A replication is only as good as its record. This chapter shows how the four records of this
repository were actually kept, what the independent review looked for, and how to publish the
work honestly. Labels follow [README.md](README.md); terms are in the [glossary](../GLOSSARY.md);
dates and sessions are in [01_timeline.md](01_timeline.md).

| Record | What it answers | Written how |
|---|---|---|
| [docs/REPRODUCTION.md](../REPRODUCTION.md) | Did each claim reproduce, and on what evidence? | generated tables from cited runs, plus a hand-written narrative |
| [docs/DECISIONS.md](../DECISIONS.md) | Why is each under-specified choice the way it is? | dated, append-only entries |
| [docs/progress/](../progress/) | What did each phase do, what failed, what is open? | one report per phase or session, written at its end |
| [docs/REVIEW.md](../REVIEW.md) | What did a fresh reader find wrong, and what was done? | numbered findings with resolutions |
| [docs/cited_runs.yaml](../cited_runs.yaml) | Which run is each section written from? | written only by `superposition report --pin` |

**[Documented: SPEC.md §10 "Documentation"; D-202; each file's header.]**

## 1. REPRODUCTION.md: generated blocks plus a hand-written narrative

### How it was done

1. **Per-experiment layout.** Each section gives the paper's claim with its page, the figures,
   the status, the evidence, the discrepancies with hypotheses, and the run IDs. **[Documented:
   SPEC.md §10; REPRODUCTION.md header.]**
2. **Generated tables.** Tables between `<!-- BEGIN AUTO:<code> -->` and `<!-- END AUTO:<code> -->`
   markers are rewritten by `superposition report` from the cited run's `acceptance.json`.
   Everything else is written by hand. **[Documented: REPRODUCTION.md header.]**
3. **Cited runs.** Phase 1's `report` used the latest run, so a notebook run could silently
   swap the evidence under the narrative (phase-1.md open issue 3). Phase 2 added pinning:
   `report --pin <exp>` records the run in `docs/cited_runs.yaml` with its profile, device,
   seed, commit, dirty flag and status summary, and refuses a dirty run unless `--allow-dirty`.
   An unpinned experiment falls back to its latest run, marked "(uncited)". **[Documented:
   D-202.]**
4. **Supplementary citations.** Rectification 2 needed a second cited run for some sections
   (E09's pilot, E11's held-out seed; later E02's seeds 1 and 2). Keys `<experiment>:<label>`
   render into `AUTO:<code>:<label>` blocks. The experiment's main citation and the status
   table do not change. **[Documented: D-R07; `docs/cited_runs.yaml` keys.]**
5. **Status vocabulary.** PASS, PARTIAL or FAIL per experiment, INFO for report-only rows. A
   single criterion can be a *graded PARTIAL*: it passes only in a degenerate form defined in
   advance, and does not count as passed. **[Documented: REPRODUCTION.md header; D-R05.]**
6. **Hand-written notes next to generated tables.** E02's seed result sits in a note under the
   generated status table, because the table shows only the cited seed. **[Documented:
   REPRODUCTION.md "Notes on the status table"; phase-6.md "Docs".]**
7. **Order of work.** Commit the code, run, write the narrative from the run, pin, then commit
   the docs. A cited run must come from a clean committed tree. **[Documented: CLAUDE.md
   "Conventions".]**

### How to do it by hand

1. **Look at the markers before editing.** Action: list the generated blocks so you never edit
   one by hand. **[Recommended]**

   ```bash
   grep -n "BEGIN AUTO" docs/REPRODUCTION.md
   ```

   Expected: one line per experiment and per supplementary key. Check: the codes match
   `docs/cited_runs.yaml`. If a block is missing, `report` cannot fill it; add the empty
   markers by hand first. **[Recommended]**
2. **Pin the run you wrote the narrative from.** **[Recommended]**

   ```bash
   uv run superposition report --pin intro_pentagon
   ```

   [Not verified: modifies tracked docs (docs/cited_runs.yaml and docs/REPRODUCTION.md)]

   For a supplementary run, name it and give a label:

   ```bash
   uv run superposition report --pin l1_removes_superposition --run 20260930T184744Z-e8c803d0 --label heldout
   ```

   [Not verified: modifies tracked docs; needs that run under results/]

   Expected: the pin is refused if the run's manifest says `dirty: true`. Do not reach for
   `--allow-dirty`; re-run from a clean tree instead. **[Recommended]**
3. **Check what changed.** **[Recommended]**

   ```bash
   git diff --stat docs/REPRODUCTION.md docs/cited_runs.yaml
   ```

   Expected: only the AUTO blocks and the pin changed. If a hand-written number now disagrees
   with a regenerated table, the narrative is stale: fix the narrative, never the table.
   **[Recommended]**
4. **Write each section from the run, not from memory.** Every number in the narrative should
   be findable in the cited run's `acceptance.json`, its tables, or a labeled source (a
   notebook, a reviewer's analysis). See [templates/REPRODUCTION_ENTRY.md](templates/REPRODUCTION_ENTRY.md).
   **[Recommended]**

## 2. DECISIONS.md: the log of choices

### Format, as used

- **Heading:** `### D-NNN · YYYY-MM-DD · title`. Numbers are grouped by phase (D-0xx … D-6xx);
  owner decisions from rectification 2 are D-R01–D-R07. **[Documented: DECISIONS.md headings.]**
- **Required fields:** decision, rationale, paper page, alternatives (SPEC §2). **[Documented:
  SPEC.md §2 "Decisions and defaults"; DECISIONS.md preamble.]**
- **Fields that appeared when needed:**
  - *Disclosure:* what had been seen before the decision (D-305, D-306, D-406, D-R04).
  - *Supersedes:* the entry it replaces (D-603 → D-306's cluster rule; D-R01 → D-408's sparse
    criterion; D-R04 → D-504's grading; D-R05 → D-505's criterion 3).
  - *Findings (review …):* the review finding that triggered it (D-603–D-606, D-608).
  - *Known before the change:* the expected outcome of a tightening (D-603).
  - *Owner decision* versus *Implementation (mine)*: what the owner decided and what the session
    chose to put it into practice (D-R01–D-R05).

  **[Documented: the cited entries.]**
- **Append-only.** A superseded decision gets a new entry that points back. D-114 withdrew
  D-113's premise this way. **[Documented: DECISIONS.md preamble; D-114.]**
- **In-place annotations, with a note.** Some entries were edited after the fact: three
  runtimes were marked "(estimated, not measured)" (D-402, D-407, D-504, under D-R06), and two
  factual errors were corrected with a note (D-404's cosine signs, R2-11; D-408's estimator
  claim, R3-05). The decisions themselves were not changed. **[Documented: D-R06; phase-6.md
  "Corrected in place"; D-408 text.]** So "append-only" held for decisions, not for every
  sentence. **[Reconstructed: from the annotated entries above.]**
- **Committed before the runs that apply them.** The owner decisions say so in their preamble,
  and D-603, D-605 and D-R04 each say their rule was fixed before the grading run.
  **[Documented: DECISIONS.md "Rectification session 2" preamble; D-603; D-605; D-R04.]**

### How to do it by hand

1. Write the entry the moment you make the choice, before the run that depends on it.
   **[Recommended]**
2. Fill every field of [templates/DECISION_ENTRY.md](templates/DECISION_ENTRY.md), including
   "what I had already seen". An empty disclosure field should mean "nothing", not "I forgot".
   **[Recommended]**
3. Commit the entry on its own (`docs: record …`). The commit date is your proof that the rule
   came first. **[Recommended]**
4. To change a decision, add a new entry with *Supersedes*. To fix a typo or a wrong fact,
   edit in place and add a dated note saying what changed. **[Recommended]**

```bash
grep -n "^### D-" docs/DECISIONS.md
```

Expected: one line per entry, in order. Check: every ID cited in REPRODUCTION.md appears here.
**[Recommended]**

## 3. The phase reports

### What SPEC asked for

SPEC §11 "Closing every phase": run `make lint test` and the quick profile of every experiment
touched; publish figures; update REPRODUCTION.md, CHANGELOG.md and the PLAN.md checkboxes;
commit; then write `docs/progress/phase-N.md` with what was built, the test status, an
acceptance table, decisions made, open issues, and a short "what the owner should look at and
learn from this phase" section with three questions. **[Documented: SPEC.md §11.]**

### What the reports actually contain

| Section | Example | Why it was added |
|---|---|---|
| Header line: date, commits, SPEC sections | every report | traceability **[Documented]** |
| Banner notes | phase-4.md and phase-5.md: "Phase 3 is not finished" | the reader must know the phase ran on an unfinished base (D-401) **[Documented]** |
| Commit note | phase-2.md (`e3d80a3`), phase-4.md (`7bde9b1`, `7ac6491`) | commits from outside the session had to be disclosed **[Documented]** |
| Before changing anything | phase-3.md, rectification.md, phase-6.md | who else was writing to the repo; what external commits changed **[Documented]** |
| Test status with commands and times | every report | the reader can re-run each tier **[Documented]** |
| Acceptance table and exit-criteria table | every report | SPEC §11; exit criteria verified, not assumed **[Documented]** |
| Decisions made | every report | an index into DECISIONS.md **[Documented]** |
| Open issues, numbered, carried over | every report | nothing silently dropped; items marked "your decision" **[Documented]** |
| What the owner should look at, three questions | every report | SPEC §11 **[Documented]** |
| Definition of done table | phase-6.md | SPEC §12 **[Documented]** |

**[Documented: the reports in `docs/progress/`.]** The reports are written in the first person
by the session, and address the owner as "you" when a decision is the owner's (for example
phase-4.md open issue 1, phase-6.md open issues 1–2). **[Documented: those reports.]**

**Commit notes in practice.** Each report that had external commits named them and said what
they contained. phase-2.md says `e3d80a3` holds exactly the intended docs plus a local
`.claude/launch.json`. phase-3.md lists `92525fc` and `5e50b46` and what each swept up.
phase-6.md states that none of its 22 commits came from outside the session.
**[Documented: those reports.]** Before writing that sentence, the Phase 6 session counted its
commits and their authors since `b50d062`. **[Documented: S 1584c849 @ 11:48Z.]**

### How to do it by hand

1. Copy [templates/PHASE_REPORT.md](templates/PHASE_REPORT.md) at the start of the phase, not
   the end, and fill "Before changing anything" first. **[Recommended]**
2. Fill the test table only with commands you ran and times you measured. Write "not run" or
   "estimated" otherwise (D-R06). **[Recommended]**
3. Carry every open issue forward until it is resolved, with the phase that resolved it.
   **[Recommended]**
4. Write the three questions as a test of yourself: if you cannot answer one, that finding is
   not ready to publish (section 6). **[Recommended]**

## 4. The independent review

### How it was done

1. **Requirement.** SPEC §11 Phase 6: a reviewer with fresh context re-derives each experiment
   from the cited pages, then audits the implementation and the evidence, looking adversarially
   for bugs and overclaims; findings and fixes go in REVIEW.md. **[Documented: SPEC.md §11.]**
2. **Setup.** Four reviewer agents, each of which had seen none of the build's history, took one group:
   the engine with E00–E02, E03–E05, E06–E08, E09–E11. A fifth reviewed E12 after it was
   written. They worked read-only, without the reference implementations, and returned
   numbered findings with evidence. **[Documented: D-607.]** They were Claude Code subagents
   spawned by the Phase 6 session at 20:17–20:18Z, the fifth at 06:50Z the next day.
   **[Documented: S 1584c849 @ 2026-09-30T20:17Z, 2026-10-01T06:50Z.]** Two of them were
   interrupted by a session limit and resumed with their context. **[Documented: phase-6.md
   "Before changing anything"; S 1584c849 @ 20:28Z.]**
3. **Output.** 59 findings: 0 critical, 12 major, 29 minor, 18 notes. **[Documented: REVIEW.md
   reviewer table; phase-6.md.]**
4. **Resolutions.** *Fixed* means code changed, the experiment was re-run from a clean commit,
   and the new run is cited. *Corrected* means a doc statement was wrong or too strong and was
   rewritten. *Open* means listed but not resolved. **[Documented: REVIEW.md "How to read the
   resolutions".]**
5. **Reviewers' numbers.** The reviewers' own scripts are not in the repository. A number that
   only they produced is labeled as theirs; a number the docs rely on was re-computed by a
   logged run first. **[Documented: D-607 "Caveat"; REVIEW.md preamble.]**
6. **What it changed.** E03 PASS 4/4 → PARTIAL 3/4; E02's n = 3 agreement shown not to hold
   across seeds (status left to the owner); E09 still PASS on tightened criteria; E12's
   write-up separating landscape from training; overclaims corrected in E00–E02, E04–E11 and
   the README; untracked code made a run dirty. **[Documented: REVIEW.md "What the review
   changed".]**
7. **What was not done.** SPEC §11 also asks to pair every experiment with an independent
   verifier during the build. Phase 3 did not, because its instructions were not to start
   agents unasked; the end-of-build review was the only one. **[Documented: SPEC.md §11
   "Parallelism"; phase-3.md open issue 11.]**

### How to do it by hand

You cannot un-know your own code, but you can imitate fresh context. **[Recommended]**

1. Wait at least a few days after finishing an experiment. **[Recommended]**
2. Re-derive the claim from the paper's pages first, on paper, before opening the code or the
   report. Write down what you would accept as evidence. **[Recommended]**
3. Then read the code, and check every number in the report against the run it cites.
   **[Recommended]**
4. Use the checklist in section 5. Log each finding with an ID, a severity (critical, major,
   minor, note) and, later, a resolution (Fixed, Corrected, Open). **[Recommended]**
5. If anyone else will read your work, ask them to do steps 2–3, and give them only the paper
   and the repository. **[Recommended]**

## 5. What a reviewer looks for

Each row below is a pattern that REVIEW.md actually found. Use the last column as your own
review question. **[Recommended, derived from the Documented findings named in each row.]**

| Pattern | Finding in this repo | Question to ask |
|---|---|---|
| Overclaim | "The neuron picture of p. 37 reproduces" when two details do not (R3-01); a README caption claiming three polytopes (R2-02); "confirms the window" (R4-06) | Does each sentence say exactly what the run shows, and no more? |
| Hypothesis stated as fact | E00's 0.0196 explanation under a "what it is" heading (R1-05) | Is every explanation labeled as a hypothesis unless tested? |
| Double counting | one Dᵢ band counted as two clusters (R2-01) | Can one observation satisfy two bins of a criterion at once? |
| Pattern versus mechanism | a motif detector that matched weights, not roles (R4-01); a role that random weights receive half the time (R4-02) | Would the detector fire on an untrained model? Is there a baseline? |
| Explanation contradicted by its own numbers | "dead sides" as the cause, with 111 sides alive (R3-02) | Do the diagnostic's numbers account for the whole gap? |
| Seed dependence | E02's n = 3 agreement on one seed (R1-01); cell maps that depend on starting signs (R5-07) | Has the pass been seen on a seed you did not tune on? |
| Wrong attribution | stuck models credited to the discard rule, not the starting sign (R1-02); a correlated-feature page cited for independent sets (R2-03); training effects called activation effects (R5-01) | Is the cause you name the one your evidence isolates? |
| Selection on the graded quantity | E09's regime chosen on the graded count (R4-10); E10's oracle selection (D-504) | Was any choice made by looking at the thing being graded? |
| Noisy or unpaired comparisons | best restarts chosen on noise (R1-06, R2-10, R3-06); unpaired variants (R3-07); estimates on different samples (R4-06) | Is the difference larger than the noise, on common samples? |
| A test that cannot show the claim | Spearman between two quantities that both rise with sparsity (R3-12); a test pinning a failing outcome (R3-14) | Could this test pass if the claim were false? |
| Comparison across settings | a sidestepping row comparing two sparsities (R3-04) | Are both sides of every comparison at the same setting (footnote 17)? |
| Untraceable number | notebook or uncited numbers in E02 (R1-07); bias medians with no source (R3-15); a run no longer on disk (R4-07) | Can you point to the run ID behind each number? |
| Provenance hole | untracked code not marking a run dirty (R1-10) | Could the code that ran differ from the commit recorded? |
| Stale string | README's "latest runs" (R1-09); a stale runtime comment (R5-04); "E12 arrives in Phase 6" found in the fresh clone (phase-6.md) | Does every sentence describe the current state? |

**[Documented: REVIEW.md, the finding IDs in the table; phase-6.md "Fresh-clone test".]**

Stale strings outlive reviews. At the starting commit of this cookbook (`2603068`), the
Makefile's help text for `make report` still says it regenerates tables "from the latest runs",
the phrase R1-09 corrected in the README; per D-202 it uses the cited runs. PLAN.md's Phase 3
entry still shows E03 as PASS 4/4 with the superseded run; the re-grade is recorded only under
Phase 6. **[Documented: `Makefile` target `report`; PLAN.md "Phase 3" and "Phase 6" sections.]**

## 6. Publishing honestly

The planning conversation gave four rules for publishing: disclose AI assistance precisely;
present the work honestly, as one early-career researcher's; write results in your own words;
publish only findings you can explain. **[Planning conversation, per owner]** This section
applies them, with this repository as the worked example.

### 6.1 Disclose AI assistance precisely

"Used AI tools" is not precise. Say which tool did which part, and what you did. Here is the
division of labor the evidence supports:

| Part | Who | Evidence |
|---|---|---|
| SPEC.md | drafted with Claude in a planning conversation; committed by the owner | COOKBOOK_SPEC §1; `7ac9cef` **[Documented]** |
| Code, tests, notebooks, lab, docs, phase reports | written by Claude Code sessions | COOKBOOK_SPEC §1; `Co-Authored-By` trailers on 118 of 131 commits (01_timeline) **[Documented]** |
| Phase prompts and their stop conditions | the owner | S 35cab3f0 @ 08:06Z; S 92884087 @ 16:29Z; S 1584c849 @ 2026-09-30T20:13Z **[Documented]** |
| Choices within a phase (D-001 … D-608) | the sessions, logged for the owner's review | SPEC.md §11 "Decide, log the decision"; DECISIONS.md **[Documented]** |
| D-R01–D-R06 | the owner; the implementation of each was the session's | DECISIONS.md "Rectification session 2" preamble **[Documented]** |
| Consulting the reference implementations | allowed by the owner; done by a session | D-R01 **[Documented]** |
| The independent review | five Claude Code subagents, not human reviewers | D-607; S 1584c849 @ 20:17Z **[Documented]** |
| Push and CI fixes | at the owner's request | S 522eb237 @ 19:42Z; S 1584c849 @ 12:09Z **[Documented]** |
| E02 status, E03 threshold | open, for the owner | phase-6.md open issues 1–2 **[Documented]** |
| How closely the owner read each result between sessions | unknown | no record outside the transcripts |

**What the repository says today.** README calls the work "an independent replication" and
mentions "an independent review by five fresh-context reviewers". It does not say that Claude
Code wrote the code, or that the reviewers were AI agents; REVIEW.md and D-607 say "agent".
CITATION.cff still lists "superposition contributors" as the author (D-006, an open owner
placeholder). **[Documented: README.md lines 1–21; CITATION.cff; D-006; phase-6.md open issue
8.]** "Independent" in the README means independent of the reference implementations (SPEC §2),
which a reader can easily misread. **[Reconstructed: from README "Related implementations" and
SPEC §2 "Independence".]**

**Disclosures that are already good models.**
- README "Related implementations" names both reference implementations, says they were not
  used as sources, and points to DECISIONS for the consultation. **[Documented: README.md.]**
- D-R01 records exactly what was read in each reference, when, with what tool, and that nothing
  was copied. **[Documented: D-R01 "Consulted".]**
- D-305, D-306 and D-R04 say what had been seen before a criterion or rule was fixed.
  **[Documented.]**

**A disclosure paragraph to adapt** **[Recommended]** (fill the brackets; delete what is not
true):

> This repository was built with Claude Code, Anthropic's coding agent. I wrote the research
> plan [with / without] Claude's help in [SPEC.md]. Claude Code wrote the code, tests,
> notebooks and first drafts of the reports, working phase by phase from that plan. I
> directed each phase, made the decisions marked "owner decision" in docs/DECISIONS.md, and
> [describe your review of the results]. The "independent review" in docs/REVIEW.md was done by
> five Claude Code agents with fresh context, not by human reviewers. The summaries on this
> page are my own words. Commits made by Claude Code carry a `Co-Authored-By: Claude` trailer.

Put it at the top of the README and of any page that presents results, not in a footnote.
**[Recommended]**

### 6.2 Present it as one early-career researcher's work

- Say what it is: a replication of a 2022 paper's toy models, on a laptop CPU, with quick
  profiles only (full profiles not run, D-R06). **[Recommended]**
- Lead with the status table as it stands: seven PASS, five PARTIAL, E12 report-only, and E02's
  open seed question. Do not lead with the gallery. **[Recommended; counts Documented in
  phase-6.md "Acceptance table".]**
- Claim no novelty. The extension (E12) and the open questions are exercises, not
  contributions, until a literature check says otherwise. **[Recommended]**
- If the site carries an organization's name or logo, say what role that organization had in
  the work. **[Recommended]**

### 6.3 Write results in your own words

The narrative in REPRODUCTION.md and every phase report were written by Claude Code sessions.
**[Documented: COOKBOOK_SPEC §1; commit trailers.]** Reusing that text as yours would misstate
authorship. **[Recommended]**

1. For each experiment you publish, close REPRODUCTION.md and write three sentences: the
   claim, what you found, and the main caveat. **[Recommended]**
2. Then check each number against the cited run's acceptance table, not against the old
   narrative. **[Recommended]**
3. Quote the repository's text only with quotation marks and a link. **[Recommended]**

### 6.4 Publish only findings you can explain

A finding is ready when you can explain it without the repository open. **[Recommended]**

1. Answer the three questions at the end of the relevant phase report, on paper.
   **[Recommended]**
2. Do the "Explain it back" drill of the experiment's recipe card in
   [experiments/](experiments/). **[Recommended]**
3. If you cannot explain a mechanism (for example, why same-sign starts get stuck in E02, which
   is open in REVIEW.md), publish it as an open question, not as a finding. **[Recommended]**

### Checklist before publishing **[Recommended]**

- [ ] The disclosure paragraph is at the top and is precise about tools, people and reviewers.
- [ ] CITATION.cff and LICENSE name the real author.
- [ ] The status table, including PARTIALs and open questions, is the first thing shown.
- [ ] Every published sentence is in your own words, and every number has a run ID.
- [ ] You can answer each phase's three questions for every finding you publish.
- [ ] The section 5 checklist has been run on the published text.
