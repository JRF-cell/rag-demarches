# Decision log

Each entry follows a short ADR format: context, decision, rejected options, rationale, consequences.

## D1. Corpus scope

- **Date:** 2026-11-24
- **Status:** accepted

**Context.** The 3.5 snapshot of 2026-11-24 contains 2 997 F sheets (`results/corpus_stats_raw.json`). Their types differ a lot in content: 87 "Recherche guidée" sheets hold 121 body words in total, and 83 of them have none; 15 "Fiche avec liens externes" hold 788 words; 22 "Fiche Comment faire si conditionné" hold 185 937 words, have no theme and are life-event guides spanning several procedures. The two main types are "Fiche d'information conditionnée" (1 497 sheets, 3 652 468 words) and "Fiche Question-réponse conditionnée" (1 376 sheets, 1 397 457 words), spread over 11 themes.

**Decision.** Keep every "Fiche d'information conditionnée" and "Fiche Question-réponse conditionnée", across all themes. Drop the other F types, and all N and R files.

**Rejected options.**
- A few themes only: cheaper to encode, but retrieval becomes artificially easy (fewer neighbouring sheets compete), the choice of themes is arbitrary, and every off-topic question would end in a refusal.
- A stratified sample per theme: covers all themes at bounded cost, but leaves holes; a question can target a sheet that is absent, which biases both recall and refusal metrics.

**Rationale.** Retrieval metrics are only meaningful on the real competition between sheets. The dropped types either carry almost no text or are life-event guides without a theme; their overlap with other sheets has not been measured.

**Consequences.**
- Encoding and fine-tuning cost is the highest of the three options; encoding throughput is measured in phase 3, not estimated.
- The parser filters on the `type` attribute of `<Publication>`.
- The scope is fixed before the train/test split and does not change afterwards.

## D2. Chunking strategy

- **Date:** 2026-11-24
- **Status:** accepted

**Context.** The 2 873 in-scope sheets parse into 28 070 sections (`results/section_stats.json`). Section length is very uneven: median 94 words, p90 357, p99 1 134, max 19 066. 4 863 sections exceed 250 words and 1 622 have fewer than 20.

**Decision.** One chunk per section (chapter or sub-chapter, prefixed by its situation when there is one). A section longer than 250 words is cut at line boundaries by greedy packing; a single line longer than 250 words is cut into 250-word slices. No overlap. Every chunk carries a context header `sheet title > section path`, which is part of the indexed text. Short sections are kept as they are.

**Rejected options.**
- Same strategy with a one-line overlap between pieces of a split section: overlap only affects split sections and creates near-duplicates that inflate the number of relevant chunks per question. Left as a variant to measure in the retrieval experiments.
- Fixed 200-word windows with a 150-word stride, ignoring structure: 31 977 chunks but 6 176 294 indexed words instead of 4 721 094, and a window can mix two sections or two situations. Kept only as a possible baseline.
- Other limits: 150 words gives 48 385 chunks; 400 words gives 31 799 chunks with longer inputs for the embedding model.

**Rationale.** A chunk never spans two sections, so a citation points to one section of one sheet, and the ground truth of the evaluation set is defined on stable units.

**Consequences.**
- 36 931 chunks; body length median 109 words, max 250; with header, max 297 words (`results/corpus_stats.json`).
- The token length of chunks must be checked against the input limit of the embedding model chosen in D3.
- Near-identical sections exist across situations of a same sheet (for instance public service branches), so several chunks can be relevant to one question.
