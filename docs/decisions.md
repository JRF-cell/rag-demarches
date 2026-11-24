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
