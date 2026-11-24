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

## D7. Train/test split

- **Date:** 2026-11-24
- **Status:** accepted, frozen

**Context.** Fine-tuning must never see the text behind a test question. Splitting by sheet is not enough on its own: some chunk bodies appear word for word in several sheets. With a plain random 80/20 split by sheet (seed 42), 609 of the 6 910 test chunks of at least 20 words have a body that also appears in a train sheet (`results/split/split_stats.json`).

**Decision.** Sheets sharing a normalised chunk body of at least 20 words are linked, and each connected group goes entirely to train or to test. Groups are assigned to test theme by theme (majority theme of the group) until each theme reaches 20 % of its sheets. Seed 42. The result is written once to `eval/split.json`; the script refuses to overwrite it with a different split.

**Rejected options.**
- Random 80/20 by sheet: simplest, but leaks shared text into train, as measured above.
- Group split at 90/10: more training sheets, but about one candidate question per test sheet, so less diversity in the evaluation set.

**Rationale.** Groups are small (2 485 groups, 2 307 single sheets, largest 25 sheets), so keeping them whole costs little balance. Theme stratification keeps every theme in the test set.

**Consequences.**
- 585 test sheets (7 645 chunks) and 2 288 train sheets (29 286 chunks). No test chunk of at least 20 words shares its body with train; 19 shorter test chunks do.
- Because groups are assigned whole, a theme can exceed 20 %: "Étranger - Europe" has 41 test sheets out of 166.
- The retrieval index stays the full corpus. The split only decides where evaluation questions come from and which sheets fine-tuning may use.
- PyTest checks that train and test are disjoint, cover the corpus, and share no long chunk body.

## D8. Relevance unit for retrieval metrics

- **Date:** 2026-11-24
- **Status:** accepted

**Context.** A question is generated from one chunk, but the same text can appear in other chunks: across situations of one sheet (for instance the FPE, FPT and FPH branches of F1690) or across sheets.

**Decision.** Relevance is defined on chunks. The relevant set of a question is its source chunk plus every chunk of the corpus with the same normalised body. It is stored with the question and can be narrowed by hand during review when the question names a specific situation. Sheet-level Recall and MRR are computed from the same labels and reported as secondary metrics.

**Rejected options.**
- Sheet level only: a sheet holds a median of 9 chunks, so finding the right sheet does not mean finding the passage that answers.
- Source chunk only: penalises a system that returns an identical copy of the right text.

**Consequences.**
- Recall@k counts a hit when any relevant chunk is in the top k; MRR uses the rank of the first relevant chunk.
- The share of questions with more than one relevant chunk is reported with the candidate set.

## D9. LLM for candidate questions

- **Date:** 2026-11-24
- **Status:** accepted

**Context.** Candidate questions are generated on test sheets only, within a zero budget.

**Decision.** Gemini Flash-Lite through the Google AI Studio free tier, called with the standard library (`urllib`). One call per sampled test sheet: it receives up to 5 chunks with their ids and returns 3 questions, each tied to one chunk. Responses are cached on disk by prompt hash, calls are paced to the rate limit, and HTTP 429 triggers a backoff. Gemini Flash is a fallback for sheets whose questions are poor.

**Rejected options.**
- Gemini Flash for everything: better French, but a much lower free daily cap, so the run would take several days.
- Qwen on a Kaggle GPU: no request cap, but weaker at rephrasing and from the same family as the answer generator.

**Consequences.**
- The judge (D4) is also a Gemini model. It grades answers produced by Qwen, not its own answers, but question style may favour it; this is listed as a limitation.
- The free tier may use prompts for training; the corpus is public under the Licence Ouverte.

## D10. Config format

- **Date:** 2026-11-24
- **Status:** accepted

**Decision.** YAML configs read with `yaml.safe_load` (PyYAML). Each experiment copies its config into `results/<name>/`.

**Rejected options.** TOML with the standard `tomllib` (no dependency, but a different format from the rest of the plan); JSON (no comments).
