# Matura z historii — CKE dataset

1795 scored tasks from official CKE *historia* papers, with the original PDFs, scoring rules and figure crops for visual tasks: 743 from 19 formula 2015/2023 sessions (poziom rozszerzony; these hold all dev/test items) and 1078 train-only items from 24 archival old-formula ("stara matura") papers, poziom podstawowy and rozszerzony, minus 26 core train items dropped by the leak check.

Rebuild from the cached PDFs (about 8 min, deterministic: two builds are byte-identical):

```bash
uv run scripts/prepare_matura_historia.py --offline   # core sessions, from raw/ only (drop --offline to download missing PDFs)
uv run scripts/prepare_matura_historia_archiwum.py --archive-dir data/matura-historia/archiwum-src --offline
                                                      # adds the archival sessions to train, then leak-filters all of train
uv run scripts/check_matura_historia_leaks.py         # re-check train vs dev/test (exit 1 on a leak; --near lists near misses)
```

Run the archival script after the core one: the core script rewrites `data.json` and `manifest.json` without the archival items and without the leak filter.

## Sessions

- **Formula 2023:** maj 2023–2026, pokazowy 2022-03, diagnostyczny 2022-12 and 2024-12, próbny 2026-01.
- **Formula 2015:** maj 2015–2023, próbny 2020-04, diagnostyczny 2021-03.
- CKE does not publish June/August historia papers, and historia has no poziom podstawowy in formula 2015/2023.
- **Old formula (train only):** maj 2010–2020 and czerwiec 2012, poziom podstawowy (P, 100 pts) and rozszerzony (R, 50 pts) = 24 papers, 1078 items. Their `formula` is `"stara"`, `level` is `"podstawowy"` or `"rozszerzony"`, and `session_key` keeps the level (`2015-maj-stara-P`). 2015–2020 old-formula sessions were for repeaters. PDFs are in `raw-archiwum/`; the full downloaded archive (2003 pilot to 2026 informators) with its catalogue is in `archiwum-src/`.
- Not in the dataset: old-formula papers before 2010, the deaf-candidate (`-ns`) variants, informators, the CKE task bank and extra material (not parsed), 2013-12 przykładowy and 2014-12 próbny formula 2015 (parsed, but multi-part tasks were merged; dropped) and maj 2024 formula 2015 (same sources as dev 2024-maj; dropped by the leak check). See [QC.md](QC.md).
- `raw/` also holds 5 official "660" `.docx` papers for blind students, which describe maps and photos in text. For 3 sessions whose task numbering matches, this text is attached to 59 visual items as `adapted_660_text`.

## Splits (whole sessions held out)

| split | sessions | items |
|---|---|---|
| test | 2026-maj, 2025-maj | 81 |
| dev | 2024-maj | 42 |
| train | 16 other formula 2015/2023 sessions + 24 old-formula papers | 1672 (594 + 1078) |

**Leak check.** A train item is dropped if its normalised question or context has difflib ratio >= 0.8 with a dev/test item's question or context, or if it shares >= 200 chars of source text with a dev/test item (one passage, or common runs of >= 30 chars summed when the same excerpt is quoted with different elisions). This dropped 26 core train items and 49 archival ones: all of maj 2024 formula 2015, plus 10 old-formula items that quote the same sources as dev/test (Nihil novi, Gall on Bolesław Śmiały, Konfederacja warszawska). After it, no train item is a near-duplicate of a dev/test item. The full lists are under `leak_check` and `archival.leak_check` in `manifest.json`.

| split | open_short | open_long | closed_abcd | true_false | matching | ordering | essay | needs_visual | auto_gradable |
|---|---|---|---|---|---|---|---|---|---|
| train | 1256 | 82 | 124 | 62 | 49 | 5 | 94 | 691 | 201 |
| ↳ formula 2015/2023 (594) | 406 | 4 | 65 | 40 | 9 | 0 | 70 | 331 | 99 |
| ↳ stara 2010–2014, P (391) | 334 | 35 | 0 | 0 | 22 | 0 | 0 | 99 | 17 |
| ↳ stara 2010–2014, R (173) | 156 | 1 | 0 | 0 | 1 | 3 | 12 | 59 | 1 |
| ↳ stara 2015–2020, P (341) | 221 | 41 | 48 | 15 | 16 | 0 | 0 | 132 | 70 |
| ↳ stara 2015–2020, R (173) | 139 | 1 | 11 | 7 | 1 | 2 | 12 | 70 | 14 |
| dev | 32 | 1 | 3 | 2 | 1 | 0 | 3 | 30 | 6 |
| test | 57 | 2 | 6 | 7 | 3 | 0 | 6 | 49 | 15 |

Old-formula items by level: podstawowy 732 (needs_visual 231, auto_gradable 87), rozszerzony 346 (129, 15). By era: 2010–2014 564 items, 2015–2020 514. The 2010–2014 papers number their choices (1–4) or ask to underline or write a letter, so those tasks are typed `open_short` with the key in `answer`; they have no `closed_abcd` / `true_false` items.

Most points are in open questions, so a real score needs rubric grading (e.g. an LLM judge using `scoring`), not only the auto-gradable keys.

## Item fields

`id, session_key, year, session, formula, exam_date, level, task, group, topic (essays), choice_group, max_points, type, context, question, options / statements, requires_justification, answer, answer_key, scoring, scoring_notes, scoring_raw, auto_gradable, needs_visual, visual_reason, images, adapted_660_text, source, split`

- `formula`: `"2023"`, `"2015"` or `"stara"` (old formula, train only). `level`: `"rozszerzony"` or `"podstawowy"` (old formula only).

- `choice_group`: `null` for normal items. Items the examinee chooses between share one string: every essay topic of a session gets `"<session_key>-essay"` (e.g. `"2024-maj-essay"`). The examinee writes ONE of them, so **a session score counts one item per `choice_group`** (e.g. the best- or the actually-chosen topic), never the sum of all topics. Counted this way every session totals exactly 60 points (formula 2023) or 50 (formula 2015); the build fails otherwise. Old-formula papers total 100 (P) or 50 (R) as parsed; a paper that does not is not shipped. The leak check runs after this, so 5 old-formula papers and some core train sessions are a few points short (listed in `manifest.json`). Essay topics are the only choose-one tasks in these papers.
- `answer_key`: exact key for closed items (e.g. `"C"`; a list for "choose two"; `{"1": "B", "2": "D"}` for two sentences with their own options, whose `options` are then `{"1": {"stem": …, "A": …}, …}`). `auto_gradable` is false when the task also asks for a justification (*uzasadnij*).
- `answer`: the example answer from the zasady oceniania for open items. Essays have no answer. Their `scoring` holds the criteria: formula 2015 has its own block per topic, formula 2023 has one criteria table shared by all topics. `scoring_raw` holds the whole block: per topic for formula 2015, all topics' requirements plus the criteria for formula 2023.
- `images`: paths relative to this folder (`images/<item-id>-<n>.png`). These are 150 dpi crops of the figure with its title, caption, legend and citation. A crop stops before the next source or the question, and is widened when the figure reaches into the page margin.

## Quality control

All 123 dev/test items were checked against the PDFs. The archival train sessions were validated per paper (point totals, every answer located under its own task in the zasady, 3+ items per paper checked by hand). See [QC.md](QC.md) for every problem found and fixed, and for what is still uncertain. Text that pymupdf reads in the wrong order is fixed by explicit, documented per-page overrides (`LAYOUT_OVERRIDES` in the script):
- two poem versions printed side by side (2024-maj zad15);
- stamp captions on one physical line (2026-maj zad13);
- genealogy trees (2024-maj zad11, 2025-maj zad5).

## Known issues

- Tables and vector charts are flattened cell by cell in the text. The PNG crops include them. Genealogy trees with an override are written one generation per line: `[box] | [box]`.
- Loose map labels (e.g. `2 1 3 4`, `PALESTYNA`) can appear in the context text.
- The context/question split is heuristic. Only 3 items have a question that doesn't start with an instruction.
- `needs_visual` is conservative: it is set for every subtask of a group that contains a figure.
- `open_long` means 3 or more points.
- Some matching keys are free text with alternatives ("Kircholm / Chocim"), so compare them with normalised matching, not exact match.
- Old-formula keys of 2010 closed tasks are stated only in the scoring rule, so their `answer` is that rule (e.g. "1 p. – za podkreślenie imienia Klejstenes (3)") and `answer_key` is null. Old-formula essays are scored by levels I–IV (0–20 pts); the level descriptors are in `scoring`.
- CKE's zasady disagree with the arkusz on points in 7 tasks; the arkusz values are kept. The 2025 essay is numbered differently in the two files. All of these are listed in `manifest.json` under `parse_warnings`.

`manifest.json` holds source URLs, SHA-256 of every PDF, counts, the split policy and all warnings.
