# Matura z historii — CKE dataset

743 scored tasks from 19 official CKE *historia* (poziom rozszerzony) sessions, with the original PDFs, scoring rules and figure crops for visual tasks.

Rebuild from the cached PDFs (about 1 min):

```bash
uv run scripts/prepare_matura_historia.py --offline   # from raw/ only
uv run scripts/prepare_matura_historia.py             # also downloads missing PDFs from cke.gov.pl
```

## Sessions

- **Formula 2023:** maj 2023–2026, pokazowy 2022-03, diagnostyczny 2022-12 and 2024-12, próbny 2026-01.
- **Formula 2015:** maj 2015–2023, próbny 2020-04, diagnostyczny 2021-03.
- CKE does not publish June/August historia papers, and historia has no poziom podstawowy.
- `raw/` also holds 5 official "660" `.docx` papers for blind students, which describe maps and photos in text. For 3 sessions whose task numbering matches, this text is attached to 59 visual items as `adapted_660_text`.

## Splits (whole sessions held out)

| split | sessions | items |
|---|---|---|
| test | 2026-maj, 2025-maj | 81 |
| dev | 2024-maj | 42 |
| train | everything else | 620 |

No source text is shared between test/dev and train.

| split | open_short | open_long | closed_abcd | true_false | matching | essay | needs_visual | auto_gradable |
|---|---|---|---|---|---|---|---|---|
| train | 426 | 6 | 66 | 40 | 12 | 70 | 345 | 103 |
| dev | 32 | 1 | 3 | 2 | 1 | 3 | 30 | 6 |
| test | 57 | 2 | 6 | 7 | 3 | 6 | 49 | 15 |

Most points are in open questions, so a real score needs rubric grading (e.g. an LLM judge using `scoring`), not only the auto-gradable keys.

## Item fields

`id, session_key, year, session, formula, exam_date, level, task, group, topic (essays), choice_group, max_points, type, context, question, options / statements, requires_justification, answer, answer_key, scoring, scoring_notes, scoring_raw, auto_gradable, needs_visual, visual_reason, images, adapted_660_text, source, split`

- `choice_group`: `null` for normal items. Items the examinee chooses between share one string: every essay topic of a session gets `"<session_key>-essay"` (e.g. `"2024-maj-essay"`). The examinee writes ONE of them, so **a session score counts one item per `choice_group`** (e.g. the best- or the actually-chosen topic), never the sum of all topics. Counted this way every session totals exactly 60 points (formula 2023) or 50 (formula 2015); the build fails otherwise. Essay topics are the only choose-one tasks in these papers.
- `answer_key`: exact key for closed items (e.g. `"C"`). `auto_gradable` is false when the task also asks for a justification (*uzasadnij*).
- `answer`: the example answer from the zasady oceniania for open items. Essays have no answer; their full criteria are in `scoring_raw`.
- `images`: paths relative to this folder (`images/<item-id>-<n>.png`, 150 dpi crops of the figure with its title and legend).

## Known issues

- Tables are flattened cell by cell in the text. The PNG crops include them.
- The context/question split is heuristic. Only 3 items have a question that doesn't start with an instruction.
- `needs_visual` is conservative: it is set for every subtask of a group that contains a figure.
- `open_long` means 3 or more points.
- Some matching keys are free text with alternatives ("Kircholm / Chocim"), so compare them with normalised matching, not exact match.
- CKE's zasady disagree with the arkusz on points in 7 tasks; the arkusz values are kept. The 2025 essay is numbered differently in the two files. All of these are listed in `manifest.json` under `parse_warnings`.

`manifest.json` holds source URLs, SHA-256 of every PDF, counts, the split policy and all warnings.
