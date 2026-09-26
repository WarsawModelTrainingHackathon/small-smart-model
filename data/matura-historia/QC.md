# QC of dev + test (2024-maj, 2025-maj, 2026-maj)

We checked all 123 dev/test items (dev 42, test 81) against the arkusz and zasady PDFs in `raw/`. Everything was fixed in `scripts/prepare_matura_historia.py`, either as a parser fix or as a documented layout override. The dataset was then rebuilt with `--offline`.

## What was checked, per item

- `answer_key` against the zasady, and consistent with the numbering of `options` / `statements`.
- `options` / `statements` / `question`: complete and correct.
- `context`: not truncated, not mixed with another task, side-by-side columns not interleaved.
- `max_points` against the arkusz.
- Every figure crop was viewed: 52 unique crops for dev/test. We looked for the right figure, nothing cut at the top, bottom or sides, and no spill into the next source.
- `answer` / `scoring` / `scoring_notes` / `scoring_raw`: complete, belong to the task, no leaked page footers, no truncated words, essays not empty.

## Result

- **Answer keys:** all dev/test `answer_key` and `max_points` values were already correct. They are unchanged by this QC (verified against the previous build: 0 changes in 123 items).
  - 2024-maj zad15.2: the key `A` was right, but the context showed the poems swapped. See the first row below.
- **Point totals:** every session totals 60 (formula 2023) or 50 (formula 2015), counting one item per `choice_group`. The build now fails if this is not the case.

## Problems found and fixed

| # | Item(s) | Problem | Fix |
|---|---|---|---|
| a | 2024-maj zad15.1, 15.2 | Wersja A (left column, x 76–294, the critical mazurka) and Wersja B (right, x 303–522, the praising one) were interleaved line by line. The A text followed the "Wersja B:" label, so the key 15.2 = A looked wrong. | Layout override `('2024-maj', p.18)`, `columns` split at x=298, band y 270–462. The text is now "Wersja A:" + left column, then "Wersja B:" + right column. |
| b | 2026-maj zad13 | Stamp captions B (x 71–224) and C (x 252–508) are one physical PDF line at y 466. They were merged into "B C Tłumaczenie: Wojna między siłami Tłumaczenie napisu…". | Override `('2026-maj', p.14)`, `columns` split at x=240. Now reads "B – Tłumaczenie: Wojna między siłami króla i parlamentu." / "C – Tłumaczenie napisu na pieczęci: Filadelfia, 29 maja 1976." Key {1: C, 2: A} was already correct. |
| c | 2024-maj zad2 (scoring), 2025-maj zad1.2 (answer), 2026-maj zad1 (scoring); in train: 2022-maj-f2015 zad1, 2023-maj-f2015 zad2.1, 2022-grudzien zad1.2, 2024-grudzien zad2.1, 2026-styczen zad1, 2022-pokazowy zad1.2 | Page-2 legal footnotes leaked into the text: "1 Rozporządzenie Ministra … Dz.U. … Załącznik nr 1/3", the "Zwracamy uwagę, że w liceum … od roku szkolnego 2025/2026" note, and the Komunikat URL. | `footnote_y()` finds the footnote area: a line starting with a superscript number in the lower half of the page. Everything below it on that page is dropped. It triggers only on p.2 of 12 zasady files, and none of those footnote areas hold task content. |
| d | 2024-maj zad4; train 2021-maj-f2015 zad3 | "Rozwiązaniem greckim/rzymskim" was cut to "m greckim". The `Rozwiązanie` header regex had no word boundary. | Added `\b` to `SOLUTION_M` and `NOTE_M`. Also, "Uwaga!" no longer leaves a leading "!". |
| e | 2025-maj zad25 t1–t3; train 2021-marzec-diagnostyczny, 2022-grudzien-diagnostyczny, 2024-grudzien-diagnostyczny essays | Empty essay `scoring`. The header is uppercase "KRYTERIA OCENIANIA WYPOWIEDZI ARGUMENTACYJNEJ", and 2021-03 has no criteria header at all. | `SCORING_M` is now case-insensitive. New `essay_scoring()`: for formula 2023, `scoring` is the shared criteria (A. NARRACJA, B. SPÓJNOŚĆ, dodatkowe informacje). For formula 2015, each topic gets its own block, cut at the `Temat N` lines (2021-03: at "Wymagania egzaminacyjne"), and `scoring_raw` is that topic's block. The build fails if an essay topic has no criteria. |
| f | 2025-maj zad5.1, 5.2 | The genealogy crop was missing its top box, Brzetysław I (the answer to fragment B). The box is an image block only 29 pt tall and was filtered out as too small. | Image blocks of at least 80×18 pt now count as figures, and the crop includes the "Źródło 1." title. |
| f′ | 2025-maj zad5.x, 2024-maj zad11.x (context) | Genealogy tree text was scrambled: dates detached from rulers, 11 `[OBRAZ]` markers, and "†" lost. | Overrides `('2025-maj', p.8)` and `('2024-maj', p.14)`, mode `boxes`: one line per generation, `[box] \| [box]`, and one `[OBRAZ]`. The Symbol-font private-use glyph U+F085 is mapped to "†"; this also fixes 2022-grudzien zad14. |
| g | 2024-maj zad2 (map), zad16 (cartoon callouts); 2025-maj zad17 (callouts) | Crops were clipped to x 45…W−45, cutting maps and callouts that stick into the margin. | The crop is widened to the figure and caption boxes (up to 5 pt from the page edge). |
| g′ | 2026-maj zad22 | This vector bar chart has no raster image, so it was found by the heading keyword. The crop took the whole task, including the question and answer lines. | Keyword-only visuals now crop only the sources, not the question. |
| g″ | 2026-maj zad1, 4, 12, 14, 15, 16, 18, 19, 23, 25; 2024-maj zad9, 19, 23; 2025-maj zad1 | Crops ran into the next "Źródło"/"Zadanie" heading or the question. Some cut a line in half (a 2-line title in 2024-maj zad9, the last caption in 2024-maj zad19), or missed the citation under a caption box (2026-maj zad23). | Crop = figure + the title lines directly above (no gap, at most 55 pt) + the caption/legend/citation lines that follow without a gap. It stops at the next source, task header or instruction and never cuts into a neighbouring line. |

Smaller text fixes, applied everywhere:
- Word-final hyphen followed by a repeated hyphen ("królewsko-\n-hiszpański") is joined into one word.
- P/F answer boxes no longer glue onto the next statement ("F Opisana…"). This also fixed `statements['1']` in 2022-pokazowy zad14 (train).
- Source headings ("Źródło 2. Fragment …") and "Na podstawie:" citations now start their own line.
- A zasady bullet drawn as its own line ("•\ntext") is joined to its text.

Other changes:
- `choice_group` was added: essay topics share `<session>-essay`.
- `scripts/__pycache__` is no longer tracked.
- Every image was re-rendered with the new crop rules. Renders from pymupdf 1.28 differ byte-wise from the old ones even where the crop box did not change.

## Checked and left as is (faithful to the PDF, or cosmetic)

- 2025-maj zad12: the answer swaps "źródło 1/2". The typos "N podstawie" (zad8) and "Teksy źródłowe" (zad7) are also CKE's own text.
- Map labels leak into the context as loose words: 2026-maj zad1 "2 1 3 4", zad4 "PALESTYNA MORZE ŚRÓDZIEMNE", 2025-maj zad1/14/16 "A B C". Figure labels are kept, because the questions refer to them.
- Tables and vector charts are flattened into a run of text: 2026-maj zad11 and zad22, 2024-maj zad23 (the referendum table sits under the poster and is in the text, not in the crop). 2026-maj zad22 has no `[OBRAZ]` marker, because the chart is vector graphics.
- 2024-maj zad6: the woodcut labels read in PDF order ("Ty ochraniaj Ty módl się A ty pracuj"), not left to right. The crop is correct.
- 2025-maj zad7: the map legend is a separate image, so the context has 2 `[OBRAZ]` markers. 2025-maj zad4: context is empty and the table is in `question`; nothing is missing.
- Formula 2023 essays: `scoring_raw` holds the requirements of all 3 topics plus the shared criteria, so it is identical across topics. The criteria really are shared.

## Remaining uncertainty

- We checked that the linearised criterion-A table and the "Wymagania" lists in essay `scoring` / `scoring_raw` are complete and have no leaks. We did not compare them word for word with the rotated or side-by-side layout of the zasady.
- `adapted_660_text` was not checked. It is not used in dev/test sessions.
- Train sessions were not QC'd item by item. They received the same parser fixes, and we spot-checked their crops on contact sheets. Some train crops may still show a thin rule from the previous line at the top edge, or miss a title printed far above the figure (e.g. 2017-maj-f2015 zad19).
- Layout overrides are page- and band-specific. If a PDF in `raw/` is replaced, re-verify them; the SHA-256 of every PDF is in `manifest.json`.
