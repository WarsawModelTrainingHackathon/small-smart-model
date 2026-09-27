# Exam package and answers.json

Official process (Anna Olchowik, 26 Sep ~14:03; [JSON guide](https://matura-json-guide.ania-olchowik.chatgpt.site/)).

## You do not host a scoring server

Organisers do **not** SSH into your vLLM. They send a **zip**; you return **`answers.json`**.

Harness may run on Forgehand, a laptop, Modal, AWS, etc. Bind `0.0.0.0` only if **you** want a stage demo.

## Package

| File | Role |
|---|---|
| `exam.json` | `exam_id`, `items[]` (`id`, `group`, `max_points`, `question`, `source_text`, `images[]`, `answer_format`) |
| `images/*.png` | Resolve `images[].path` next to `exam.json`. Send **pixels**, not the filename. `sha256` for integrity. |
| `answers-template.json` | Same ids, empty `answer` strings |
| `README.md` | Pack notes |

`input_format`: `separate-text-and-images-v1`. Mock: **history-2023-mock-v1**, 37 items, 60 points, ~19 PNGs.

## Output

```json
{
  "exam_id": "history-2023-mock-v1",
  "answers": [
    {"id": "1", "answer": "…"},
    {"id": "2.1", "answer": "…"}
  ]
}
```

- Keep `exam_id` and every `id`. Top-level keys: only `exam_id`, `answers`. Entry keys: only `id`, `answer`.
- All `answer` values are **strings** (including A/B/C and P/F).
- Essay: one id (mock `"26"`), topic number + ≥300 words.
- No traces/CoT in the file. Team code + solution name on the **web** form.

## Map from our eval harness

CKE `data.json` ids (`2024-maj-R-zad22.2`) **are not** mock ids (`"2.1"`). Write an adapter: exam item → `eval_matura` / generate → string → template slot. Do not submit `report.json`.

Sunday package: **new** `exam_id`; do not reuse the mock id.
