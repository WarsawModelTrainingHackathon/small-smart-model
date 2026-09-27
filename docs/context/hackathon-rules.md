# Warsaw Model Trainers — rules & context (updated Saturday 26 Sep 2026)

Organiser chat + Anna’s testing guide. Older Fri docs are superseded where they conflict. Subject is **history (historia) only** — **not geography**. Confirmed by organisers 25 Sep 21:40.

## What you showcase vs what you submit

**Scoring is not a public model server.** You do **not** need vLLM/Gradio on the internet for the leaderboard.

1. Download an **exam package**: `exam.json` + `images/*.png` + `answers-template.json` ([guide](https://matura-json-guide.ania-olchowik.chatgpt.site/)).
2. Run the model/harness **wherever you want** (laptop, Forgehand/AWS, Modal). Organisers will **not** call your GPU.
3. Fill the template → **`answers.json` only**. Upload on their page with **team code** + **solution name**.
4. Format check immediately; **LLM grading ~every 30 minutes** (may slip if credits run out).
5. **Sunday stage** is a short **talk** (pipeline + scores). That is separate from `answers.json`. A demo URL is optional for the jury, not the official test.

`k3exam.py` / TEAM_KEY auto-query of your server: **old idea, dropped**. Details were promised Saturday before lunch (~14:00); Anna published the JSON flow ~14:03.

Mock package: May 2023 history, **37 items / 60 points**, `exam_id` like `history-2023-mock-v1`. Final exam has a **new** `exam_id` and template.

## Size (Saturday update — supersedes “LoRA never counts”)

Piotr / Paulina 26 Sep afternoon:

- **Base** (quantized, as you would run it): **≤ 8 GB** on disk.
- **After fine-tuning**: **≤ 8.8 GB** (8 GB + 10%). Not further.
- Slight overflow from LoRA is tolerated; **not** extra copies, voting ensembles, or bumping precision to inflate the submitted weights.
- Parameter **count** does not matter; **weight size on disk** does.

Friday text (“LoRA and RAG KB do not count at all”) is **softer now**: RAG KB still not the 8 GB blob; **merged** weights after FT must stay **≤ 8.8 GB**.

## Exam content

- **History matura**, no geography surprise.
- **Each question separately.** Text and **PNG separately** if the item needs a figure (text-only models can still attempt).
- Not one PDF to parse. No PDF extraction required.
- Essays: full text in the essay id (mock uses `"26"`); include **chosen topic number**, **≥ 300 words**.
- Answers: **Polish strings** only. Follow `answer_format` for syntax (examples are **not** keys). Empty `""` if skipped. UTF-8 JSON, **≤ 1 MiB**, ≤ 100k chars per answer.

## Tracks and submissions

- Teams **may enter all prize categories**.
- **One submission per track.**
- Progress track still needs a **base** vs **trained** story; baseline = best **untouched** model alone.

## Offline / APIs

Unchanged: **Sunday answers** from **your** open model + local harness. No ChatGPT/Claude/Gemini/web search in the solution. Closed APIs OK **while building**.

Training cutoff / repo / `SOURCE.md` / roster: see original Fri sheet unless organisers post otherwise. Inference can take as long as you need **off-stage**; stage talk is still short.

## Contacts / compute

- Telegram: t.me/warsawmodeltrainers
- Forgehand: Michał (orange hat) — SSH keys in Settings.
- Nebius: Gleb. Solari codes: Telegram (rotate when expired).
- Mentors Sat 12:00–18:00.

Guide: https://matura-json-guide.ania-olchowik.chatgpt.site/
