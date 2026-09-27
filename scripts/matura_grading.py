"""Grading for matura z historii answers.

* Closed items (closed_abcd / true_false / matching) with a key: graded automatically from
  the LAST 'Odpowiedź:' line (scripts/matura_format.py), with partial credit where the CKE
  scoring text gives it ("2 pkt – trzy prawidłowe wskazania, 1 pkt – dwa ...").
* Closed items that require a justification: 0 if no answer-key component is correct; otherwise
  the judge scores the response and its justification against the CKE rubric.
* Open items and essays: LLM judge with the CKE rubric (scoring / scoring_raw) and the
  example answer, returning JSON {"points": int, "reason": str}.

Judge backends: none | hf (a local transformers model) | openai (any OpenAI-compatible
chat endpoint; env MATURA_JUDGE_BASE_URL, MATURA_JUDGE_API_KEY, MATURA_JUDGE_MODEL, falling
back to OPENAI_BASE_URL / OPENAI_API_KEY).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections import defaultdict

import matura_format as mf

# --------------------------------------------------------------------------- partial credit

_NUM = {
    'jeden': 1, 'jedna': 1, 'jednego': 1, 'jednej': 1, 'jednym': 1, 'jedno': 1,
    'dwa': 2, 'dwie': 2, 'dwoch': 2, 'dwu': 2, 'dwoje': 2, 'dwojga': 2, 'dwoma': 2, 'dwiema': 2,
    'trzy': 3, 'trzech': 3, 'trzem': 3, 'trzema': 3, 'troje': 3,
    'cztery': 4, 'czterech': 4, 'czterem': 4, 'czterema': 4, 'czterej': 4,
    'piec': 5, 'pieciu': 5, 'piecioro': 5,
    'szesc': 6, 'szesciu': 6, 'szescioro': 6,
    'siedem': 7, 'siedmiu': 7, 'osiem': 8, 'osmiu': 8,
    'dziewiec': 9, 'dziewieciu': 9, 'dziesiec': 10, 'dziesieciu': 10,
}
_POINTS_LINE = re.compile(r'^\s*(\d+)\s*(?:p\.|pkt\.?|punkt\w*)\s*[–—-]\s*(.*)$', re.I)


def _counts_in(text, n_parts):
    """Numbers of correct elements mentioned on one scoring line."""
    t = text.lower()
    if re.search(r'\bwszystk\w*', t):
        return {n_parts}
    found = set()
    for word in re.findall(r'\b\w+', t):
        if word.isdigit() and 0 < int(word) <= 10:
            found.add(int(word))
            continue
        value = _NUM.get(mf.normalize(word))
        if value is not None:
            found.add(value)
    return found


def credit_table(item):
    """[(min_correct, points)] from the CKE scoring text; None if not derivable."""
    n = len(item['answer_key']) if isinstance(item.get('answer_key'), dict) else 1
    rows = []
    for line in (item.get('scoring') or '').split('\n'):
        m = _POINTS_LINE.match(line)
        if not m or int(m.group(1)) == 0:
            continue
        counts = {c for c in _counts_in(m.group(2), n) if c <= n}
        if counts:
            rows.append((min(counts), int(m.group(1))))
    if not rows or max(p for _, p in rows) != item.get('max_points', 1):
        return None
    return sorted(rows, key=lambda r: -r[1])


def closed_points(item, n_correct, n_parts):
    max_points = item.get('max_points', 1)
    if n_correct == n_parts:
        return max_points
    table = credit_table(item) if n_parts > 1 else None
    if table:
        for min_correct, pts in table:
            if n_correct >= min_correct:
                return pts
    return 0


def grade_closed(item, text):
    pred = mf.parse_prediction(item, text)
    n_correct, n_parts = mf.closed_correct(item, pred)
    return dict(points=closed_points(item, n_correct, n_parts), max_points=item.get('max_points', 1),
                parsed=pred, parse_ok=pred is not None, n_correct=n_correct, n_parts=n_parts, method='auto')


# --------------------------------------------------------------------------- judge

JUDGE_SYSTEM = ('Jesteś doświadczonym egzaminatorem CKE oceniającym maturę z historii (poziom rozszerzony). '
                'Oceniasz ściśle według zasad oceniania. Przykładowa odpowiedź pokazuje oczekiwany zakres, '
                'ale akceptujesz każde merytorycznie poprawne sformułowanie spełniające kryteria. '
                'Błędy merytoryczne, brak wymaganych elementów lub odpowiedź nie na temat obniżają ocenę.')


def judge_prompt(item, answer_text):
    rubric = (item.get('scoring_raw') or item.get('scoring') or '').strip()
    notes = (item.get('scoring_notes') or '').strip()
    example = (item.get('answer') or '').strip()
    parts = [f"Zadanie {item.get('task')} (maksymalnie {item.get('max_points', 1)} pkt, typ: {item.get('type')}).",
             'Polecenie:\n' + (item.get('question') or '').strip()]
    if item.get('type') != mf.ESSAY and (item.get('context') or '').strip():
        parts.append('Materiały źródłowe (skrót):\n' + item['context'].strip()[:4000])
    if mf.is_closed(item):
        parts.append('Klucz (część zamknięta): ' + json.dumps(item['answer_key'], ensure_ascii=False))
    parts.append('Zasady oceniania CKE:\n' + rubric)
    if notes:
        parts.append('Uwagi do oceniania:\n' + notes)
    if example:
        parts.append('Przykładowa odpowiedź z klucza:\n' + example)
    parts.append('Odpowiedź zdającego:\n<<<\n' + (answer_text or '').strip()[:12000] + '\n>>>')
    parts.append(f'Przyznaj liczbę punktów od 0 do {item.get("max_points", 1)} zgodnie z zasadami oceniania. '
                 'Zwróć WYŁĄCZNIE obiekt JSON: {"points": <liczba całkowita>, "reason": "<krótkie uzasadnienie po polsku>"}')
    return '\n\n'.join(parts)


def parse_judge(text, max_points):
    text = mf.strip_thinking(text or '')
    for m in reversed(list(re.finditer(r'\{[^{}]*\}', text, re.S))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if 'points' in obj:
            try:
                pts = int(round(float(obj['points'])))
            except (TypeError, ValueError):
                continue
            return dict(points=max(0, min(max_points, pts)), reason=str(obj.get('reason', ''))[:1000])
    m = re.search(r'"?points"?\s*[:=]\s*(\d+)', text)
    if m:
        return dict(points=max(0, min(max_points, int(m.group(1)))), reason='(nieparsowalny JSON)')
    return None


class NoJudge:
    name = 'none'

    def __call__(self, prompts):
        return [None] * len(prompts)


class OpenAIJudge:
    """OpenAI-compatible /chat/completions endpoint (vLLM, OpenRouter, OpenAI, ...)."""
    name = 'openai'

    def __init__(self, model=None, base_url=None, api_key=None, max_tokens=512):
        self.base_url = (base_url or os.environ.get('MATURA_JUDGE_BASE_URL') or os.environ.get('OPENAI_BASE_URL')
                         or 'https://api.openai.com/v1').rstrip('/')
        self.api_key = api_key or os.environ.get('MATURA_JUDGE_API_KEY') or os.environ.get('OPENAI_API_KEY', '')
        self.model = model or os.environ.get('MATURA_JUDGE_MODEL', 'gpt-4.1-mini')
        self.max_tokens = max_tokens
        self.name = f'openai:{self.model}'

    def _one(self, prompt):
        body = json.dumps(dict(model=self.model, temperature=0, max_tokens=self.max_tokens,
                               messages=[{'role': 'system', 'content': JUDGE_SYSTEM},
                                         {'role': 'user', 'content': prompt}])).encode()
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.base_url + '/chat/completions', data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.loads(r.read())['choices'][0]['message']['content']
            except Exception as e:  # noqa: BLE001 - retry any transport error
                if attempt == 4:
                    return f'JUDGE_ERROR: {e}'
                time.sleep(2 ** attempt)

    def __call__(self, prompts):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as ex:
            return list(ex.map(self._one, prompts))


class HFJudge:
    """Local transformers chat model as judge (e.g. the base model itself, or a bigger one)."""

    def __init__(self, model_id, load_4bit=False, max_new_tokens=384, batch_size=4):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM
        self.torch = torch
        kw = dict(dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map='auto')
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        tok = getattr(self.processor, 'tokenizer', self.processor)
        tok.padding_side = 'left'
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
        except (ValueError, KeyError):
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.name = f'hf:{model_id}'

    def __call__(self, prompts):
        out = []
        tok = getattr(self.processor, 'tokenizer', self.processor)
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i:i + self.batch_size]
            texts = [self.processor.apply_chat_template(
                [{'role': 'system', 'content': [{'type': 'text', 'text': JUDGE_SYSTEM}]},
                 {'role': 'user', 'content': [{'type': 'text', 'text': p}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in chunk]
            enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
            out += tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        return out


def make_judge(backend='none', model=None, load_4bit=False):
    if backend in (None, 'none'):
        return NoJudge()
    if backend == 'openai':
        return OpenAIJudge(model=model)
    if backend == 'hf':
        if not model:
            raise ValueError('--judge hf needs --judge-model')
        return HFJudge(model, load_4bit=load_4bit)
    raise ValueError(f'unknown judge backend {backend}')


# --------------------------------------------------------------------------- grade

def needs_judge(item):
    return not mf.auto_gradable(item)


def grade_records(items_by_id, generations, judge):
    """generations: [{id, output}] -> graded records (one per generation)."""
    graded, to_judge = [], []
    for g in generations:
        item = items_by_id[g['id']]
        text = mf.strip_thinking(g.get('output', ''))
        rec = dict(id=item['id'], type=item['type'], session_key=item['session_key'], max_points=item.get('max_points', 1),
                   needs_visual=bool(item.get('needs_visual')), essay_group=mf.essay_group(item) if item['type'] == mf.ESSAY else None)
        if mf.is_closed(item):
            rec.update(grade_closed(item, text))
            rec['parsed'] = rec['parsed'] if not isinstance(rec['parsed'], set) else list(rec['parsed'])
            if not mf.auto_gradable(item):
                if rec['n_correct'] == 0:
                    rec.update(points=0, method='auto-wrong-choice')
                else:
                    rec.update(points=None, method='judge')
                    to_judge.append((rec, item, text))
        else:
            rec.update(points=None, parse_ok=None, method='judge')
            to_judge.append((rec, item, text))
        graded.append(rec)
    if to_judge:
        raw = judge([judge_prompt(item, text) for _, item, text in to_judge])
        for (rec, item, _), r in zip(to_judge, raw):
            if r is None:
                rec.update(points=None, method='ungraded', judge=judge.name)
                continue
            parsed = parse_judge(r, rec['max_points'])
            rec.update(judge=judge.name, judge_raw=r[:2000])
            if parsed is None:
                rec.update(points=None, method='judge-failed')
            else:
                rec.update(points=parsed['points'], reason=parsed['reason'])
    return graded


# --------------------------------------------------------------------------- report

def _agg():
    return dict(points=0.0, max=0.0, n=0, graded=0, ungraded=0)


def _add(bucket, points, max_points):
    bucket['n'] += 1
    if points is None:
        bucket['ungraded'] += 1
    else:
        bucket['graded'] += 1
        bucket['points'] += points
        bucket['max'] += max_points


def _finish(bucket):
    bucket['pct'] = round(100 * bucket['points'] / bucket['max'], 2) if bucket['max'] else None
    bucket['points'] = round(bucket['points'], 2)
    return bucket


def summarize(graded, items_by_id, essay_policy='mean'):
    """Totals with ONE essay per session (alternative topics are merged by `essay_policy`).

    Percentages are over graded items only (ungraded = judge 'none' or failed); the
    `exam_max` field is the full paper maximum of the evaluated items (one essay per group).
    """
    units = []  # (points or None, max, type, session, visual)
    groups = defaultdict(list)
    for r in graded:
        if r['type'] == mf.ESSAY:
            groups[r['essay_group']].append(r)
        else:
            units.append((r['points'], r['max_points'], r['type'], r['session_key'], r['needs_visual']))
    for key, rs in groups.items():
        pts = [r['points'] for r in rs if r['points'] is not None]
        if not pts:
            p = None
        elif essay_policy == 'best':
            p = max(pts)
        elif essay_policy == 'first':
            # Preserve the input order used by eval_matura.py --essays one.
            p = rs[0]['points']
        else:
            p = sum(pts) / len(pts)
        units.append((p, max(r['max_points'] for r in rs), mf.ESSAY, rs[0]['session_key'], any(r['needs_visual'] for r in rs)))
    total, by_type, by_session, by_visual = _agg(), defaultdict(_agg), defaultdict(_agg), defaultdict(_agg)
    auto = _agg()
    exam_max = 0
    for p, mx, t, s, v in units:
        exam_max += mx
        for b in (total, by_type[t], by_session[s], by_visual['visual' if v else 'text']):
            _add(b, p, mx)
        if t in mf.CLOSED_TYPES:
            _add(auto, p, mx)
    closed = [r for r in graded if r['type'] in mf.CLOSED_TYPES]
    unparseable = sum(1 for r in closed if not r.get('parse_ok'))
    by_session_max = defaultdict(float)
    for p, mx, t, s, v in units:
        by_session_max[s] += mx
    sessions = {}
    for s, b in sorted(by_session.items()):
        sessions[s] = dict(_finish(b), exam_max=by_session_max[s])
    return dict(
        total=_finish(total), exam_max=exam_max, essay_policy=essay_policy,
        closed_items=_finish(auto),
        by_type={k: _finish(v) for k, v in sorted(by_type.items())},
        by_session=sessions,
        by_visual={k: _finish(v) for k, v in sorted(by_visual.items())},
        closed_unparseable=dict(n=unparseable, of=len(closed), rate=round(unparseable / len(closed), 4) if closed else None),
        n_generations=len(graded), n_essay_groups=len(groups),
    )


def report_markdown(summary, meta=None):
    meta = meta or {}
    t = summary['total']
    lines = [f"# Matura historia eval – {meta.get('label', '')}", '']
    for k in ('model', 'adapter', 'load_4bit', 'split', 'text_only', 'thinking', 'judge', 'n_items'):
        if k in meta:
            lines.append(f'- **{k}**: {meta[k]}')
    lines += ['', f"**Total (graded): {t['points']}/{t['max']} = {t['pct']}%**  (paper max of evaluated items: "
              f"{summary['exam_max']}, ungraded units: {t['ungraded']}; one essay per session, policy={summary['essay_policy']})", '',
              f"Closed items (auto): {summary['closed_items']['points']}/{summary['closed_items']['max']} = {summary['closed_items']['pct']}%  ",
              f"Unparseable closed answers: {summary['closed_unparseable']['n']}/{summary['closed_unparseable']['of']}", '']

    def table(title, rows, extra=False):
        out = [f'## {title}', '', '| | points | max | % | n | ungraded |' + (' paper max |' if extra else ''),
               '|---|---|---|---|---|---|' + ('---|' if extra else '')]
        for k, b in rows.items():
            out.append(f"| {k} | {b['points']} | {b['max']} | {b['pct']} | {b['n']} | {b['ungraded']} |"
                       + (f" {b['exam_max']} |" if extra else ''))
        return out + ['']
    lines += table('By type', summary['by_type'])
    lines += table('By session', summary['by_session'], extra=True)
    lines += table('Visual vs text', summary['by_visual'])
    return '\n'.join(lines)
