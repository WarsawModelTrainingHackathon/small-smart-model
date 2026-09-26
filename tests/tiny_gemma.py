"""Build a tiny, randomly initialised Gemma 4 model + processor fully offline.

Used only by the CPU tests: huggingface.co is not reachable in CI/sandboxes and the
real 12B checkpoint must never be downloaded there. The architecture classes are the
real ones from transformers (Gemma4Unified / Gemma4 ForConditionalGeneration); only
the sizes, the tokenizer (a small byte-level BPE trained on the fly) and the chat
template (an approximation of the Gemma 4 template with <|turn> ... <turn|>) differ.

    python tests/tiny_gemma.py /tmp/tiny-gemma4 [--kind unified|gemma4]
"""
import argparse
import json
from pathlib import Path

SPECIAL = ['<pad>', '<eos>', '<bos>', '<unk>', '<|turn>', '<turn|>', '<|image>', '<image|>', '<|image|>',
           '<|audio>', '<audio|>', '<|audio|>', '<|video|>', '<|think|>', '<|channel>', '<channel|>']

# Mirrors the structure of the Gemma 4 template: system turn (with <|think|> when thinking
# is enabled), user/model turns closed by <turn|>, images as <|image|> placeholders.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{%- set sys = messages[0] if messages and messages[0]['role'] == 'system' else none -%}"
    "{%- if sys is not none or enable_thinking -%}"
    "<|turn>system{{ '\\n' }}{% if enable_thinking %}<|think|>{% endif %}"
    "{%- if sys is not none -%}{%- if sys['content'] is string -%}{{ sys['content'] }}"
    "{%- else -%}{%- for c in sys['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}{%- endif -%}{%- endfor -%}{%- endif -%}{%- endif -%}"
    "<turn|>{{ '\\n' }}"
    "{%- endif -%}"
    "{%- for m in messages -%}{%- if m['role'] != 'system' -%}"
    "<|turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}{{ '\\n' }}"
    "{%- if m['content'] is string -%}{{ m['content'] }}{%- else -%}"
    "{%- for c in m['content'] -%}{%- if c['type'] == 'image' -%}<|image|>{%- elif c['type'] == 'text' -%}{{ c['text'] }}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}<turn|>{{ '\\n' }}"
    "{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|turn>model{{ '\\n' }}{%- endif -%}"
)


def _corpus():
    base = ['Odpowiedź: A B C D; 1-P; 2-F; 3-P', 'Jesteś uczniem zdającym maturę z historii.',
            'Zadanie Źródło tekst mapa ilustracja Polska Rzeczpospolita król wojna pokój']
    here = Path(__file__).resolve().parent / 'fixtures' / 'items.json'
    if here.exists():
        for r in json.loads(here.read_text(encoding='utf-8')):
            base += [r.get('context') or '', r.get('question') or '', r.get('answer') or '']
    return base * 3


def build_tokenizer(out):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=1024, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(_corpus(), trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token='<bos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>',
        extra_special_tokens={'image_token': '<|image|>', 'boi_token': '<|image>', 'eoi_token': '<image|>',
                              'audio_token': '<|audio|>', 'boa_token': '<|audio>', 'eoa_token': '<audio|>',
                              'sot_token': '<|turn>', 'eot_token': '<turn|>', 'think_token': '<|think|>',
                              'soc_token': '<|channel>', 'eoc_token': '<channel|>'},
        padding_side='left')
    fast.chat_template = CHAT_TEMPLATE
    return fast


def build(out, kind='unified', seed=0):
    import torch
    import transformers as tf
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(out)
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL}
    torch.manual_seed(seed)
    text = dict(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
                max_position_embeddings=8192, sliding_window=64, layer_types=['sliding_attention', 'full_attention'],
                pad_token_id=ids['<pad>'], eos_token_id=[ids['<eos>'], ids['<turn|>']], bos_token_id=ids['<bos>'])
    mm = dict(boi_token_id=ids['<|image>'], eoi_token_id=ids['<image|>'], image_token_id=ids['<|image|>'],
              video_token_id=len(tokenizer), audio_token_id=ids['<|audio|>'], boa_token_id=ids['<|audio>'])
    if kind == 'unified':
        cfg = tf.Gemma4UnifiedConfig(
            text_config=tf.Gemma4UnifiedTextConfig(**text),
            vision_config=tf.Gemma4UnifiedVisionConfig(mm_embed_dim=64, mm_posemb_size=64, output_proj_dims=64),
            eoa_token_index=ids['<audio|>'], **mm)
        model = tf.Gemma4UnifiedForConditionalGeneration(cfg)
        image_processor = tf.Gemma4UnifiedImageProcessor(max_soft_tokens=70)
        processor_cls = tf.Gemma4UnifiedProcessor
        feature_extractor = tf.Gemma4UnifiedAudioFeatureExtractor()
        video_processor = tf.Gemma4UnifiedVideoProcessor()
    else:
        raise ValueError(f'unknown kind {kind}')
    # The processor registers <|video|> as an extra token; keep the embedding big enough.
    processor = processor_cls(feature_extractor=feature_extractor, image_processor=image_processor,
                              tokenizer=tokenizer, video_processor=video_processor, chat_template=CHAT_TEMPLATE)
    if len(tokenizer) > model.config.text_config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.config.video_token_id = processor.video_token_id
    model.generation_config.eos_token_id = [ids['<eos>'], ids['<turn|>']]
    model.generation_config.pad_token_id = ids['<pad>']
    model.save_pretrained(out)
    processor.save_pretrained(out)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out')
    p.add_argument('--kind', default='unified', choices=['unified'])
    a = p.parse_args()
    print(build(a.out, a.kind))
