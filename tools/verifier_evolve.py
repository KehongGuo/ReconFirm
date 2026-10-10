"""Self-evolving verifier (paper Sec. III-D): practice-fold benchmark, corrections, DPO pairs, QLoRA DPO, and the
two upper rungs (the student correcting itself; evolving the instruction that writes corrections).

Crops come from practice runs with save_crops (configs/practice_round.yaml): each holds the verifier's input, the
exact prompt it was asked, its answer, and whether the candidate's track is a real target (annotation, evaluation
only). An answer is right when it names the target exactly when the track is real.

One round (r = 1, 2, ...), with the practice fold C collected once by the base model:
  python tools/verifier_evolve.py bench out/practice_C/* --model qwen3vl --out bench_r0.json     # round-0 score
  python tools/verifier_evolve.py jobs out/practice_R1/* --out jobs_r1.jsonl       # wrong crops + truth, for the teacher
  #   the teacher writes corr_r1.jsonl following data/teacher_instruction.md (labels always from the annotation)
  python tools/verifier_evolve.py pairs out/practice_R1/* --corrections corr_r1.jsonl --out pairs_r1.jsonl
  python tools/verifier_evolve.py train pairs_r1.jsonl --out adapters/r1        # QLoRA DPO, fresh from the base
  #   serve it beside the base (scripts/serve_vllm.sh with LORA=1) and load it:
  python tools/verifier_evolve.py load r1 adapters/r1
  python tools/verifier_evolve.py bench out/practice_C/* --model r1 --out bench_r1.json
  python tools/verifier_evolve.py pope data/pope/pope_adv_300.jsonl --model r1 --out pope_r1.json   # forgetting guard
From round 2 the student corrects its own wrong answers given the truth (selfcorrect), the rest go to the teacher,
and the teacher audits a sample (jobs --audit). evolve runs the bounded Promptbreeder-style meta level.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))      # category_standard, verifier_prompt

BASE = 'Qwen/Qwen3-VL-8B-Instruct'          # BF16, for training; vLLM serves the FP8 copy with the adapter on top
ENDPOINT = 'http://localhost:{port}/v1/chat/completions'


def crops(run_dirs):
    """Every saved verifier question in these runs, with an id that is stable across rounds."""
    for d in map(Path, run_dirs):
        f = d / 'crops' / 'crops.jsonl'
        if not f.exists() or '.' in d.name:          # .svcerr / .prefix_fail copies are not results
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get('prompt'):
                yield {**r, 'id': f"{d.name}/{r['k']}", 'image': str(d / r['file'])}


def canonical(category, target):
    """The answer under the benchmark's naming standard (e.g. a flowerpot counts as a plant on HM3D)."""
    import category_standard as cs
    return cs.canonical(category, target) if category else category


def right(category, rec):
    return (canonical(category, rec['target']) == rec['target']) == bool(rec['track_was_real'])


def ask(session, rec, model, port, extra=''):
    """The verifier's question on this crop, as asked in the run; `extra` follows the question (self-correction)."""
    from verifier_prompt import parse
    img = base64.b64encode(Path(rec['image']).read_bytes()).decode()
    payload = {'model': model, 'max_tokens': 120, 'temperature': 0.0, 'seed': 0,
               'messages': [{'role': 'user', 'content': [
                   {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + img}},
                   {'type': 'text', 'text': rec['prompt'] + extra}]}]}
    r = session.post(ENDPOINT.format(port=port), json=payload, timeout=180)
    r.raise_for_status()
    raw = r.json()['choices'][0]['message']['content']
    return raw, parse(raw, rec['vocabulary'])


def bench(a):
    import requests
    s = requests.Session()
    rows = []
    for rec in crops(a.runs):
        raw, ans = ask(s, rec, a.model, a.port)
        rows.append({'id': rec['id'], 'real': rec['track_was_real'], 'category': ans['category'],
                     'clear': ans['clear'], 'right': right(ans['category'], rec)})
    real = [r for r in rows if r['real']]
    false = [r for r in rows if not r['real']]
    out = {'model': a.model, 'n': len(rows), 'accuracy': sum(r['right'] for r in rows) / max(len(rows), 1),
           'real_named': sum(r['right'] for r in real) / max(len(real), 1), 'n_real': len(real),
           'false_not_named': sum(r['right'] for r in false) / max(len(false), 1), 'n_false': len(false),
           'rows': rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != 'rows'})


def wrong(a):
    with open(a.out, 'w') as fh:
        for rec in crops(a.runs):
            if not right(rec['answer']['category'], rec):
                fh.write(json.dumps({k: rec[k] for k in ('id', 'image', 'target', 'track_was_real', 'gt_top', 'answer')}) + '\n')


def answer_text(category, clear, description, reflection=False):
    return (f"CATEGORY: {category}\nCLEAR: {'yes' if clear else 'no'}\nDESCRIPTION: {description}\n"
            f"REFLECTION: {'yes' if reflection else 'no'}")


def pairs(a):
    """chosen = the correction, rejected = the student's own answer. Replay: a share of the answers the student got
    right, against the same answer with the target decision flipped, so training on corrections does not move them."""
    corr = {c['id']: c for c in map(json.loads, Path(a.corrections).read_text().splitlines()) if c.get('category')}
    rng = random.Random(0)
    out, replay = [], []
    for rec in crops(a.runs):
        raw = rec['answer'].get('raw') or answer_text(rec['answer']['category'], rec['answer']['clear'],
                                                       rec['answer']['description'])
        base = {'id': rec['id'], 'image': rec['image'], 'prompt': rec['prompt']}
        if rec['id'] in corr:
            c = corr[rec['id']]
            chosen = answer_text(c['category'], c['clear'], c['description'], c.get('reflection', False))
            if chosen.strip() != raw.strip():
                out.append({**base, 'chosen': chosen, 'rejected': raw, 'kind': 'correction'})
        elif right(rec['answer']['category'], rec) and rec['answer']['category']:
            flip = 'none' if rec['track_was_real'] else rec['target']
            rejected = re.sub(r'(CATEGORY\W*\s*)\S+', lambda m: m.group(1) + flip, raw, count=1)
            if rejected != raw:
                replay.append({**base, 'chosen': raw, 'rejected': rejected, 'kind': 'replay'})
    rng.shuffle(replay)
    out += replay[:int(a.replay * len(out) / (1 - a.replay))] if a.replay < 1 else replay
    with open(a.out, 'w') as fh:
        for p in out:
            fh.write(json.dumps(p) + '\n')
    print(len(out), 'pairs:', sum(p['kind'] == 'correction' for p in out), 'corrections ->', a.out)


def train(a):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    rows = [json.loads(l) for f in a.pairs for l in Path(f).read_text().splitlines()]

    def gen():
        for r in rows:
            yield {'images': [Image.open(r['image']).convert('RGB')],
                   'prompt': [{'role': 'user', 'content': r['prompt']}],
                   'chosen': [{'role': 'assistant', 'content': r['chosen']}],
                   'rejected': [{'role': 'assistant', 'content': r['rejected']}]}
    ds = Dataset.from_generator(gen)
    processor = AutoProcessor.from_pretrained(BASE, max_pixels=602112)       # as vLLM serves it
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map='auto',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM',
                      # the language model's projections only: the vision tower stays as served
                      target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)')
    cfg = DPOConfig(output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=8,
                    num_train_epochs=a.epochs, learning_rate=a.lr, beta=a.beta, max_length=2048,
                    gradient_checkpointing=True, bf16=True, logging_steps=5, save_strategy='no',
                    report_to=[], seed=0, max_steps=a.max_steps)
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=processor, peft_config=lora)
    trainer.train()
    trainer.save_model(a.out)
    meta = {'pairs': {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in a.pairs}, 'n': len(rows),
            'beta': a.beta, 'lr': a.lr, 'epochs': a.epochs, 'base': BASE}
    (Path(a.out) / 'round.json').write_text(json.dumps(meta, indent=1))
    print('adapter ->', a.out)

# ---------------------------------------------------------------- rung 2: the student writes its own corrections
STRUCTURE = {'floor', 'wall', 'ceiling', 'unknown', 'misc', 'void', 'object'}

# The initial correction instruction P_corr (rung 2); rung 3 evolves it. {truth} is filled from the annotation.
P_CORR = ('\n\nYour previous answer for this image was wrong. Ground truth: {truth}. Answer again in exactly the same '
          'format. Name the correct category from the list, say whether the view is clear, and in the description '
          'point out the visible feature that tells this object apart from what it could be mistaken for.')
# The initial mutation prompt M (rung 3), and the fixed prompt that mutates M itself (the recursive step).
M_INIT = ('Rewrite the instruction below, which asks a vision model to correct a wrong answer about an image, so '
          'that the corrected answers teach the model to tell objects apart from their look-alikes more reliably. '
          'Keep the placeholder {truth} exactly once and keep the required answer format. Return only the new '
          'instruction.')
H_META = ('Below is a prompt that rewrites instructions for correcting a vision model\'s wrong answers. Rewrite this '
          'prompt so that the instructions it produces lead to better corrections. Keep the placeholder {truth} '
          'mentioned as something the instruction must contain exactly once. Return only the new prompt.')


def truth_text(rec):
    """What the annotation says the centre object is, for the student (rung 2) and the teacher (rung 1)."""
    if rec['track_was_real']:
        return f"the centre object is a {rec['target']}"
    names = [n for n, _ in (rec.get('gt_top') or []) if n.lower() not in STRUCTURE]
    return (f"the centre object is not a {rec['target']}"
            + (f"; the scene annotation under it lists: {', '.join(names)}" if names else ''))


def wrong_records(runs):
    return [r for r in crops(runs) if not right(r['answer']['category'], r)]


def self_corrections(recs, model, port, instruction=P_CORR):
    """Rung 2: the student answers again given the truth; a correction is kept only if it is right (STaR)."""
    import requests
    s = requests.Session()
    kept, failed = [], []
    for rec in recs:
        raw, ans = ask(s, rec, model, port, extra=instruction.format(truth=truth_text(rec)))
        ok = ans['category'] and not ans['category'].startswith('?') and right(ans['category'], rec) and ans['description']
        row = {'id': rec['id'], 'category': ans['category'], 'clear': bool(ans['clear']),
               'description': ans['description'], 'reflection': bool(re.search(r'REFLECTION\W*\s*yes', raw, re.I)),
               'author': 'student', 'instruction': hashlib.sha256(instruction.encode()).hexdigest()[:12]}
        (kept if ok else failed).append(row)
    return kept, failed


def selfcorrect(a):
    instruction = Path(a.instruction).read_text() if a.instruction else P_CORR
    kept, failed = self_corrections(wrong_records(a.runs), a.model, a.port, instruction)
    with open(a.out, 'w') as fh:
        for r in kept:
            fh.write(json.dumps(r) + '\n')
    # the crops the student could not correct go to the teacher; their ids, for the teacher's file
    Path(a.out).with_suffix('.for_teacher.jsonl').write_text(''.join(json.dumps({'id': r['id']}) + '\n' for r in failed))
    print(f'{len(kept)} kept, {len(failed)} for the teacher -> {a.out}')


def jobs(a):
    """Teacher jobs (data/teacher_instruction.md): every wrong crop, or with --ids only those ids
    (the crops the student could not correct); with --audit, a sample of the student's corrections to audit."""
    recs = {r['id']: r for r in wrong_records(a.runs)}
    if a.audit:
        corr = [json.loads(l) for l in Path(a.audit).read_text().splitlines()]
        corr = random.Random(0).sample(corr, round(a.share * len(corr)))
        sel = [(recs[c['id']], {'student_correction': {k: c[k] for k in ('category', 'clear', 'description')}})
               for c in corr]
    else:
        ids = [json.loads(l)['id'] for l in Path(a.ids).read_text().splitlines()] if a.ids else list(recs)
        sel = [(recs[i], {}) for i in ids]
    with open(a.out, 'w') as fh:
        for r, extra in sel:
            fh.write(json.dumps({'id': r['id'], 'image': str(Path(r['image']).resolve()), 'prompt': r['prompt'],
                                 'vocabulary': r['vocabulary'], 'target': r['target'], 'truth': truth_text(r),
                                 'student_answer': r['answer'].get('raw'), **extra}) + '\n')
    print(len(sel), 'jobs ->', a.out)


def pope(a):
    """Forgetting guard: POPE (adversarial, COCO) yes/no object questions, as a general visual-question check that an
    adapter trained on the verifier's corrections leaves the model's object grounding intact."""
    import requests
    s, rows = requests.Session(), []
    root = Path(a.questions).parent
    for q in map(json.loads, Path(a.questions).read_text().splitlines()):
        img = base64.b64encode((root / 'img' / q['image']).read_bytes()).decode()
        r = s.post(ENDPOINT.format(port=a.port), timeout=180, json={
            'model': a.model, 'max_tokens': 5, 'temperature': 0.0, 'seed': 0, 'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + img}},
                {'type': 'text', 'text': q['text'] + ' Answer yes or no.'}]}]})
        r.raise_for_status()
        ans = 'yes' if r.json()['choices'][0]['message']['content'].strip().lower().startswith('yes') else 'no'
        rows.append((ans, q['label']))
    tp = sum(x == y == 'yes' for x, y in rows); fp = sum(x == 'yes' != y for x, y in rows)
    fn = sum(x == 'no' != y for x, y in rows)
    out = {'model': a.model, 'n': len(rows), 'accuracy': sum(x == y for x, y in rows) / len(rows),
           'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'yes_rate': sum(x == 'yes' for x, _ in rows) / len(rows)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(out)


# ---------------------------------------------------------------- rung 3: evolve the correction instruction
def chat_text(text, model, port, temperature=0.7, seed=0):
    import requests
    r = requests.post(ENDPOINT.format(port=port), timeout=180, json={
        'model': model, 'max_tokens': 300, 'temperature': temperature, 'seed': seed,
        'messages': [{'role': 'user', 'content': text}]})
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content'].strip()


def mutate(prompt, target_text, model, port, seed, need_truth=True):
    """One variant of `target_text` by `prompt`; None unless it keeps {truth} exactly once (an instruction)."""
    out = chat_text(f'{prompt}\n\n---\n{target_text}\n---', model, port, seed=seed).strip('-` \n')
    if need_truth:
        if out.count('{truth}') != 1 or '{' in out.replace('{truth}', '') or '}' in out.replace('{truth}', ''):
            return None
        out = '\n\n' + out.strip()
    return out


def load_adapter(port, name, path):
    import requests
    requests.post(f'http://localhost:{port}/v1/unload_lora_adapter', json={'lora_name': name}, timeout=60)
    r = requests.post(f'http://localhost:{port}/v1/load_lora_adapter', json={'lora_name': name, 'lora_path': str(path)},
                      timeout=300)
    r.raise_for_status()


def fitness(instruction, a, tag):
    """Student corrections under this instruction -> pairs (with the earlier rounds' pairs) -> a quick adapter ->
    accuracy on the practice fold's crops. The fitness is computed on fold C only."""
    import subprocess
    work = Path(a.work) / tag
    work.mkdir(parents=True, exist_ok=True)
    (work / 'instruction.txt').write_text(instruction)
    kept, _ = self_corrections(wrong_records(a.runs), a.model, a.port, instruction)
    corr = work / 'corr.jsonl'
    corr.write_text(''.join(json.dumps(r) + '\n' for r in kept))
    pairs(argparse.Namespace(runs=a.runs, corrections=str(corr), replay=0.2, out=str(work / 'pairs.jsonl')))
    subprocess.run(a.train_cmd.split() + [*a.prior_pairs, str(work / 'pairs.jsonl'), '--out', str(work / 'adapter'),
                                          '--epochs', '1'], check=True)
    load_adapter(a.port, 'evo', (work / 'adapter').resolve())
    bench(argparse.Namespace(runs=a.bench_runs, model='evo', port=a.port, out=str(work / 'bench.json')))
    acc = json.loads((work / 'bench.json').read_text())['accuracy']
    return {'tag': tag, 'accuracy': acc, 'n_corrections': len(kept), 'instruction': instruction}


def evolve(a):
    """Promptbreeder-style, bounded: n variants of P_corr by M; with --hyper also variants of M by the fixed H, each
    producing its own P_corr variants. Keeps the best instruction; the log says whether the evolved M helped."""
    p0 = Path(a.p_corr).read_text() if a.p_corr else P_CORR
    m0 = Path(a.m).read_text() if a.m else M_INIT
    ms = [('M0', m0)]
    if a.hyper:
        for j in range(a.n):
            m = mutate(H_META, m0, a.mutator, a.port, seed=100 + j, need_truth=False)
            if m:
                ms.append((f'M{j + 1}', m))
    log = {'p0': p0, 'ms': dict(ms), 'scores': []}
    log['scores'].append({'by': None, **fitness(p0, a, 'P0')})
    for mi, (mname, m) in enumerate(ms):
        for i in range(a.n):
            v = None
            for k in range(3):                       # a variant that drops {truth} is redrawn, at most twice
                v = mutate(m, p0.strip(), a.mutator, a.port, seed=1000 * mi + 10 * i + k)
                if v:
                    break
            if v:
                log['scores'].append({'by': mname, **fitness(v, a, f'{mname}_P{i + 1}')})
            Path(a.out).write_text(json.dumps(log, indent=1))
    best = max(log['scores'], key=lambda r: r['accuracy'])
    log['best'] = best['tag']
    for mname, _ in ms:
        sc = [r['accuracy'] for r in log['scores'] if r['by'] == mname]
        log.setdefault('best_by_m', {})[mname] = max(sc) if sc else None
    Path(a.out).write_text(json.dumps(log, indent=1))
    print({k: v for k, v in log.items() if k in ('best', 'best_by_m')}, 'P0', log['scores'][0]['accuracy'])



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('bench')
    b.add_argument('runs', nargs='+')
    b.add_argument('--model', default='qwen3vl')
    b.add_argument('--port', type=int, default=8010)
    b.add_argument('--out', required=True)
    w = sub.add_parser('wrong')
    w.add_argument('runs', nargs='+')
    w.add_argument('--out', required=True)
    p = sub.add_parser('pairs')
    p.add_argument('runs', nargs='+')
    p.add_argument('--corrections', required=True)
    p.add_argument('--replay', type=float, default=0.2, help='share of replay pairs in the output')
    p.add_argument('--out', required=True)
    t = sub.add_parser('train')
    t.add_argument('pairs', nargs='+')
    t.add_argument('--out', required=True)
    t.add_argument('--beta', type=float, default=0.1)
    t.add_argument('--lr', type=float, default=5e-5)
    t.add_argument('--epochs', type=float, default=2)
    t.add_argument('--max-steps', type=int, default=-1)
    c = sub.add_parser('selfcorrect', help='rung 2: the student corrects its wrong answers given the truth')
    c.add_argument('runs', nargs='+')
    c.add_argument('--model', required=True, help='the student as served (base name or LoRA name)')
    c.add_argument('--port', type=int, default=8010)
    c.add_argument('--instruction', help='a correction instruction file (rung 3 output); default P_CORR')
    c.add_argument('--out', required=True)
    j = sub.add_parser('jobs', help="the teacher's input: wrong crops with their truth, or an audit sample")
    j.add_argument('runs', nargs='+')
    j.add_argument('--ids', help='only these ids (selfcorrect .for_teacher.jsonl)')
    j.add_argument('--audit', help="a student corrections file to sample for the teacher's audit")
    j.add_argument('--share', type=float, default=0.2)
    j.add_argument('--out', required=True)
    q = sub.add_parser('pope', help='forgetting guard: POPE yes/no questions (questions file beside img/)')
    q.add_argument('questions')
    q.add_argument('--model', default='qwen3vl')
    q.add_argument('--port', type=int, default=8010)
    q.add_argument('--out', required=True)
    ld = sub.add_parser('load', help='load (or reload) a LoRA adapter into the running vLLM server')
    ld.add_argument('name')
    ld.add_argument('path')
    ld.add_argument('--port', type=int, default=8010)
    e = sub.add_parser('evolve', help='rung 3: evolve the correction instruction (and with --hyper, its mutator)')
    e.add_argument('runs', nargs='+', help="this round's learning-fold runs")
    e.add_argument('--bench-runs', nargs='+', required=True, help='practice fold C runs (fitness)')
    e.add_argument('--model', required=True, help='the student that writes corrections')
    e.add_argument('--mutator', default='qwen3vl', help='the model that applies M and H')
    e.add_argument('--port', type=int, default=8010)
    e.add_argument('--prior-pairs', nargs='*', default=[])
    e.add_argument('--train-cmd', required=True, help='e.g. "/path/lora/python verifier_evolve.py train"')
    e.add_argument('--p-corr'); e.add_argument('--m')
    e.add_argument('--n', type=int, default=2)
    e.add_argument('--hyper', action='store_true')
    e.add_argument('--work', required=True)
    e.add_argument('--out', required=True)
    a = ap.parse_args()
    {'bench': bench, 'wrong': wrong, 'pairs': pairs, 'train': train, 'selfcorrect': selfcorrect,
     'jobs': jobs, 'pope': pope, 'evolve': evolve,
     'load': lambda a: load_adapter(a.port, a.name, Path(a.path).resolve())}[a.cmd](a)


if __name__ == '__main__':
    main()
