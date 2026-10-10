"""Case memory: similar past outcomes as a boundary check before a declaration.

An earlier design put accepted/refused examples and acceptance rates for the target category into the judge's state
for every candidate decision. It lowered the judge's P on real targets (0.65 -> 0.58, P >= 0.5 from 90% to 71%) and
success fell by 0.16: the rates were a base rate, and a calibrated model with a base rate in its state shifts to it.

This memory keeps the judge's candidate state exactly as without memory and changes one thing: when the
gate is about to let the dog declare -- the irreversible step -- it asks Jev one more
question in a separate request. That request holds the dog's own descriptions of this
candidate and the most similar past declarations of the same category, as many accepted
as refused (so they carry no base rate). If the mission owner would accept the candidate
with P below the store's veto gate (REJECT_GATE unless chosen on held-out practice runs at
freeze time), the declaration becomes a team-wide reject and the team searches on.
The latency of that request is charged to the mission clock like every other call.

Feedback is one bit per declaration (the operator looking at the dog's camera in
deployment; the goal-object truth in simulation). Only declared candidates get a label,
so a veto would stop its own evidence from arriving: while learning, a veto is ignored
with probability EXPLORE (a deterministic hash of episode and track) and that declaration
is labelled like any other. That is the exploration half of the loop: memory changes what
the team declares, and what it declares is what memory learns next.

Protocol: learn only on training/development episodes; a frozen store is read-only and
refuses an episode from a scene it was built on.

  python src/team_experience_v3.py build --split hm3d_train --out memory/store.json        # an empty store
  python src/team_experience_v3.py build out/<run>/* --split hm3d_train --out store.json  # from finished runs
  python src/team_experience_v3.py check out/<held-out runs>/* --store store.json [--jev --allow-paid-inference]
  python src/team_experience_v3.py freeze store.json --veto-gate 0.3 --out frozen.json
Learning while running (practice episodes only): configs/practice_round.yaml (memory.learn: true).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

K = 3                     # past cases per side
EXPLORE = 0.25            # ponytail: fixed epsilon; make it depend on how many similar cases exist if labels stay scarce
LEARN_ROLES = ('hm3d_train', 'development', 'study', 'debug')

STOP = set('a an the and or of in on at to with by from is are its it this that there as be into near '
           'next behind front centre center image object visible seen appears partially slightly some '
           'other which while its their small large'.split())


def words(text):
    return Counter(w for w in re.findall(r'[a-z]+', (text or '').lower()) if w not in STOP and len(w) > 2)


def similarity(a, b):
    if not a or not b:
        return 0.0
    dot = sum(v * b[w] for w, v in a.items() if w in b)
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def evidence(answers, near):
    """The dog's own second opinions on one candidate (at most the last two) and what is near it."""
    own = [a for a in answers if str(a.get('by', '')).startswith('dog')][-2:]
    return {'descriptions': [a['description'][:200] for a in own if a.get('description')],
            'categories': [a['category'] for a in own], 'near': list(near)[:6]}


def text(ev):
    return ' '.join(ev['descriptions'])


# ------------------------------------------------------------------ memory
def harvest(run_dir):
    """One case per declaration: the dog's evidence on the declared candidate and the owner's bit."""
    run = Path(run_dir)
    if not (run / 'events.jsonl').exists() or not (run / 'manifest.json').exists():
        return None
    ev = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines() if line]
    dec = next((e for e in ev if e['event'] == 'declare'), None)
    if dec is None:
        return None
    episode = json.loads((run / 'manifest.json').read_text())['episode']
    tid = dec['track']
    answers = [{**e['answer'], 'by': e['robot']} for e in ev
               if e['event'] == 'verify_reply' and e['track'] == tid and e['t'] <= dec['t']]
    last = next((e for e in reversed(ev) if e['event'] == 'candidate_decision' and e['track'] == tid
                 and e['t'] <= dec['t']), None)
    near = last['context'].get('near_labels', []) if last else []
    return {'run': run.name, 'episode': episode['episode_id'], 'scene': episode['scene'],
            'target': episode['target'], 'accepted': bool(dec['track_was_real']),   # the one feedback bit
            'p': last['record'].get('p_target') if last else None, **evidence(answers, near)}


def new_store(split):
    return {'kind': 'team_experience', 'version': 3, 'frozen': False, 'split': split,
            'feedback': 'mission outcome, one bit per declaration', 'source_scenes': [], 'cases': []}


def add(store, case):
    if case and case['descriptions'] and not any(c['run'] == case['run'] for c in store['cases']):
        store['cases'].append(case)
        if case['scene'] not in store['source_scenes']:
            store['source_scenes'].append(case['scene'])


def save(store, path):
    tmp = Path(f'{path}.tmp{os.getpid()}')
    tmp.write_text(json.dumps(store, indent=1))
    os.replace(tmp, path)          # a reader never sees half a file


def similar(store, target, ev, k=K, exclude_episode=None):
    """The k most similar accepted and the k most similar refused cases, as many of each."""
    q = words(text(ev))
    pool = [c for c in store['cases'] if c['target'] == target and c['episode'] != exclude_episode]
    acc = sorted((c for c in pool if c['accepted']), key=lambda c: -similarity(q, words(text(c))))
    ref = sorted((c for c in pool if not c['accepted']), key=lambda c: -similarity(q, words(text(c))))
    n = min(k, len(acc), len(ref))
    return (acc[:n], ref[:n]) if n and q else ([], [])


# ------------------------------------------------------------------ the boundary question
def boundary_request(target, ev, acc, ref):
    t = target.replace('_', ' ')
    show = lambda c: {'description': ' '.join(c['descriptions']), 'objects_near_it': c['near']}
    state = {
        'mission': (f'A robot dog is about to declare a {t} found. The mission owner decides whether the '
                    f'object counts as a {t}; earlier missions showed where the owner draws that line.'),
        'candidate': {'dog_descriptions': ev['descriptions'], 'vision_model_categories': ev['categories'],
                      'objects_near_it': ev['near']},
        'past_cases': {'accepted_by_owner': [show(c) for c in acc], 'refused_by_owner': [show(c) for c in ref]},
        'note': ('The past cases are the most similar-looking ones, the same number from each side; how many '
                 'there are says nothing about how often the owner accepts.'),
    }
    questions = {'owner_accepts': {'type': 'noul', 'instructions': (
        f'Would the mission owner accept `candidate` as a {t}? Compare what kind of object it is with '
        f'`past_cases.accepted_by_owner` and `past_cases.refused_by_owner`: what the object actually is, '
        f'not wording, lighting or blur. If it is like none of them, judge from `candidate` alone.')}}
    return state, questions


def explored(episode_id, track_id):
    h = hashlib.sha256(f'{episode_id}:{track_id}'.encode()).digest()
    return int.from_bytes(h[:4], 'big') / 2 ** 32 < EXPLORE


def install(store_path, episode, learn=False):
    """Patch the runner for one episode: the boundary check, and in learn mode the write-back."""
    import team_agents as ag
    import team_run

    store = json.loads(Path(store_path).read_text())
    if store.get('version') != 3:
        raise SystemExit(f'{store_path} is not a v3 team experience store')
    if learn and store['frozen']:
        raise SystemExit(f'{store_path} is frozen; learning needs an unfrozen store')
    if learn and episode.get('role') not in LEARN_ROLES:
        raise SystemExit(f"learning on a {episode.get('role')} episode would train on the evaluation set")
    if episode['scene'] in store['source_scenes'] and not learn:
        raise SystemExit(f"{store_path} was built on scene {episode['scene']}: not a held-out test")

    base_candidate = ag.JevDecider.candidate

    def candidate(self, role, name, track, ctx, now):
        action, latency, rec = base_candidate(self, role, name, track, ctx, now)
        if role != 'dog' or action != 'declare_found':
            return action, latency, rec
        ev = evidence(getattr(track, 'vlm', []) or [], ctx.get('near_labels', []))
        acc, ref = similar(store, ctx['target'], ev)
        if not acc:
            rec['rsi'] = {'cases': 0}
            return action, latency, rec
        t0 = time.perf_counter()
        a = self.client.judge(*boundary_request(ctx['target'], ev, acc, ref))
        latency += time.perf_counter() - t0
        p = float(a['owner_accepts']['value'])
        rec['rsi'] = {'p_owner': round(p, 4), 'cases': len(acc), 'accepted': [c['run'] for c in acc],
                      'refused': [c['run'] for c in ref]}
        if p >= store.get('veto_gate', ag.REJECT_GATE):
            return action, latency, rec
        if learn and explored(episode['episode_id'], track.track_id):
            rec['rsi']['explore'] = True          # declare anyway: this is how a veto gets its label
            return action, latency, rec
        rec['rsi']['veto'] = True
        return 'reject', latency, rec
    ag.JevDecider.candidate = candidate

    if learn:
        base_run = team_run.Runner.run

        def run(self):
            summary = base_run(self)
            if not summary['counts'].get('service_errors'):          # the runner reruns those episodes
                fresh = json.loads(Path(store_path).read_text())      # single writer: runs are sequential
                add(fresh, harvest(self.out))
                save(fresh, store_path)
            return summary
        team_run.Runner.run = run


# ------------------------------------------------------------------ offline check
def check(run_dirs, jev=None, store=None):
    """Does "closer to refused than to accepted" separate the declarations of these runs? Memory is
    `store` (held-out check) or the runs themselves, leaving each episode out. Free without `jev`:
    it tells whether the memory has any structure to use before paying for it."""
    cases = [c for c in (harvest(d) for d in run_dirs) if c and c['descriptions']]
    store = store or {'cases': cases}
    scored = []
    for c in cases:
        acc, ref = similar(store, c['target'], c, exclude_episode=c['episode'])
        if acc:
            q = words(text(c))
            gap = similarity(q, words(text(ref[0]))) - similarity(q, words(text(acc[0])))
            p_owner = None
            if jev is not None:           # the live boundary question, replayed on this finished declaration
                p_owner = float(jev.judge(*boundary_request(c['target'], c, acc, ref))['owner_accepts']['value'])
            scored.append((gap, c['accepted'], c['p'], p_owner, c['run']))

    def auc(xs):         # P(score of a refused case > score of an accepted one)
        pos = [s for s, ok in xs if not ok]
        neg = [s for s, ok in xs if ok]
        if not pos or not neg:
            return None
        return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    with_p = [(ok, p) for _, ok, p, _, _ in scored if p is not None]
    out = {'declarations': len(cases), 'refused': sum(not c['accepted'] for c in cases),
           'with_both_sides': len(scored), 'auc_similarity_gap': auc([(g, ok) for g, ok, *_ in scored]),
           'auc_jev_p_same_cases': auc([(-p, ok) for ok, p in with_p]),
           'by_target': dict(Counter(f"{c['target']} {'accepted' if c['accepted'] else 'refused'}" for c in cases))}
    if jev is not None:
        import team_agents as ag
        out['auc_p_owner'] = auc([(-po, ok) for _, ok, _, po, _ in scored])
        out['vetoed_at_gate'] = {g: {'refused': sum(po < g and not ok for _, ok, _, po, _ in scored),
                                     'accepted': sum(po < g and ok for _, ok, _, po, _ in scored)}
                                 for g in (ag.REJECT_GATE, 0.4, 0.5, 0.6)}
        out['cases'] = [{'run': r, 'accepted': ok, 'p': p, 'p_owner': round(po, 3)} for _, ok, p, po, r in scored]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('runs', nargs='*', help='none: an empty store to learn into')
    b.add_argument('--out', required=True)
    b.add_argument('--split', required=True, help='where the runs come from, e.g. hm3d_train')
    f = sub.add_parser('freeze')
    f.add_argument('store')
    f.add_argument('--out', required=True)
    f.add_argument('--veto-gate', type=float, help='chosen on held-out practice runs; default REJECT_GATE')
    c = sub.add_parser('check')
    c.add_argument('runs', nargs='+')
    c.add_argument('--store', help='memory to check the runs against (held out); default: the runs, leave one out')
    c.add_argument('--jev', action='store_true', help='also replay the boundary question (paid, one call per case)')
    c.add_argument('--allow-paid-inference', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'build':
        store = new_store(a.split)
        for d in a.runs:
            add(store, harvest(d))
        save(store, a.out)
        print(len(store['cases']), 'cases from', len(store['source_scenes']), 'scenes')
    elif a.cmd == 'freeze':
        store = json.loads(Path(a.store).read_text())
        store['frozen'] = True
        if a.veto_gate is not None:
            store['veto_gate'] = a.veto_gate
        with Path(a.out).open('x') as fh:
            json.dump(store, fh, indent=1)
        os.chmod(a.out, 0o444)
        print(a.out, hashlib.sha256(Path(a.out).read_bytes()).hexdigest())
    else:
        if a.jev and not a.allow_paid_inference:
            raise SystemExit('--jev needs --allow-paid-inference')
        from jev import JevClient
        store = json.loads(Path(a.store).read_text()) if a.store else None
        print(json.dumps(check([d for d in a.runs if Path(d).is_dir()], JevClient() if a.jev else None, store),
                         indent=1))


if __name__ == '__main__':
    main()
