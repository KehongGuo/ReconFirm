"""Table I (MP3D, 50 missions in 10 buildings) and its paired contrasts, from a run.py output directory.

Per mission: success (the legged robot declared within 1 m of a target), penalized time min(tau*, T) (T on failure),
false trips (legged-robot trips to false candidates), discarded (real-target candidates rejected). Precision is pooled:
correct declarations / declarations. With several passes (p1/, p2/, ...) every per-mission metric is first averaged
over the passes. Contrasts: ours minus each other row, differences averaged per building over the missions both
conditions finished, a 95% t interval over buildings and an exact sign-flip test (all 2^n sign patterns).

  python tools/table.py out/mp3d_table          # prints the table and the contrasts, writes table.json
"""
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROWS = [('Frontier+Rules', 'frontier_rules'), ('VLM-Direct', 'vlm_direct'), ('Ours', 'ours'),
        ('w/o gates (rules)', 'wo_gates'), ('w/o semantic scheduler', 'wo_semantic_scheduler'),
        ('gates on legged only', 'gates_legged_only'), ('gates on drones only', 'gates_drones_only'),
        ('latency-free clock', 'latency_free')]
OURS = 'ours'
METRICS = ('sr', 'time', 'false_trips', 'disc')


def mission(d):
    s = json.loads((d / 'summary.json').read_text())
    T = json.loads((d / 'manifest.json').read_text())['mission_s']
    real = {}
    disc = 0
    for line in open(d / 'events.jsonl'):
        e = json.loads(line)
        if e['event'] == 'candidate_new':
            real[e['track']] = bool(e.get('real'))
        elif e['event'] == 'candidate_rejected' and real.get(e['track']):
            disc += 1
    declared = s['outcome'] in ('success', 'false_declaration')
    return {'episode': s['episode_id'], 'building': s['episode_id'].split(':')[0],
            'sr': float(bool(s['success'])), 'time': float(s['t']) if s['success'] else T,
            'false_trips': float(s['dog_trips_to_false']), 'disc': float(disc),
            'declared': declared, 'correct': declared and bool(s['success'])}


def load(out):
    """cond -> episode -> metrics, averaged over the passes (out/p1, out/p2, ...) when there are several."""
    passes = sorted(Path(out).glob('p[0-9]*/')) or [Path(out)]
    per = defaultdict(lambda: defaultdict(list))
    for p in passes:
        for d in sorted(p.glob('*__*/')):
            if '.' in d.name or not (d / 'summary.json').exists():
                continue                 # moved-aside runs (.svcerr*) and unfinished ones
            m = mission(d)
            per[d.name.split('__', 1)[1]][m['episode']].append(m)
    runs = defaultdict(dict)
    for cond, eps in per.items():
        for e, ms in eps.items():
            runs[cond][e] = {**{k: float(np.mean([m[k] for m in ms])) for k in METRICS}, 'building': ms[0]['building'],
                             'declared': sum(m['declared'] for m in ms), 'correct': sum(m['correct'] for m in ms)}
    return runs


def contrast(a, b, key):
    eps = sorted(set(a) & set(b))
    by = defaultdict(list)
    for e in eps:
        by[a[e]['building']].append(a[e][key] - b[e][key])
    d = np.array([np.mean(v) for v in by.values()])
    n = len(d)
    if n < 2:
        return None
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    obs = abs(d.mean())
    flips = [abs(np.dot(sg, d)) / n for sg in itertools.product((1, -1), repeat=n)]
    p = float(np.mean([f >= obs - 1e-12 for f in flips]))
    return {'diff': float(d.mean()), 'ci': [float(d.mean() - half), float(d.mean() + half)], 'p': p,
            'buildings': n, 'missions': len(eps)}


def main(out):
    runs = load(out)
    table, rows = {}, []
    for name, cond in ROWS:
        r = runs.get(cond, {})
        if not r:
            rows.append(f'{name:24s} (no runs)'); continue
        v = list(r.values()); declared = sum(m['declared'] for m in v)
        row = {k: float(np.mean([m[k] for m in v])) for k in METRICS}
        row['prec'] = sum(m['correct'] for m in v) / declared if declared else float('nan')
        row['n'] = len(v)
        if cond != OURS and OURS in runs:
            row['vs_ours'] = {k: contrast(runs[OURS], r, k) for k in METRICS}
        table[cond] = row
        rows.append(f"{name:24s} {row['sr']:.2f}  {row['time']:6.1f}  {row['prec']:.2f}  {row['false_trips']:.2f}  "
                    f"{row['disc']:.2f}   n={row['n']}")
    print(f"{'Method':24s} SR    Time    Prec  FalseT Disc"); print('\n'.join(rows))
    print('\nOurs minus row: diff [95% t CI over buildings] p (exact sign-flip)')
    for name, cond in ROWS:
        for k, c in table.get(cond, {}).get('vs_ours', {}).items():
            if c:
                print(f"  {name:24s} {k:11s} {c['diff']:+7.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] p={c['p']:.3f}"
                      f"  ({c['buildings']} bldg, {c['missions']} missions)")
    (Path(out) / 'table.json').write_text(json.dumps(table, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'out/mp3d_table')
