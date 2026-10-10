"""HM3D ObjectNav results of a run.py output directory: success and SPL at the official 0.1 m and the relaxed 0.12 m
distance to a goal view point, over all episodes of the episode file (a missing episode counts as a failure).

  python tools/hm3d_summary.py out/hm3d_val [--episodes data/episodes/team_episodes_hm3d_val.json]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('out')
ap.add_argument('--episodes', help='the episode file (default: the one in out/config.yaml)')
a = ap.parse_args()
if a.episodes:
    eps = json.loads(Path(a.episodes).read_text())['episodes']
else:
    import yaml
    cfg = yaml.safe_load((Path(a.out) / 'config.yaml').read_text())
    eps = json.loads((Path(__file__).resolve().parent.parent / cfg['episodes']).read_text())['episodes']
rows, by_cat = {}, defaultdict(list)
for d in Path(a.out).glob('*/'):
    if '.' not in d.name and (d / 'summary.json').exists():
        s = json.loads((d / 'summary.json').read_text())
        rows[s['episode_id']] = s
n = len(eps)
for tag, sk, pk in (('0.1 m (official)', 'objectnav_success', 'objectnav_spl'),
                    ('0.12 m (relaxed)', 'objectnav_success_012', 'objectnav_spl_012')):
    sr = sum(bool(rows.get(e['episode_id'], {}).get(sk)) for e in eps) / n
    spl = sum(float(rows.get(e['episode_id'], {}).get(pk) or 0) for e in eps) / n
    print(f'{tag:18s} SR {sr:.3f}  SPL {spl:.3f}   ({len(rows)}/{n} episodes run)')
for e in eps:
    by_cat[e['target']].append(bool(rows.get(e['episode_id'], {}).get('objectnav_success')))
print('by category (0.1 m):', {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())})
