"""HM3D ObjectNav v2 val (or sampled train) episodes in the team runner's episode format.

Start pose, target category and the shortest geodesic distance are the
benchmark's own; the goal objects and view points are read at run time from
the benchmark file (`objectnav` reference), so scoring uses the official goals.

  python tools/make_hm3d_episodes.py --out data/episodes/team_episodes_hm3d_val.json
  python tools/make_hm3d_episodes.py --split train --per-scene 1 --out data/episodes/team_episodes_hm3d_train.json

Train episodes are practice for team memory (team_experience_v3): a few per scene, the target
categories rotating over scenes. Each scene falls in one fold by a hash of its name: R1/R2/R3 (~70%,
one per self-evolution round; memory learns there), B (chooses the veto gate and DPO settings) or C (the
practice fold that scores every round). --fold R keeps R1-R3.
"""
import argparse
import glob
import gzip
import hashlib
import json
import math
import os

ROOT = os.path.join(os.environ.get('SCOUT_OBJECTNAV_ROOT', ''), '{split}', 'content')


def convert(path, role='hm3d_val'):
    doc = json.load(gzip.open(path))
    out = []
    for i, e in enumerate(doc['episodes']):   # episode_id repeats within a scene file; use the index
        scene = os.path.basename(e['scene_id']).split('.')[0]
        qx, qy, qz, qw = e['start_rotation']
        out.append({'scene': scene, 'seed': i, 'target': e['object_category'],
                    'floor_y': float(e['start_position'][1]), 'start': [float(v) for v in e['start_position']],
                    'start_yaw': 2 * math.atan2(qy, qw),       # rotation about +y, as Body.place uses it
                    'shortest_geodesic_m': float(e['info']['geodesic_distance']),
                    'episode_id': f"{scene}:{i}", 'objectnav_episode_id': e['episode_id'], 'role': role,
                    'objectnav': {'file': os.path.relpath(path, os.environ.get('SCOUT_OBJECTNAV_ROOT', '')), 'key': f"{os.path.basename(e['scene_id'])}_{e['object_category']}"}})
    return out


def scene_fold(name):
    h = hashlib.sha256(name.encode()).digest()
    if h[0] < 179:                                  # ~70%: the learning folds
        return f'R{h[1] % 3 + 1}'
    return 'B' if h[1] < 128 else 'C'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', choices=['val', 'train'], default='val')
    ap.add_argument('--per-scene', type=int, default=1, help='train only')
    ap.add_argument('--fold', choices=['R', 'R1', 'R2', 'R3', 'B', 'C'], help='train only: keep one fold')
    a = ap.parse_args()
    files = sorted(glob.glob(f'{ROOT.format(split=a.split)}/*.json.gz'))
    if a.split == 'val':
        eps = [e for f in files for e in convert(f)]
        assert len(eps) == 1000 and len({e['episode_id'] for e in eps}) == 1000, len(eps)
    else:
        cats = ['chair', 'sofa', 'toilet', 'bed', 'plant', 'tv_monitor']
        eps = []
        for k, f in enumerate(files):
            fold = scene_fold(os.path.basename(f))
            by_cat = {}
            for e in convert(f, 'hm3d_train'):
                by_cat.setdefault(e['target'], e)          # the first episode of each category
            for j in range(a.per_scene):
                e = next((by_cat[c] for c in cats[(k + j) % 6:] + cats[:(k + j) % 6] if c in by_cat), None)
                if e and a.fold in (None, fold, fold[0] if fold[0] == 'R' else None):
                    eps.append({**e, 'fold': fold})
                    by_cat.pop(e['target'])
    json.dump({'source': f'objectnav_hm3d_v2 {a.split}', 'episodes': eps}, open(a.out, 'w'))
    print(len(eps), 'episodes ->', a.out)


if __name__ == '__main__':
    main()
