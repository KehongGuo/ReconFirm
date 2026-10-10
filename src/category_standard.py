"""The benchmark's own naming standard for the six HM3D ObjectNav categories.

HM3D files a flowerpot, a flower vase and a decorative plant all under
"plant", a monitor and a display under "tv_monitor", an armchair and a
beanbag under "chair". A detector asked for "plant" misses a flowerpot, and a
second opinion that answers "flowerpot" does not name the target. This module
states what each category covers, taken only from the goal objects of the
TRAIN split (never val), keeping names seen at least MIN_COUNT times.

  python src/category_standard.py --build     # writes data/hm3d_category_standard.json
"""
import argparse
import collections
import glob
import os
import gzip
import json
import re
from pathlib import Path

TRAIN = os.path.join(os.environ.get('SCOUT_OBJECTNAV_ROOT', ''), 'train', 'content')
FILE = Path(__file__).resolve().parent.parent / 'data' / 'hm3d_category_standard.json'
MIN_COUNT = 3
# For tv_monitor a name must also be at least 5% of its train goals: 8 whiteboards among ~430 tv
# goals made a whiteboard a tv (retest HY1NcmCgn3n:17); this drops whiteboard and bulletin board.
# Every other category keeps all names seen MIN_COUNT times.
MIN_SHARE = 0.05
SHARE_RULE = {'tv_monitor'}
WORD = {'tv_monitor': 'tv'}     # the word already in the category's own detector phrase


def build():
    names = collections.defaultdict(collections.Counter)
    for f in sorted(glob.glob(f'{TRAIN}/*.json.gz')):
        for goals in json.load(gzip.open(f))['goals_by_category'].values():
            for g in goals:
                names[g['object_category']][re.sub(r'_\d+$', '', g['object_name'])] += 1
    def keep(c, cnt):
        total = sum(cnt.values())
        return [n for n, k in cnt.most_common()
                if k >= MIN_COUNT and (c not in SHARE_RULE or k >= MIN_SHARE * total)]
    std = {c: keep(c, cnt) for c, cnt in sorted(names.items())}
    FILE.write_text(json.dumps({'source': 'objectnav_hm3d_v2 train goal object names', 'min_count': MIN_COUNT,
                                'min_share': MIN_SHARE, 'share_rule': sorted(SHARE_RULE), 'categories': std}, indent=1))
    return std


def load():
    return json.loads(FILE.read_text())['categories']


def members(target):
    return load().get(target, [])


def detector_aliases(target):
    """Names the detector should also search for: those not already containing the
    category's own word (a 'dining chair' is found by 'chair' anyway)."""
    word = WORD.get(target, target.replace('_', ' '))
    return [n for n in members(target) if word not in n and n != target]


def describe(target):
    """The phrase used in every prompt for this target under the benchmark standard."""
    names = [n for n in members(target) if n != target][:8]
    base = {'tv_monitor': 'television or computer monitor', 'plant': 'plant'}.get(target, target.replace('_', ' '))
    return f"{base} (in this benchmark that includes: {', '.join(names)})" if names else base


def boundaries(target, per=6):
    """What the benchmark files under its other categories, to draw the target's boundary: a
    second opinion called an upholstered armchair a 'sofa'; HM3D files armchairs as chair."""
    std = load()
    return '; '.join(f"{c.replace('_', ' ')}: {', '.join(n for n in names[:per])}"
                     for c, names in std.items() if c != target)


def canonical(name, target):
    """A member name answered by a model counts as the target."""
    key = ' '.join(str(name).lstrip('?').replace('_', ' ').lower().split())
    return target if key in members(target) else name


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', action='store_true')
    if ap.parse_args().build:
        for c, n in build().items():
            print(c, n)
    assert canonical('?flowerpot', 'plant') == 'plant' and canonical('chair', 'plant') == 'chair'
    assert 'flowerpot' in detector_aliases('plant') and 'dining chair' not in detector_aliases('chair')
