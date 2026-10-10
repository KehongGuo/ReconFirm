#!/bin/bash
# Forgetting guard data: 300 POPE (adversarial, COCO val2014) yes/no questions on the first 50 images.
set -e
D=${1:-data/pope}; mkdir -p $D/img
curl -sfL -o $D/coco_pope_adversarial.json https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_adversarial.json
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
Q = [json.loads(l) for l in open(f'{d}/coco_pope_adversarial.json')]
imgs = list(dict.fromkeys(q['image'] for q in Q))[:50]
open(f'{d}/pope_adv_300.jsonl', 'w').write(''.join(json.dumps(q) + '\n' for q in Q if q['image'] in set(imgs)))
open(f'{d}/images.txt', 'w').write('\n'.join(imgs) + '\n')
PY
while read f; do [ -f $D/img/$f ] || curl -sf -o $D/img/$f http://images.cocodataset.org/val2014/$f; done < $D/images.txt
