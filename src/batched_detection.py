"""One Grounding DINO forward pass for every robot that looks at the same tick.

`perception.OpenVocabDetector.detect` handles one image; with four robots that
is four passes per detection tick. This batches the images, then post-processes
each exactly as `detect` does (same threshold, same label mapping, same
top-k and box clipping). It is only used after `check` shows it reproduces the
per-image results on real frames; embeddings are not needed here and are
not returned.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from perception import VisualDetection


@torch.no_grad()
def detect_batch(detector, rgbs):
    assert detector.backend == 'gdino'
    imgs = [Image.fromarray(np.asarray(r)[..., :3].astype(np.uint8)) for r in rgbs]
    inputs = detector.processor(images=imgs, text=[detector.text] * len(imgs), return_tensors='pt')
    inputs = {k: (v.to(detector.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
    outputs = detector.model(**inputs)
    results = detector.processor.post_process_grounded_object_detection(
        outputs, inputs['input_ids'], threshold=detector.threshold,
        target_sizes=[(im.height, im.width) for im in imgs])
    out = []
    for im, r in zip(imgs, results):
        labels = [detector._label_from_phrase(t) for t in r.get('text_labels', r.get('labels', []))]
        dets = []
        order = torch.argsort(r['scores'], descending=True)[: detector.max_detections]
        for i in order:
            i = int(i)
            box = [int(v) for v in r['boxes'][i].tolist()]
            x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(im.width, box[2]), min(im.height, box[3])
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            dets.append(VisualDetection(label=labels[i] if i < len(labels) else '?',
                                        score=float(r['scores'][i]), box=(x0, y0, x1, y1),
                                        pixels=(x1 - x0) * (y1 - y0), embedding=None))
        out.append(dets)
    return out


def check(detector, frames):
    """Compare batched and per-image detection on the same frames."""
    single = [detector.detect(f) for f in frames]
    batched = detect_batch(detector, frames)
    same, worst = 0, 0.0
    for a, b in zip(single, batched):
        ok = len(a) == len(b) and all(p.label == q.label and max(abs(u - v) for u, v in zip(p.box, q.box)) <= 2
                                      for p, q in zip(a, b))
        same += ok
        worst = max([worst] + [abs(p.score - q.score) for p, q in zip(a, b)])
    return same, len(frames), worst
