"""Real perception: an open-vocabulary detector run on the rendered image.

The previous detector read object identities out of the simulator's semantic
channel and added synthetic noise to pretend it had not. That is privileged
information, and it quietly decided every result that followed -- the noise
curve I invented, rather than any real visual difficulty, was what the whole
system was reacting to.

Here a robot sees only pixels. It proposes boxes for a text vocabulary and
reports what it thinks each one is, with the confidence the detector actually
produces. Ground truth stays on the scoring side, where it belongs: matching a
detection to the object it overlaps, so we can say afterwards whether the robot
was right, without the robot ever having been told.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

MODEL_ID = "google/owlv2-base-patch16-ensemble"

# mpcat40's category names are database identifiers, and feeding them to an
# open-vocabulary detector as if they were English wastes the one thing it is
# for. "a photo of a board_panel" is not a sentence, and it turned out to match
# almost anything: across 82 planted objects the detector answered board_panel
# 24 times, for sofas, pictures, televisions and cabinets alike. The abstract
# bucket names are worse than useless -- "seating", "furniture", "objects" and
# "misc" describe a taxonomy, not an appearance -- so they are dropped rather
# than translated, and the label they would have won goes to a word that
# actually denotes something visible.
PROMPTS = {
    "tv_monitor": "television screen",
    "chest_of_drawers": "chest of drawers",
    "gym_equipment": "exercise equipment",
    "board_panel": "whiteboard",
    "appliances": "kitchen appliance",
    "shelving": "shelf",
    "lighting": "lamp",
    "blinds": "window blinds",
    "counter": "kitchen counter",
    "cushion": "cushion on a seat",
    "clothes": "clothing",
    "plant": "potted plant",
    "picture": "framed picture on a wall",
    "curtain": "window curtain",
    "shower": "shower cubicle",
    "cabinet": "cabinet",
    "seating": "",      # dropped: a taxonomy bucket, not a thing you can see
    "furniture": "",
    "objects": "",
    "misc": "",
    "stairs": "",
}


def prompt_for(category: str) -> str:
    """The phrase to search for, given an mpcat40 category name.

    Empty means the category is not searchable and should be left out.
    """
    if category in PROMPTS:
        return PROMPTS[category]
    return category.replace("_", " ")


@dataclass
class VisualDetection:
    """One box the detector proposed, in image and world terms."""

    label: str            # the detector's best text match
    score: float          # the detector's own confidence
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels
    pixels: int
    # Filled in by the grounding step, from depth + camera pose.
    position: np.ndarray | None = None
    distance: float = float("inf")
    # The detector's own embedding of this box, taken from the forward pass we
    # already ran. The request gate had nothing but hand-derived scalars to look
    # at and lost to a single raw feature; this is the cheapest way to give it
    # something that actually saw the object.
    embedding: np.ndarray | None = None
    # Scoring-side only: which simulator object this box actually covers.
    matched_semantic_id: int = -1
    match_iou: float = 0.0


GDINO_ID = "IDEA-Research/grounding-dino-base"


class OpenVocabDetector:
    """One open-vocabulary detector, shared by every robot on the team.

    One instance serves the whole team -- four small models on one GPU would be
    an implementation detail masquerading as an architecture, and nothing in the
    system depends on the detectors being separate.

    Grounding DINO by default. Chosen on this task rather than on published
    numbers: over 101 keyframes against planted objects of known position, at a
    matched budget of boxes it found 15 of 17 where OWLv2 found 14 with 35% more
    boxes, and 12 vs 11 at a third of that budget. It is also a third faster.
    The advantage is in localisation only -- of the detections that land on a
    planted object, both name it correctly about the same fraction of the time,
    so the 29% label accuracy this system has to cope with is not fixed by the
    swap. OWLv2 is kept selectable because it is what every result before this
    was measured on.
    """

    def __init__(self, vocabulary: list[str], device: str = "cuda",
                 threshold: float = 0.25, max_detections: int = 12,
                 backend: str = "gdino", model_id: str | None = None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.backend = backend
        self.device = device if torch.cuda.is_available() else "cpu"
        mid = model_id or (GDINO_ID if backend == "gdino" else MODEL_ID)
        self.processor = AutoProcessor.from_pretrained(mid)
        # fp32: Grounding DINO's text tower runs a float32 path internally and
        # half precision dies inside it.
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid)
        self.model = self.model.to(self.device).eval()
        # 0.25 from a sweep against planted objects of known position. Lowering
        # it is not the trade it looks like: every surviving box becomes a graph
        # node competing for the brain's budget, so the cheap direction is the
        # strict one.
        self.threshold = threshold
        self.max_detections = max_detections
        self.set_vocabulary(vocabulary)

    def set_vocabulary(self, vocabulary: list[str]) -> None:
        """Text queries the detector will look for.

        The category name is translated into a real phrase first and categories
        with no visual meaning are dropped; `self.vocabulary` still holds the
        mpcat40 names, so what the detector reports stays in the label space
        everything downstream expects.
        """
        keep, phrases = [], []
        for v in dict.fromkeys(vocabulary):
            if not v:
                continue
            phrase = prompt_for(v)
            if not phrase:
                continue
            keep.append(v)
            phrases.append(phrase)
        self.vocabulary = keep
        self.phrases = phrases
        # OWLv2 scores one sentence per query; Grounding DINO takes a single
        # period-separated string and reports back whichever phrase matched.
        self.queries = [f"a photo of a {p}" for p in phrases]
        self.text = ". ".join(phrases) + "."

    def _label_from_phrase(self, phrase: str) -> str:
        p = str(phrase).strip().lower()
        # Exact phrase first: with substring matching alone "board" matched "headboard" or
        # "whiteboard" when those were listed earlier, and a wall panel became a bed or a tv.
        for lab, ph in zip(self.vocabulary, self.phrases):
            if p == ph.lower():
                return lab
        for lab, ph in zip(self.vocabulary, self.phrases):
            if p == ph.lower() or p in ph.lower() or ph.lower() in p:
                return lab
        return "?"

    @torch.no_grad()
    def detect(self, rgb: np.ndarray) -> list[VisualDetection]:
        """Everything the detector finds in one frame, with its embeddings."""
        img = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8))
        owl = self.backend != "gdino"
        if owl:
            inputs = self.processor(text=[self.queries], images=img,
                                    return_tensors="pt")
        else:
            inputs = self.processor(images=img, text=self.text, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        outputs = self.model(**inputs)

        if owl:
            target = torch.tensor([[img.height, img.width]], device=self.device)
            r = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target, threshold=self.threshold)[0]
            labels = [self.vocabulary[int(i)] if int(i) < len(self.vocabulary) else "?"
                      for i in r["labels"]]
            # OWLv2's per-box embedding, in the space the text queries live in.
            per_box = outputs.class_embeds[0]
            scores_all = outputs.logits[0].sigmoid().max(dim=-1).values
        else:
            r = self.processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=self.threshold,
                target_sizes=[(img.height, img.width)])[0]
            labels = [self._label_from_phrase(t)
                      for t in r.get("text_labels", r.get("labels", []))]
            # The decoder's query embeddings: one per candidate box.
            per_box = outputs.last_hidden_state[0]
            scores_all = outputs.logits[0].sigmoid().max(dim=-1).values

        # Recover which candidate each surviving box came from, so its embedding
        # can travel with it. post_process returns kept boxes in query order.
        kept_idx = torch.nonzero(scores_all > self.threshold).flatten()
        if kept_idx.numel() != len(r["scores"]):
            # Cannot prove postprocessed box/query identity in this case.
            # Keep detections, but never attach another query's feature.
            kept_idx = None

        out: list[VisualDetection] = []
        order = torch.argsort(r["scores"], descending=True)[: self.max_detections]
        for i in order:
            i = int(i)
            box = [int(v) for v in r["boxes"][i].tolist()]
            x0, y0, x1, y1 = (max(0, box[0]), max(0, box[1]),
                              min(img.width, box[2]), min(img.height, box[3]))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            emb = None
            if kept_idx is not None and i < len(kept_idx):
                emb = per_box[int(kept_idx[i])].float().cpu().numpy()
            out.append(VisualDetection(
                label=labels[i] if i < len(labels) else "?",
                score=float(r["scores"][i]),
                box=(x0, y0, x1, y1),
                pixels=(x1 - x0) * (y1 - y0),
                embedding=emb))
        return out


def foreground_depth(depths) -> float:
    """Depth of the nearest cluster of a box's pixels: the object usually stands in front of
    whatever else its box covers. The plain median of a small vase's box was the wall behind
    it, and the dog photographed that wall 14 times (retest CrMo8WxCyVb:12)."""
    d = np.asarray(depths, dtype=np.float64).ravel()
    near = np.percentile(d, 10)
    sel = d[(d >= near - 0.1) & (d <= near + 0.3)]
    return float(np.median(sel)) if sel.size >= 0.08 * d.size else float(np.median(d))


FLOOR_MARGIN_M = 0.05     # box pixels this close to the floor are the floor, not the object


def ground_detections(detections: list[VisualDetection], depth: np.ndarray,
                      sim, robot_name: str, floor_y: float | None = None) -> list[VisualDetection]:
    """Put each box somewhere in the world, using depth and the camera pose.

    Median depth over the box rather than the centre pixel: the centre often
    lands on a hole in an MP3D scan, and one bad pixel would throw the object
    metres away. With floor_y, the box's floor pixels are dropped first: the
    nearest cluster of a sofa's or bed's box was the floor in front of it, and
    the dog's own estimate sat at 0.24 of the true distance (HM3D val rerun: 69
    of 74 right-object declarations away from every view point).
    """
    robot = sim.team[sim.index[robot_name]]
    eye, rotation = sim.camera_transform(robot_name)

    d = np.asarray(depth)
    h, w = d.shape[:2]
    hfov = np.deg2rad(robot.hfov)
    fx = (w / 2.0) / np.tan(hfov / 2.0)
    fy = fx  # square pixels; habitat's vertical fov follows from the aspect

    for det in detections:
        x0, y0, x1, y1 = det.box
        x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
        y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
        det.position, det.distance = None, float("inf")
        patch = d[y0:y1, x0:x1]
        ok = np.isfinite(patch) & (patch > 0.05)
        if floor_y is not None and ok.any():
            jj, ii = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            ray_y = rotation[1, 0] * (jj - w / 2.0) / fx - rotation[1, 1] * (ii - h / 2.0) / fy - rotation[1, 2]
            above = eye[1] + ray_y * patch > floor_y + FLOOR_MARGIN_M
            if (ok & above).any():
                ok &= above
        patch = patch[ok]
        if patch.size == 0:
            continue
        depth_m = foreground_depth(patch)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # Pixel offsets from the principal point, in camera units.
        u = (cx - w / 2.0) / fx
        v = (cy - h / 2.0) / fy
        # Habitat pinhole depth is optical-axis depth, NOT Euclidean range.
        # Its camera looks down -Z, with +Y up. Do not normalize this ray.
        camera_point = np.array([u, -v, -1.0]) * depth_m
        det.position = eye + rotation @ camera_point
        det.distance = float(np.linalg.norm(camera_point))
    return detections


def crop_box(rgb_frame, box: tuple[int, int, int, int], context: float = 0.8,
             min_side: int = 128, max_side: int = 336) -> bytes | None:
    """Cut out what the detector boxed, with its surroundings, to show the brain.

    The other crop function in this codebase cuts along the semantic mask, which
    gives a perfect silhouette of the object -- and is exactly the privileged
    information we took away from the detector. A real robot can only offer the
    rectangle its detector drew, background and all.

    Two choices here follow from what the brain is actually being asked. The
    question is whether an object belongs in the room it is in, so the room has
    to be visible: the box is expanded by `context` of its own size rather than
    a fixed few pixels, which keeps the object framed by its surroundings at any
    distance. And the result is floored at `min_side`, because a tight crop off
    a distant object came out around 50x66 pixels, which is below what a vision
    model can identify anything from -- it answered from the detector's text
    label instead, and inherited its mistakes.
    """
    import io

    rgb = np.asarray(rgb_frame)[..., :3]
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    px, py = int(bw * context / 2), int(bh * context / 2)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(w, x1 + px), min(h, y1 + py)
    if (x1 - x0) < 4 or (y1 - y0) < 4:
        return None

    img = Image.fromarray(rgb[y0:y1, x0:x1].astype(np.uint8))
    longest = max(img.size)
    if longest < min_side:
        s = min_side / longest
    elif longest > max_side:
        s = max_side / longest
    else:
        s = 1.0
    if s != 1.0:
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def match_to_truth(detections: list[VisualDetection], semantic_frame: np.ndarray,
                   min_overlap: float = 0.15,
                   structural_ids: set[int] | None = None) -> list[VisualDetection]:
    """Scoring side only: which simulator object does each box actually cover?

    Used to decide afterwards whether a detection was right. Nothing the robot
    acts on may read this.

    `structural_ids` names the walls, floors and ceilings. They are excluded
    from the vote rather than allowed to win it. A box drawn around a picture or
    a towel is mostly the wall behind it, so plain majority share attributed
    correct detections of thin objects to the surface they hang on -- the
    detector's vocabulary contains no structural words at all, so a box can
    never be *about* a wall. If nothing non-structural clears `min_overlap` the
    detection stays unmatched and counts as a false positive, which is the
    honest outcome for a box that really did land on blank wall.
    """
    sem = np.asarray(semantic_frame)
    for det in detections:
        x0, y0, x1, y1 = det.box
        window = sem[y0:y1, x0:x1]
        if window.size == 0:
            continue
        ids, counts = np.unique(window, return_counts=True)
        keep = ids >= 0
        ids, counts = ids[keep], counts[keep]
        if structural_ids:
            keep = np.array([int(i) not in structural_ids for i in ids], dtype=bool)
            ids, counts = ids[keep], counts[keep]
        if ids.size == 0:
            continue
        best = int(np.argmax(counts))
        share = float(counts[best]) / float(window.size)
        if share >= min_overlap:
            det.matched_semantic_id = int(ids[best])
            det.match_iou = share
    return detections
