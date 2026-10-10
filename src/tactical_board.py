"""The tactical board: one shared, versioned picture of what the team knows.

Built only from `SearchWorld`'s observable side -- the observed map, the tracker,
the robots' own poses -- so the brain and Jev read exactly what a real team
could have. Two renderings of the same snapshot: a compact JSON for text
reasoning, and a top-down image in which frontiers, robots and landmarks carry
the same ids as the JSON, so the brain can point at what it sees.

Frontier ids persist across recomputation (a new cluster inherits the id of the
old cluster whose centroid is within MATCH_RADIUS), because a plan that says
"r0 to F3" has to mean the same exit when it arrives seconds later.
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from observed_map import FREE, OCCUPIED, UNKNOWN

MATCH_RADIUS = 1.5
LANDMARK_RADIUS = 3.0
MAX_FRONTIERS_ON_BOARD = 8
ROBOT_COLOURS = [(230, 60, 60), (40, 120, 230), (30, 170, 80), (200, 120, 20)]
FRONTIER_COLOURS = [(255, 170, 0), (190, 70, 220), (0, 190, 190), (240, 90, 150),
                    (120, 200, 40), (90, 120, 255), (255, 110, 60), (160, 160, 40)]

_FONT = None


def font(size=13):
    global _FONT
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size)
    except OSError:
        return ImageFont.load_default()


class FrontierRegistry:
    """Frontier identity that survives the map growing under it.

    Matching on centroids alone gave an exit a new id whenever a robot's view
    reshaped it, so ids lived about two seconds -- less than the brain takes to
    answer -- and 37 of 82 brain assignments in the first debug episode arrived
    pointing at ids that no longer existed. A new cluster now inherits the id of
    the old cluster whose (slightly dilated) cells it overlaps most, falling back
    to centroid distance; and `successor` maps a retired id to the live exit
    nearest to where it last was, so a plan made against an older board still
    means something when it lands.
    """
    OVERLAP_DILATE = 3        # cells
    SUCCESSOR_RADIUS = 2.5    # metres

    def __init__(self):
        self.active = {}      # fid -> frontier dict (cells, size, centroid, goal)
        self._n = 0
        self.retired = set()
        self.last_centroid = {}

    def successor(self, fid):
        if fid in self.active:
            return fid
        c = self.last_centroid.get(fid)
        if c is None:
            return None
        best, best_d = None, self.SUCCESSOR_RADIUS
        for oid, f in self.active.items():
            d = math.hypot(f['centroid'][0] - c[0], f['centroid'][1] - c[1])
            if d <= best_d:
                best, best_d = oid, d
        return best

    def update(self, world, fields):
        """Recompute clusters, keep ids stable, attach per-robot path costs."""
        from scipy import ndimage
        m = world.map
        fresh = m.frontiers()
        # Old clusters painted into a label image, dilated so a cluster that
        # shifted by a cell or two still overlaps its former self.
        old_ids = list(self.active)
        label = None
        if old_ids:
            label = np.zeros((m.n, m.n), np.int32)
            for k, fid in enumerate(old_ids, start=1):
                ii, jj = self.active[fid]['cells']
                label[ii, jj] = k
            label = ndimage.grey_dilation(label, size=(2 * self.OVERLAP_DILATE + 1,) * 2)
        scored = []
        for n_, f in enumerate(fresh):
            ranking = []
            if label is not None:
                ii, jj = f['cells']
                ks, counts = np.unique(label[ii, jj], return_counts=True)
                ranking = [(int(c), old_ids[k - 1]) for k, c in zip(ks, counts) if k > 0]
            scored.append((n_, sorted(ranking, reverse=True)))
        taken, out = set(), {}
        # Largest overlaps claim their ids first.
        order = sorted(scored, key=lambda s: -(s[1][0][0] if s[1] else 0))
        for n_, ranking in order:
            f = fresh[n_]
            best = next((fid for _c, fid in ranking if fid not in taken), None)
            if best is None:
                cx, cz = f['centroid']
                best_d = MATCH_RADIUS
                for fid, old in self.active.items():
                    if fid in taken:
                        continue
                    d = math.hypot(cx - old['centroid'][0], cz - old['centroid'][1])
                    if d <= best_d:
                        best, best_d = fid, d
            if best is None:
                self._n += 1
                best = f'F{self._n}'
            taken.add(best)
            f['id'] = best
            f['cost'], f['goal'] = {}, {}
            for name, field_ in fields.items():
                gi, gj, d = m.goal_cell_for(field_, f)
                f['cost'][name] = d
                if gi is not None and np.isfinite(d):
                    f['goal'][name] = tuple(float(v) for v in m.world(gi, gj))
            if not f['goal']:
                continue       # nobody can reach it through known space
            out[best] = f
            self.last_centroid[best] = f['centroid']
        self.retired |= set(self.active) - set(out)
        self.active = out
        return out


def nearby_labels(world, x, z, radius=LANDMARK_RADIUS, exclude=None):
    """Distinct landmark labels seen near a point, most-viewed first."""
    tracks = world.tracker.near(x, z, radius, exclude=exclude)
    tally = {}
    for tr in tracks:
        tally[tr.label] = tally.get(tr.label, 0) + len(tr.views)
    return [k for k, _ in sorted(tally.items(), key=lambda kv: -kv[1])][:6]


def snapshot(world, registry, tasks, t, target, mission_s, candidates, version, plan_note=None):
    """The JSON half of the board. `tasks` maps robot -> task dict or None."""
    robots = []
    for name in world.names:
        x, z, yaw = world.pose(name)
        task = tasks.get(name)
        desc = 'none'
        if task:
            desc = (f"exploring frontier {task['fid']}" if task['kind'] == 'frontier'
                    else f"investigating candidate {task['track_id']}")
        robots.append({'id': name, 'xz': [round(x, 1), round(z, 1)],
                       'heading_deg': round(math.degrees(yaw)) % 360,
                       'task': desc, 'status': world.movers[name].status,
                       'travelled_m': round(world.movers[name].travelled, 1)})
    fronts = sorted(registry.active.values(),
                    key=lambda f: min(f['cost'].values()))[:MAX_FRONTIERS_ON_BOARD]
    frontiers = []
    for f in fronts:
        cx, cz = f['centroid']
        assigned = [n for n, tk in tasks.items() if tk and tk.get('fid') == f['id']]
        frontiers.append({
            'id': f['id'], 'width_m': round(f['size'] * world.map.res, 1),
            'xz': [round(cx, 1), round(cz, 1)],
            'path_m': {n: (round(d, 1) if np.isfinite(d) else None) for n, d in f['cost'].items()},
            'landmarks_near': nearby_labels(world, cx, cz),
            'assigned_to': assigned})
    cands = [dict(c) for c in candidates]
    landmarks = sorted(world.tracker.tracks.values(), key=lambda tr: -len(tr.views))
    seen = [{'label': tr.label, 'xz': [round(float(tr.position[0]), 1), round(float(tr.position[2]), 1)],
             'views': len(tr.views)} for tr in landmarks if len(tr.views) >= 2][:25]
    return {'board_version': version, 'target': target, 'time_s': round(t, 1),
            'time_left_s': round(mission_s - t, 1),
            'explored_free_m2': round(world.map.known_area_m2(), 1),
            'robots': robots, 'frontiers': frontiers, 'target_candidates': cands,
            'landmarks': seen, 'previous_plan': plan_note}


def render(world, registry, tasks, candidates, size=512, frontier_ids=None, show_legend=True):
    """Top-down board image with the same ids as the JSON."""
    m = world.map
    s = m.state()
    bb = m.known_fraction_bbox()
    if bb is None:
        return Image.new('RGB', (size, size), (60, 60, 60))
    pad = 12
    i0, i1, j0, j1 = bb[0] - pad, bb[1] + pad, bb[2] - pad, bb[3] + pad
    side = max(i1 - i0, j1 - j0)
    ci, cj = (i0 + i1) // 2, (j0 + j1) // 2
    i0, j0 = ci - side // 2, cj - side // 2
    i1, j1 = i0 + side, j0 + side
    crop = np.full((side, side), UNKNOWN, np.uint8)
    si0, si1, sj0, sj1 = max(i0, 0), min(i1, m.n), max(j0, 0), min(j1, m.n)
    crop[si0 - i0:si1 - i0, sj0 - j0:sj1 - j0] = s[si0:si1, sj0:sj1]
    img = np.zeros((side, side, 3), np.uint8)
    img[crop == UNKNOWN] = (70, 70, 70)
    img[crop == FREE] = (235, 235, 230)
    img[crop == OCCUPIED] = (25, 25, 25)
    vis = m.visited[max(i0, 0):min(i1, m.n), max(j0, 0):min(j1, m.n)]
    sub = img[si0 - i0:si1 - i0, sj0 - j0:sj1 - j0]
    sub[vis & (crop[si0 - i0:si1 - i0, sj0 - j0:sj1 - j0] == FREE)] = (205, 220, 240)
    fronts = sorted(registry.active.values(), key=lambda f: min(f['cost'].values()))
    fronts = fronts[:MAX_FRONTIERS_ON_BOARD] if frontier_ids is None else \
        [f for f in fronts if f['id'] in frontier_ids]
    colour_of = {}
    for k, f in enumerate(fronts):
        col = FRONTIER_COLOURS[k % len(FRONTIER_COLOURS)]
        colour_of[f['id']] = col
        ii, jj = f['cells']
        ok = (ii >= i0) & (ii < i1) & (jj >= j0) & (jj < j1)
        img[ii[ok] - i0, jj[ok] - j0] = col
    scale = size / side
    im = Image.fromarray(img).resize((size, size), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    f13, f11 = font(13), font(11)

    def px(x, z):
        i, j = m.cell(x, z)
        return (float(j - j0 + 0.5) * scale, float(i - i0 + 0.5) * scale)

    for tr in world.tracker.tracks.values():
        if len(tr.views) < 2 and max(tr.scores) < 0.4:
            continue
        x, y = px(tr.position[0], tr.position[2])
        dr.text((x + 3, y - 6), tr.label.replace('_', ' ')[:12], fill=(90, 60, 30), font=f11)
        dr.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(150, 100, 40))
    for c in candidates:
        x, y = px(*c['xz'])
        col = (220, 0, 0) if c.get('status') != 'rejected' else (120, 120, 120)
        dr.ellipse([x - 7, y - 7, x + 7, y + 7], outline=col, width=3)
        dr.text((x + 8, y - 16), c['id'], fill=col, font=f13)
    for f in fronts:
        cx, cz = f['centroid']
        x, y = px(cx, cz)
        col = colour_of[f['id']]
        dr.rectangle([x - 13, y - 9, x + 13, y + 9], fill=(255, 255, 255), outline=col, width=2)
        dr.text((x - 10, y - 8), f['id'], fill=(0, 0, 0), font=f13)
    for k, name in enumerate(world.names):
        col = ROBOT_COLOURS[k % len(ROBOT_COLOURS)]
        trail = world.trails[name]
        if len(trail) > 1:
            dr.line([px(a, b) for a, b in trail], fill=col, width=2)
        x, z, yaw = world.pose(name)
        X, Y = px(x, z)
        fx, fz = x - math.sin(yaw) * 0.8, z - math.cos(yaw) * 0.8
        dr.line([X, Y, *px(fx, fz)], fill=col, width=3)
        dr.ellipse([X - 7, Y - 7, X + 7, Y + 7], fill=col, outline=(0, 0, 0))
        dr.text((X - 6, Y - 7), name[-1], fill=(255, 255, 255), font=f13)
        task = tasks.get(name)
        if task and task.get('goal'):
            gx, gy = px(*task['goal'])
            dr.line([X, Y, gx, gy], fill=col, width=1)
    if show_legend:
        dr.rectangle([0, 0, 200, 18], fill=(0, 0, 0))
        dr.text((4, 2), f'1 cell = {m.res:.1f} m, {side * m.res:.0f} m across', fill=(255, 255, 255), font=f11)
    return im


def png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    return buf.getvalue()
