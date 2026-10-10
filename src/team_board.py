"""Tactical board for the aerial-legged team.

Same contract as `tactical_board`: built only from what the team observed, with
ids shared between the JSON and the image. Adds what a scheduler of a mixed
team has to see: robot type, battery seconds left and flight time back to the
dog for each drone, candidate status (who is inspecting or verifying it, all
second opinions with their descriptions, the dog's walking distance to it).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

import tactical_board as tb
from observed_map import FREE, OCCUPIED
from team_agents import flight_home_s

MAX_FRONTIERS = 8
COLOUR = {'d0': (230, 60, 60), 'd1': (40, 120, 230), 'd2': (30, 170, 80), 'dog0': (150, 90, 20)}


def fit(snap, budget):
    """Trim the board, least useful first, until its JSON fits the budget:
    landmarks, then closed candidates, then opinion descriptions, then
    frontier landmark lists, then the weakest open candidates."""
    import json

    def size():
        return len(json.dumps(snap, separators=(',', ':')))
    trimmed = []
    while size() > budget and len(snap['landmarks']) > 4:
        snap['landmarks'].pop()
        trimmed.append('landmark')
    cands = snap['target_candidates']
    while size() > budget and any(c['status'] != 'open' for c in cands):
        cands.remove(next(c for c in reversed(cands) if c['status'] != 'open'))
        trimmed.append('closed_candidate')
    for c in cands:
        if size() <= budget:
            break
        for o in c['second_opinions']:
            o.pop('description', None)
        trimmed.append('descriptions')
    for f in snap['frontiers']:
        if size() <= budget:
            break
        f['landmarks_near'] = f['landmarks_near'][:2]
        trimmed.append('frontier_landmarks')
    cands.sort(key=lambda c: -c['max_score'])
    while size() > budget and len(cands) > 1:
        cands.pop()
        trimmed.append('open_candidate')
    while size() > budget and len(snap['frontiers']) > 3:
        snap['frontiers'].pop()
        trimmed.append('frontier')
    if trimmed:
        snap['trimmed'] = sorted(set(trimmed))
    return snap


def task_text(task):
    if not task:
        return 'none'
    k, tg = task['kind'], task.get('target')
    return f'{k}({tg})' if tg else k


def snapshot(world, registry, tasks, t, target, mission_s, candidates, version, plan_note=None):
    robots = []
    for name, b in world.bodies.items():
        if b.state == 'down':
            continue            # lost for the mission; the scheduler kept assigning tasks to them
        r = {'id': name, 'type': 'drone' if b.aerial else 'dog',
             'xz': [round(b.x, 1), round(b.z, 1)], 'task': task_text(tasks.get(name)),
             'status': b.state if b.aerial else b.status}
        if b.aerial:
            r['battery_s'] = round(max(b.battery, 0.0))
            r['flight_to_dog_s'] = round(flight_home_s(world, name), 1)
        robots.append(r)
    fronts = sorted(registry.active.values(), key=lambda f: min(f['cost'].values()))[:MAX_FRONTIERS]
    frontiers = []
    for f in fronts:
        cx, cz = f['centroid']
        frontiers.append({
            'id': f['id'], 'width_m': round(f['size'] * world.map.res, 1), 'xz': [round(cx, 1), round(cz, 1)],
            # Who can take it, and at what distance, is in the per-robot
            # `options` menu the runner adds.
            'landmarks_near': tb.nearby_labels(world, cx, cz),
            'assigned_to': [n for n, tk in tasks.items() if tk and tk.get('target') == f['id']]})
    seen = sorted(world.tracker.tracks.values(), key=lambda tr: -len(tr.views))
    landmarks = [{'label': tr.label, 'xz': [round(float(tr.position[0]), 1), round(float(tr.position[2]), 1)]}
                 for tr in seen if len(tr.views) >= 2][:14]
    return {'board_version': version, 'target': target, 'time_s': round(t, 1),
            'time_left_s': round(mission_s - t, 1), 'explored_m2': round(world.map.known_area_m2(), 1),
            'robots': robots, 'frontiers': frontiers, 'target_candidates': candidates,
            'landmarks': landmarks, 'previous_plan': plan_note}


def render(world, registry, tasks, candidates, size=512, legend=True):
    m = world.map
    der = m._derived()
    s, known = der['state'], der['known']
    bb = m.known_fraction_bbox()
    if bb is None:
        return Image.new('RGB', (size, size), (60, 60, 60))
    pad = 12
    i0, i1, j0, j1 = bb[0] - pad, bb[1] + pad, bb[2] - pad, bb[3] + pad
    side = max(i1 - i0, j1 - j0)
    ci, cj = (i0 + i1) // 2, (j0 + j1) // 2
    i0, j0 = ci - side // 2, cj - side // 2
    i1, j1 = i0 + side, j0 + side
    img = np.full((side, side, 3), 70, np.uint8)
    si0, si1, sj0, sj1 = max(i0, 0), min(i1, m.n), max(j0, 0), min(j1, m.n)
    sub = img[si0 - i0:si1 - i0, sj0 - j0:sj1 - j0]
    ks, ss = known[si0:si1, sj0:sj1], s[si0:si1, sj0:sj1]
    air_blocked = m.occ_air[si0:si1, sj0:sj1] > 0
    sub[ks] = (235, 235, 230)
    sub[ss == OCCUPIED] = (150, 150, 150)       # blocks the dog only
    sub[air_blocked] = (25, 25, 25)              # blocks everyone
    vis = m.visited[si0:si1, sj0:sj1] & ks
    sub[vis] = (205, 220, 240)
    fronts = sorted(registry.active.values(), key=lambda f: min(f['cost'].values()))[:MAX_FRONTIERS]
    for k, f in enumerate(fronts):
        col = tb.FRONTIER_COLOURS[k % len(tb.FRONTIER_COLOURS)]
        ii, jj = f['cells']
        ok = (ii >= i0) & (ii < i1) & (jj >= j0) & (jj < j1)
        img[ii[ok] - i0, jj[ok] - j0] = col
    scale = size / side
    im = Image.fromarray(img).resize((size, size), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    f13, f11 = tb.font(13), tb.font(11)

    def px(x, z):
        i, j = m.cell(x, z)
        return (float(j - j0 + 0.5) * scale, float(i - i0 + 0.5) * scale)

    for tr in world.tracker.tracks.values():
        if len(tr.views) < 2 and max(tr.scores) < 0.4:
            continue
        x, y = px(tr.position[0], tr.position[2])
        dr.text((x + 3, y - 6), tr.label.replace('_', ' ')[:12], fill=(90, 60, 30), font=f11)
    for c in candidates:
        x, y = px(*c['xz'])
        col = (220, 0, 0) if c['status'] == 'open' else (120, 120, 120)
        dr.ellipse([x - 7, y - 7, x + 7, y + 7], outline=col, width=3)
        dr.text((x + 8, y - 16), c['id'], fill=col, font=f13)
    for k, f in enumerate(fronts):
        x, y = px(*f['centroid'])
        dr.rectangle([x - 13, y - 9, x + 13, y + 9], fill=(255, 255, 255),
                     outline=tb.FRONTIER_COLOURS[k % len(tb.FRONTIER_COLOURS)], width=2)
        dr.text((x - 10, y - 8), f['id'], fill=(0, 0, 0), font=f13)
    for name, b in world.bodies.items():
        col = COLOUR.get(name, (200, 200, 200))
        trail = world.trails[name]
        if len(trail) > 1:
            dr.line([px(a, c) for a, c in trail], fill=col, width=2)
        X, Y = px(b.x, b.z)
        if b.state == 'down':
            dr.line([X - 7, Y - 7, X + 7, Y + 7], fill=col, width=3)
            dr.line([X - 7, Y + 7, X + 7, Y - 7], fill=col, width=3)
            continue
        fx, fz = b.x - math.sin(b.yaw) * 0.8, b.z - math.cos(b.yaw) * 0.8
        dr.line([X, Y, *px(fx, fz)], fill=col, width=3)
        if b.aerial:
            a = b.yaw
            pts = [(X - 9 * math.sin(a), Y - 9 * math.cos(a)),
                   (X - 7 * math.sin(a + 2.4), Y - 7 * math.cos(a + 2.4)),
                   (X - 7 * math.sin(a - 2.4), Y - 7 * math.cos(a - 2.4))]
            dr.polygon(pts, fill=col, outline=(0, 0, 0))
            dr.text((X + 8, Y + 2), f"{name[-1]} {max(b.battery, 0):.0f}s", fill=col, font=f11)
        else:
            dr.rectangle([X - 8, Y - 8, X + 8, Y + 8], fill=col, outline=(0, 0, 0))
            dr.text((X - 5, Y - 7), 'D', fill=(255, 255, 255), font=f13)
        task = tasks.get(name)
        if task and task.get('goal'):
            dr.line([X, Y, *px(*task['goal'])], fill=col, width=1)
    if legend:
        dr.rectangle([0, 0, 250, 18], fill=(0, 0, 0))
        dr.text((4, 2), f'{side * m.res:.0f} m across; grey = blocks dog only', fill=(255, 255, 255), font=f11)
    return im
