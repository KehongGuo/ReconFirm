"""Two-layer observed map for a mixed aerial / legged team.

Same rule as `observed_map.ObservedMap`: a cell is known only after some robot's
depth camera has seen it, and nothing from the simulator's navmesh or
annotations reaches a planner. What changes is that "blocked" depends on who is
moving. A drone cruising at 1.2-1.5 m clears tables, beds and sofas but not
walls or tall wardrobes; the dog is stopped by whatever is below its own headroom
(it walks under a table top). So every depth point is sorted into two height bands:

  ground band  FLOOR_BAND .. ground_top   obstacles for the dog (ground_top: its headroom + a margin)
  air band     AIR_LO .. AIR_HI           obstacles for a drone

and each layer gets its own inflated traversability. Frontiers are the same for
both -- known space touching unknown space -- but a frontier's goal cell is found
on the layer of the robot it is being costed for.

"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from observed_map import FLOOR_BAND, FREE, MAX_RANGE, OCCUPIED, UNKNOWN, ObservedMap

GROUND_TOP = 1.30             # default only: the world passes the dog's own headroom (ground_top)
GROUND_HEADROOM_MARGIN = 0.10 # above the dog's navmesh headroom: depth noise on an overhang's underside
# Obstacles for the drones: 0.95 m up to the highest flight level (1.5 m) + 0.15 m for body and
# props. 1.85 m made a 1.85 m basement ceiling (7MXmsvcQjpJ) an obstacle everywhere.
AIR_LO, AIR_HI = 0.95, 1.65
SEEN_HI = 1.85                # hits below this count as looked-at space (unchanged)
GROUND_RADIUS = 0.20          # the dog's own collision radius (team_sim ground navmesh); 0.25 closed tight bathrooms
AIR_RADIUS = 0.30
LAYERS = ('ground', 'air')
# Object search needs more than "some camera saw this cell": the first debug
# house was fully known after 20 s -- every room glimpsed through its door --
# while the only toilet stayed undetected by the drones and three drones idled
# for four minutes. A cell counts as searched only once a camera has seen it
# from within LOOK_RANGE, where the detector reliably fires on small objects;
# frontiers are the edge of searched space, so exploration continues into
# rooms seen only from afar (the coverage stage of exploration planners such
# as FUEL/RACER, reduced to a range test).
LOOK_RANGE = 3.0
SEARCH_MIN_CELLS = 4       # 10 cells dropped a 0.6 m exit


class AerialMap(ObservedMap):
    def __init__(self, center_xz, floor_y, size_m=80.0, res=0.1, ground_top=GROUND_TOP):
        super().__init__(center_xz, floor_y, size_m=size_m, res=res)
        # Obstacles for the dog end at its own headroom. A fixed 1.30 m band made table tops and other overhangs it
        # walks under into walls: once a drone mapped the table above it, every cell round the dog was blocked and
        # it stood still for the rest of the mission (uNb9QFRL6hY:v2s21, 1 reachable cell, 5 of 400 MP3D runs).
        self.ground_top = ground_top
        self.occ_air = np.zeros((self.n, self.n), dtype=np.uint8)
        self.seen = np.zeros((self.n, self.n), dtype=bool)
        self.looked = np.zeros((self.n, self.n), dtype=bool)
        self._disk_air = self._disk_of(AIR_RADIUS)
        self._disk_ground = self._disk_of(GROUND_RADIUS)

    def _disk_of(self, radius):
        r = int(np.ceil(radius / self.res))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return (xx ** 2 + yy ** 2) <= (radius / self.res) ** 2 + 1e-6

    # ------------------------------------------------------------ integration
    def integrate(self, depth, eye, rotation, hfov_deg, t, stride=2):
        d = np.asarray(depth, dtype=np.float32)
        h, w = d.shape[:2]
        fx = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        z = d[vs, us]
        ok = np.isfinite(z) & (z > 0.05) & (z < MAX_RANGE)
        u = (us[ok] - w / 2.0) / fx
        v = (vs[ok] - h / 2.0) / fx
        z = z[ok]
        cam = np.stack([u * z, -v * z, -z], axis=1)
        pts = cam @ np.asarray(rotation).T + np.asarray(eye)
        height = pts[:, 1] - self.floor_y
        i, j = self.cell(pts[:, 0], pts[:, 2])
        inside = self.inside(i, j)
        floor = inside & (np.abs(height) < FLOOR_BAND)
        occ_g = inside & (height >= FLOOR_BAND) & (height < self.ground_top)
        occ_a = inside & (height >= AIR_LO) & (height < AIR_HI)
        any_hit = inside & (height > -FLOOR_BAND) & (height < SEEN_HI)

        # Space the camera looked through is empty, whatever height the hit was.
        if np.any(any_hit):
            sel = np.flatnonzero(any_hit)[::4]
            ex, ez = float(np.asarray(eye)[0]), float(np.asarray(eye)[2])
            px, pz = pts[sel, 0], pts[sel, 2]
            length = np.hypot(px - ex, pz - ez)
            k = int(max(1, (length / (self.res * 0.5)).max()))
            frac = np.linspace(0.0, 1.0, k + 1)[None, :]
            keep = frac <= (1.0 - 0.15 / np.maximum(length, 1e-3))[:, None]
            rx = ex + (px - ex)[:, None] * frac
            rz = ez + (pz - ez)[:, None] * frac
            ri, rj = self.cell(rx[keep], rz[keep])
            m = self.inside(ri, rj)
            self.free_hits[ri[m], rj[m]] = 1
            self.seen[ri[m], rj[m]] = True
            close = keep & ((frac * length[:, None]) <= LOOK_RANGE)
            li, lj = self.cell(rx[close], rz[close])
            lm = self.inside(li, lj)
            self.looked[li[lm], lj[lm]] = True

        self.free_hits[i[floor], j[floor]] = 1
        self.occ_hits[i[occ_g], j[occ_g]] = 1
        self.occ_air[i[occ_a], j[occ_a]] = 1
        self.seen[i[any_hit], j[any_hit]] = True
        near = any_hit & (np.hypot(pts[:, 0] - float(np.asarray(eye)[0]),
                                   pts[:, 2] - float(np.asarray(eye)[2])) <= LOOK_RANGE)
        self.looked[i[near], j[near]] = True
        for mask in (floor, occ_g, occ_a):
            fi, fj = i[mask], j[mask]
            fresh = np.isinf(self.first_seen[fi, fj])
            self.first_seen[fi[fresh], fj[fresh]] = t
        self._cache = None

    def stand(self, x, z, t):
        super().stand(x, z, t)
        i, j = self.cell(x, z)
        r = int(np.ceil(0.3 / self.res))
        self.seen[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1] = True
        self.looked[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1] = True

    # ------------------------------------------------------------ layers
    def _derived(self):
        if self._cache is None:
            s = np.full((self.n, self.n), UNKNOWN, dtype=np.uint8)
            s[self.free_hits > 0] = FREE
            s[self.occ_hits > 0] = OCCUPIED
            known = self.seen | (s != UNKNOWN)
            blocking = self.occ_hits > 0
            if getattr(self, 'collision', None) is not None:
                blocking = blocking | self.collision      # where the dog bumped; cleared per goal
            ground = (s == FREE) & ~ndimage.binary_dilation(blocking, structure=self._disk_ground)
            air_block = self.occ_air > 0
            if getattr(self, 'collision_air', None) is not None:
                air_block = air_block | self.collision_air   # what a drone's camera stopped it at
            air = known & ~ndimage.binary_dilation(air_block, structure=self._disk_air)
            self._cache = {'state': s, 'known': known, 'traversable': ground,
                           'ground': ground, 'air': air}
        return self._cache

    def known_fraction_bbox(self):
        known = self._derived()['known']
        ii, jj = np.nonzero(known)
        if ii.size == 0:
            return None
        return int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max())

    def known_area_m2(self):
        return float(np.count_nonzero(self._derived()['known'])) * self.res ** 2

    def distance_field(self, x, z, layer='ground'):
        der = self._derived()
        trav = der[layer]
        bb = self.known_fraction_bbox()
        si, sj = self.cell(x, z)
        if layer == 'air' and self.inside(si, sj) and not trav[si, sj]:
            # A drone outside the flyable layer escapes to its nearest flyable cell first
            # (Body._escape_cell): measure from there, or it never gets a frontier to escape for.
            r = 10
            a0, b0 = max(si - r, 0), max(sj - r, 0)
            ii, jj = np.nonzero(trav[a0:si + r + 1, b0:sj + r + 1])
            if ii.size:
                k = int(np.argmin((ii + a0 - si) ** 2 + (jj + b0 - sj) ** 2))
                si, sj = int(ii[k] + a0), int(jj[k] + b0)
        if bb is None:
            bb = (si, si, sj, sj)
        i0, i1 = max(min(bb[0], si) - 2, 0), min(max(bb[1], si) + 3, self.n)
        j0, j1 = max(min(bb[2], sj) - 2, 0), min(max(bb[3], sj) + 3, self.n)
        win = trav[i0:i1, j0:j1].copy()
        li, lj = si - i0, sj - j0
        r = 3
        # The ground robot may leave its own inflation halo through it. Drones may not: a path through
        # non-flyable halo cells met the drones' stride check (never fly out of the flyable layer)
        # and they re-planned in place; a drone outside the layer escapes first (Body._escape_cell).
        if layer != 'air':
            blk = (slice(max(li - r, 0), li + r + 1), slice(max(lj - r, 0), lj + r + 1))
            win[blk] |= (der['state'] == FREE)[i0:i1, j0:j1][blk]
        win[li, lj] = True
        h, w = win.shape
        idx = -np.ones((h, w), dtype=np.int64)
        nodes = np.flatnonzero(win.ravel())
        idx.ravel()[nodes] = np.arange(nodes.size)
        rows, cols, wts = [], [], []
        for di, dj, cost in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5)):
            c0, c1 = max(0, -dj), w - max(0, dj)
            sa = (slice(0, h - di), slice(c0, c1))
            sb = (slice(di, h), slice(c0 + dj, c1 + dj))
            both = win[sa] & win[sb]
            rows.append(idx[sa][both]); cols.append(idx[sb][both])
            wts.append(np.full(int(both.sum()), cost * self.res))
        rows = np.concatenate(rows); cols = np.concatenate(cols); wts = np.concatenate(wts)
        graph = csr_matrix((np.concatenate([wts, wts]),
                            (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                           shape=(nodes.size, nodes.size))
        dist_nodes, pred = dijkstra(graph, directed=False, indices=idx[li, lj], return_predecessors=True)
        dist = np.full((h, w), np.inf)
        dist.ravel()[nodes] = dist_nodes
        return {'dist': dist, 'pred': pred, 'nodes': nodes, 'idx': idx,
                'i0': i0, 'j0': j0, 'shape': (h, w), 'layer': layer}

    # ------------------------------------------------------------ frontiers
    def searched_area_m2(self):
        return float(np.count_nonzero(self.looked)) * self.res ** 2

    def frontiers(self, min_cells=SEARCH_MIN_CELLS):
        """Edge of searched space: looked-at cells next to cells never seen
        from close range (unknown, or known only from afar)."""
        der = self._derived()
        known = der['known']
        unsearched = ~self.looked
        touch = ndimage.binary_dilation(unsearched, structure=np.ones((3, 3), bool))
        either = der['ground'] | der['air']
        cand = known & self.looked & touch & ~self.visited & ~self.dead
        cand &= (der['state'] == FREE) | der['air']        # not wall faces
        cand &= ndimage.binary_dilation(either, structure=np.ones((5, 5), bool))
        labels, count = ndimage.label(cand, structure=np.ones((3, 3), bool))
        out = []
        for k in range(1, count + 1):
            ii, jj = np.nonzero(labels == k)
            if ii.size < min_cells:
                continue
            cx, cz = self.world(ii.mean(), jj.mean())
            out.append({'cells': (ii, jj), 'size': int(ii.size), 'centroid': (float(cx), float(cz))})
        return out

    def goal_cell_for(self, field, frontier):
        ii, jj = frontier['cells']
        d = self.dist_at(field, ii, jj)
        best = int(np.argmin(d))
        if np.isfinite(d[best]):
            return int(ii[best]), int(jj[best]), float(d[best])
        trav = self._derived()[field.get('layer', 'ground')]
        r = 4
        best = (None, None, np.inf)
        for ci, cj in zip(ii[::3], jj[::3]):
            i0, i1, j0, j1 = max(ci - r, 0), min(ci + r + 1, self.n), max(cj - r, 0), min(cj + r + 1, self.n)
            bi, bj = np.nonzero(trav[i0:i1, j0:j1])
            if bi.size == 0:
                continue
            dd = self.dist_at(field, bi + i0, bj + j0)
            k = int(np.argmin(dd))
            if dd[k] < best[2]:
                best = (int(bi[k] + i0), int(bj[k] + j0), float(dd[k]))
        return best
