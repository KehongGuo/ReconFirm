"""A map the team builds from its own depth frames, and nothing else.

Every earlier round planned on a topological graph sampled at start-up from the
complete navmesh plus the room annotations, which covered 87-92% of each
building before a single frame was taken. That is not exploration; it is touring
a map the robots were handed. This module is the replacement. A cell is known
only once some robot's depth camera has seen it, and route planning runs on the
known-free cells only. The simulator's navmesh is still used by the motion layer
for one thing -- collision, i.e. `try_step` refusing to walk through a wall -- and
never for planning, reachability, or goal selection.

Assumptions, stated rather than hidden: poses are known (no SLAM), the team
shares one frame, and the search stays on the starting floor. Stairs rise into
the obstacle band and so read as walls, which keeps planning single-storey.

Grid conventions: row index i runs along world +z, column j along world +x, and
cell (0, 0) is the corner at `self.origin` (x, z). Height is taken relative to
the floor the team started on.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

UNKNOWN, FREE, OCCUPIED = 0, 1, 2

# Height bands relative to the starting floor, metres.
FLOOR_BAND = 0.12          # |h| below this is walkable floor
OBSTACLE_TOP = 1.30        # anything between FLOOR_BAND and this blocks a robot
MAX_RANGE = 5.0            # depth beyond this is not trusted for mapping
ROBOT_RADIUS = 0.25        # inflation for planning


class ObservedMap:
    def __init__(self, center_xz, floor_y: float, size_m: float = 80.0, res: float = 0.1):
        self.res = float(res)
        self.n = int(round(size_m / res))
        self.origin = np.asarray(center_xz, dtype=np.float64) - size_m / 2.0
        self.floor_y = float(floor_y)
        self.free_hits = np.zeros((self.n, self.n), dtype=np.uint16)
        self.occ_hits = np.zeros((self.n, self.n), dtype=np.uint16)
        self.first_seen = np.full((self.n, self.n), np.inf, dtype=np.float32)
        self.visited = np.zeros((self.n, self.n), dtype=bool)
        # Frontier cells a robot has stood beside without the unknown behind them
        # ever resolving (mesh holes, drops below the floor band, glass). Left in
        # place they are exits nobody can use, and a robot sent to one "arrives"
        # and is sent straight back -- 151 times in one debug episode.
        self.dead = np.zeros((self.n, self.n), dtype=bool)
        self._cache = None
        r = int(np.ceil(ROBOT_RADIUS / self.res))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        self._disk = (xx ** 2 + yy ** 2) <= (ROBOT_RADIUS / self.res) ** 2 + 1e-6

    # ------------------------------------------------------------ geometry
    def cell(self, x, z):
        j = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(int)
        i = np.floor((np.asarray(z) - self.origin[1]) / self.res).astype(int)
        return i, j

    def world(self, i, j):
        return (self.origin[0] + (np.asarray(j) + 0.5) * self.res,
                self.origin[1] + (np.asarray(i) + 0.5) * self.res)

    def inside(self, i, j):
        return (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)

    # ------------------------------------------------------------ integration
    def integrate(self, depth, eye, rotation, hfov_deg: float, t: float, stride: int = 2):
        """Fold one depth frame in. Perfect simulator depth, so one hit counts."""
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
        occ = inside & (height >= FLOOR_BAND) & (height < OBSTACLE_TOP)

        # Clear along each ray up to its hit: the space a camera looked through
        # is empty even where the floor under it was hidden from view.
        ray_ok = inside & (height < OBSTACLE_TOP) & (height > -FLOOR_BAND)
        if np.any(ray_ok):
            sel = np.flatnonzero(ray_ok)[::4]
            ex, ez = np.asarray(eye)[0], np.asarray(eye)[2]
            px, pz = pts[sel, 0], pts[sel, 2]
            length = np.hypot(px - ex, pz - ez)
            steps = np.maximum(1, (length / (self.res * 0.5)).astype(int))
            k = int(steps.max())
            frac = np.linspace(0.0, 1.0, k + 1)[None, :]
            keep = frac <= (1.0 - 0.15 / np.maximum(length, 1e-3))[:, None]
            rx = ex + (px - ex)[:, None] * frac
            rz = ez + (pz - ez)[:, None] * frac
            ri, rj = self.cell(rx[keep], rz[keep])
            m = self.inside(ri, rj)
            self.free_hits[ri[m], rj[m]] = 1
            fresh = np.isinf(self.first_seen[ri[m], rj[m]])
            self.first_seen[ri[m][fresh], rj[m][fresh]] = t

        # Perfect simulator depth: one hit is enough, so mark rather than count.
        self.free_hits[i[floor], j[floor]] = 1
        self.occ_hits[i[occ], j[occ]] = 1
        for mask in (floor, occ):
            fi, fj = i[mask], j[mask]
            fresh = np.isinf(self.first_seen[fi, fj])
            self.first_seen[fi[fresh], fj[fresh]] = t
        self._cache = None

    def stand(self, x, z, t: float):
        """A robot is standing here, so its footprint is free -- proprioception."""
        i, j = self.cell(x, z)
        r = int(np.ceil(0.3 / self.res))
        i0, i1, j0, j1 = max(i - r, 0), min(i + r + 1, self.n), max(j - r, 0), min(j + r + 1, self.n)
        self.free_hits[i0:i1, j0:j1] = np.maximum(self.free_hits[i0:i1, j0:j1], 1)
        self.visited[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = True
        block = self.first_seen[i0:i1, j0:j1]
        block[np.isinf(block)] = t
        self._cache = None

    # ------------------------------------------------------------ derived state
    def state(self):
        s = np.full((self.n, self.n), UNKNOWN, dtype=np.uint8)
        s[self.free_hits > 0] = FREE
        s[self.occ_hits > 0] = OCCUPIED
        return s

    def _derived(self):
        if self._cache is None:
            s = self.state()
            occ = s == OCCUPIED
            inflated = ndimage.binary_dilation(occ, structure=self._disk)
            traversable = (s == FREE) & ~inflated
            self._cache = {'state': s, 'traversable': traversable}
        return self._cache

    def known_fraction_bbox(self):
        s = self._derived()['state']
        ii, jj = np.nonzero(s != UNKNOWN)
        if ii.size == 0:
            return None
        return int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max())

    def known_area_m2(self) -> float:
        return float(np.count_nonzero(self._derived()['state'] == FREE)) * self.res ** 2

    # ------------------------------------------------------------ planning
    def distance_field(self, x, z):
        """Geodesic distance (m) over known-free, inflated space from (x, z).

        Returns (dist, predecessors, window) where window = (i0, j0, shape). The
        start cell is always allowed, so a robot hugging a wall can still leave.
        """
        der = self._derived()
        trav = der['traversable'].copy()
        bb = self.known_fraction_bbox()
        si, sj = self.cell(x, z)
        if bb is None:
            bb = (si, si, sj, sj)
        i0, i1 = max(min(bb[0], si) - 2, 0), min(max(bb[1], si) + 3, self.n)
        j0, j1 = max(min(bb[2], sj) - 2, 0), min(max(bb[3], sj) + 3, self.n)
        win = trav[i0:i1, j0:j1].copy()
        li, lj = si - i0, sj - j0
        r = 3
        # Let the robot out of its own inflation halo: cells it stands in.
        win[max(li - r, 0):li + r + 1, max(lj - r, 0):lj + r + 1] |= (
            der['state'][i0:i1, j0:j1][max(li - r, 0):li + r + 1, max(lj - r, 0):lj + r + 1] == FREE)
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
        src = idx[li, lj]
        dist_nodes, pred_nodes = dijkstra(graph, directed=False, indices=src,
                                          return_predecessors=True)
        dist = np.full((h, w), np.inf)
        dist.ravel()[nodes] = dist_nodes
        return {'dist': dist, 'pred': pred_nodes, 'nodes': nodes, 'idx': idx,
                'i0': i0, 'j0': j0, 'shape': (h, w)}

    def path_to(self, field, i, j):
        """Cells from the source to (i, j), as world (x, z) points; [] if unreachable."""
        li, lj = i - field['i0'], j - field['j0']
        h, w = field['shape']
        if not (0 <= li < h and 0 <= lj < w) or not np.isfinite(field['dist'][li, lj]):
            return []
        node = field['idx'][li, lj]
        out = []
        while node >= 0:
            flat = field['nodes'][node]
            ci, cj = divmod(int(flat), w)
            out.append((ci + field['i0'], cj + field['j0']))
            node = field['pred'][node]
            if node == -9999:
                break
        out.reverse()
        return [self.world(ci, cj) for ci, cj in out]

    def dist_at(self, field, i, j) -> float:
        li, lj = np.asarray(i) - field['i0'], np.asarray(j) - field['j0']
        h, w = field['shape']
        ok = (li >= 0) & (li < h) & (lj >= 0) & (lj < w)
        out = np.full(np.shape(li), np.inf)
        out[ok] = field['dist'][li[ok], lj[ok]]
        return out

    # ------------------------------------------------------------ frontiers
    def frontiers(self, min_cells: int = 6):
        """Clusters of known-free cells that touch unknown space.

        Each cluster is an exit the team has not looked through yet. Returned
        with its cells so that per-robot costs can be taken as the cheapest cell.
        """
        der = self._derived()
        s, trav = der['state'], der['traversable']
        unknown = s == UNKNOWN
        touch = ndimage.binary_dilation(unknown, structure=np.ones((3, 3), bool))
        cand = (s == FREE) & touch & ~self.visited & ~self.dead
        # A frontier cell must be somewhere a robot can stand near.
        near_trav = ndimage.binary_dilation(trav, structure=np.ones((5, 5), bool))
        cand &= near_trav
        labels, count = ndimage.label(cand, structure=np.ones((3, 3), bool))
        out = []
        for k in range(1, count + 1):
            ii, jj = np.nonzero(labels == k)
            if ii.size < min_cells:
                continue
            cx, cz = self.world(ii.mean(), jj.mean())
            out.append({'cells': (ii, jj), 'size': int(ii.size),
                        'centroid': (float(cx), float(cz))})
        return out

    def exhaust(self, x, z, radius=0.6):         # 1.5 m erased corridor mouths (HM3D 0.6-0.9 m corridors)
        """Retire every frontier cell within `radius` of a robot that reached it."""
        i, j = self.cell(x, z)
        r = int(np.ceil(radius / self.res))
        i0, i1, j0, j1 = max(i - r, 0), min(i + r + 1, self.n), max(j - r, 0), min(j + r + 1, self.n)
        yy, xx = np.mgrid[i0:i1, j0:j1]
        disk = (yy - i) ** 2 + (xx - j) ** 2 <= r * r
        self.dead[i0:i1, j0:j1] |= disk
        self._cache = None

    def goal_cell_for(self, field, frontier):
        """The cheapest traversable cell within reach of a frontier cluster."""
        ii, jj = frontier['cells']
        d = self.dist_at(field, ii, jj)
        best = int(np.argmin(d))
        if np.isfinite(d[best]):
            return int(ii[best]), int(jj[best]), float(d[best])
        # Frontier cells often sit inside the inflation halo; look one ring out.
        trav = self._derived()['traversable']
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
