"""
Mesh decimation — reduce a reference mesh's triangle/vertex count in place.

The heavy CSO weapons carry ~20k triangles, which forces a model into many
submodels (studiomdl budgets 2048 vertices each) and bloats the file.  There is
no lossless win — the geometry is not padded — so this trades detail for size,
the way Blender's Decimate modifier does.

These decompiled meshes are *triangle soup*: heavily UV-seamed and non-manifold
(a third of the edges are open borders), which stops edge-collapse cold — almost
every vertex sits on a seam it may not cross.  So this uses **quadric-error
vertex clustering** (Rossignac–Borrel with Garland-Heckbert representatives),
which ignores connectivity: vertices are grouped by a spatial grid and each group
is replaced by the single point that best fits the surfaces that met there.  It
is robust on messy input and its reduction is predictable.

Safety for this pipeline:

*   **Bone bindings are never mixed.**  A grid cell is split by bone, so a
    cluster only ever merges vertices bound to the same bone and the
    representative keeps it — every vertex still rides one bone and the
    animation steps are unaffected.
*   **Texture UVs ride along per corner**, so a triangle keeps its mapping; the
    only texture movement is the (bounded) shift of a corner to its
    representative, never a jump across a UV seam.
*   **Normals are recomputed** from the decimated surface, so shading stays
    correct without having to preserve smoothing groups.

The achieved counts are returned so the caller can report them — a ratio the
grid cannot quite hit (very small meshes, or a target below the seam skeleton)
comes back as whatever was actually reached.
"""

from __future__ import annotations

import math

import numpy as np

from goldsource.smd import SMD, Triangle, Vertex


_POS_ROUND = 5          # decimals used to weld triangle corners into vertices


def decimate_mesh(smd: SMD, ratio: float) -> tuple[int, int]:
    """
    Reduce *smd* toward ``ratio`` × its current vertex count, in place.

    Returns ``(triangles_before, triangles_after)``.  A ratio ≥ 1 (or a mesh too
    small to touch) leaves it unchanged.
    """
    before = len(smd.triangles)
    if ratio >= 1.0 or before < 4:
        return before, before

    verts, faces = _build(smd)
    if len(verts) < 8:
        return before, before

    target = max(4, int(round(len(verts) * ratio)))
    rep_of, rep_pos = _cluster(verts, target)
    smd.triangles = _emit(verts, faces, rep_of, rep_pos)
    return before, len(smd.triangles)


# ---------------------------------------------------------------------------
# Build: weld corners into (position, bone) vertices and accumulate quadrics
# ---------------------------------------------------------------------------

class _V:
    __slots__ = ("pos", "bone_id", "quadric")

    def __init__(self, pos, bone_id):
        self.pos = pos
        self.bone_id = bone_id
        self.quadric = np.zeros((4, 4))


def _build(smd: SMD):
    key_to_id: dict[tuple, int] = {}
    verts: list[_V] = []

    def vid(v: Vertex) -> int:
        pos = (round(v.x, _POS_ROUND), round(v.y, _POS_ROUND), round(v.z, _POS_ROUND))
        key = (pos, v.bone_id)
        got = key_to_id.get(key)
        if got is None:
            got = len(verts)
            key_to_id[key] = got
            verts.append(_V(np.array([v.x, v.y, v.z], dtype=float), v.bone_id))
        return got

    # A face stores its three vertex ids plus the original corners, so UV and
    # material ride through untouched.
    faces: list[tuple[list[int], Triangle]] = []
    for tri in smd.triangles:
        ids = [vid(tri.v0), vid(tri.v1), vid(tri.v2)]
        if len({tuple(verts[i].pos) for i in ids}) < 3:
            continue
        faces.append((ids, tri))
        plane = _plane(*(verts[i].pos for i in ids))
        if plane is not None:
            K = np.outer(plane, plane)
            for i in ids:
                verts[i].quadric += K

    return verts, faces


def _plane(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    length = np.linalg.norm(n)
    if length < 1e-12:
        return None
    n = n / length
    return np.array([n[0], n[1], n[2], -float(n @ p0)])


# ---------------------------------------------------------------------------
# Cluster: pick a grid that yields ~target cells, one representative per cell
# ---------------------------------------------------------------------------

def _cluster(verts: list[_V], target: int):
    pos = np.array([v.pos for v in verts])
    bones = np.array([v.bone_id for v in verts])
    lo, hi = pos.min(0), pos.max(0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0

    def cells_for(h: float):
        grid = np.floor((pos - lo) / h).astype(np.int64)
        return grid

    def count(h: float) -> int:
        grid = cells_for(h)
        seen = set(zip(grid[:, 0].tolist(), grid[:, 1].tolist(),
                       grid[:, 2].tolist(), bones.tolist()))
        return len(seen)

    # Binary-search the cell size: smaller cells → more clusters.  Aim a touch
    # above target, never below, so we do not over-decimate.
    small, large = diag / 4096, diag
    for _ in range(24):
        mid = math.sqrt(small * large)
        if count(mid) > target:
            small = mid
        else:
            large = mid
    h = large

    grid = cells_for(h)
    cell_id: dict[tuple, int] = {}
    rep_of = np.empty(len(verts), dtype=np.int64)
    members: list[list[int]] = []
    for i in range(len(verts)):
        key = (int(grid[i, 0]), int(grid[i, 1]), int(grid[i, 2]), int(bones[i]))
        c = cell_id.get(key)
        if c is None:
            c = len(members)
            cell_id[key] = c
            members.append([])
        rep_of[i] = c
        members[c].append(i)

    rep_pos = [_representative([verts[i] for i in group]) for group in members]
    return rep_of, rep_pos


def _representative(group: list[_V]) -> np.ndarray:
    """
    The point minimising summed quadric error, clamped to the cluster.

    On flat or ill-conditioned clusters the quadric optimum can land far outside
    the cell — which shows up as spikes — so a solved point that strays past the
    cluster's own extent is rejected in favour of the centroid.
    """
    positions = np.array([v.pos for v in group])
    centroid = positions.mean(0)
    if len(group) == 1:
        return centroid

    Q = np.zeros((4, 4))
    for v in group:
        Q += v.quadric
    A = Q[:3, :3]
    b = -Q[:3, 3]
    try:
        if abs(np.linalg.det(A)) > 1e-9:
            solved = np.linalg.solve(A, b)
            # Accept only if it stays within the cluster's own bounds (plus a
            # little slack); otherwise it is a runaway optimum.
            span = positions.max(0) - positions.min(0)
            slack = 0.25 * span + 1e-6
            if np.all(solved >= positions.min(0) - slack) and \
               np.all(solved <= positions.max(0) + slack):
                return solved
    except np.linalg.LinAlgError:
        pass
    return centroid


# ---------------------------------------------------------------------------
# Emit: rebuild triangles onto representatives, recompute smooth normals
# ---------------------------------------------------------------------------

def _emit(verts, faces, rep_of, rep_pos) -> list[Triangle]:
    kept: list[tuple[list[int], Triangle]] = []
    for ids, tri in faces:
        reps = [int(rep_of[i]) for i in ids]
        if len(set(reps)) < 3:
            continue  # collapsed to a line or point
        pts = [rep_pos[r] for r in reps]
        if len({tuple(np.round(p, 6)) for p in pts}) < 3:
            continue
        kept.append((reps, tri))

    # Smooth vertex normals from the decimated surface (area-weighted).
    normal = {}
    for reps, _tri in kept:
        p = [rep_pos[r] for r in reps]
        fn = np.cross(p[1] - p[0], p[2] - p[0])
        for r in reps:
            normal[r] = normal.get(r, np.zeros(3)) + fn

    def unit(r, fallback):
        n = normal.get(r)
        if n is None:
            return fallback
        length = np.linalg.norm(n)
        return n / length if length > 1e-9 else fallback

    out: list[Triangle] = []
    for reps, tri in kept:
        corners_in = (tri.v0, tri.v1, tri.v2)
        new = []
        for slot, r in enumerate(reps):
            src = corners_in[slot]
            p = rep_pos[r]
            n = unit(r, np.array([src.nx, src.ny, src.nz]))
            new.append(Vertex(
                bone_id=src.bone_id,
                x=float(p[0]), y=float(p[1]), z=float(p[2]),
                nx=float(n[0]), ny=float(n[1]), nz=float(n[2]),
                u=src.u, v=src.v,
            ))
        out.append(Triangle(material=tri.material, v0=new[0], v1=new[1], v2=new[2]))
    return out
