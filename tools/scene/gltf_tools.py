"""Inspect, render and thin the downloaded Poly Haven glTF models, without Unreal and without a 3D library.

    uv run python tools/scene/gltf_tools.py stats  <gltf> [...]      per-primitive tri/vertex counts + size in m
    uv run python tools/scene/gltf_tools.py render <gltf> -o out.png  software render of the actual file
    uv run python tools/scene/gltf_tools.py lite   <gltf> --target 90000   write <id>_2k_lite.gltf

Why this exists
---------------
* We cannot open the editor (the orchestrator owns it), so the only honest proof that a downloaded tree IS a
  tree is to render the file we downloaded. `render` is a surface-sampling rasteriser: it samples points over
  the triangles by area, projects them through a perspective camera, resolves occlusion with a painter's sort
  and shades them with the model's own base-colour texture. It is not a PBR renderer; it is enough to see
  whether the thing is a broadleaf tree, a shrub or a 4 KB error page.
* Poly Haven's geometry-nodes trees model every leaf as real geometry (1.5M-4M triangles) and ship LODs only
  inside the .blend. `lite` builds a usable game-resolution version by dropping whole leaves/twigs - the
  connected components of the leaf mesh - and scaling the survivors about their own centroid so the crown keeps
  its density and silhouette. Deleting whole leaves is what a foliage artist does by hand; running a quadric
  decimator over leaf geometry shreds it into spikes.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


# ---------------------------------------------------------------------------------------------- loading
class Gltf:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.j = json.loads(self.path.read_text(encoding="utf-8"))
        self.buffers = []
        for b in self.j.get("buffers", []):
            uri = b.get("uri", "")
            if uri.startswith("data:"):
                self.buffers.append(np.frombuffer(base64.b64decode(uri.split(",", 1)[1]), np.uint8))
            else:
                self.buffers.append(np.fromfile(self.path.parent / uri, np.uint8))
            if self.buffers[-1].nbytes < b["byteLength"]:
                raise RuntimeError(f"{self.path.name}: buffer {uri} is {self.buffers[-1].nbytes} B, "
                                   f"glTF declares {b['byteLength']} B - truncated download")

    def accessor(self, idx: int) -> np.ndarray:
        a = self.j["accessors"][idx]
        n, comp = a["count"], NC[a["type"]]
        dt = CT[a["componentType"]]
        if "bufferView" not in a:
            return np.zeros((n, comp), dt)
        bv = self.j["bufferViews"][a["bufferView"]]
        buf = self.buffers[bv["buffer"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride") or comp * np.dtype(dt).itemsize
        if stride == comp * np.dtype(dt).itemsize:
            raw = buf[off:off + n * stride].view(dt)
            return raw.reshape(n, comp)
        out = np.empty((n, comp), dt)
        for i in range(n):
            out[i] = buf[off + i * stride:off + i * stride + comp * np.dtype(dt).itemsize].view(dt)
        return out

    def primitives(self):
        """Yields (mesh_name, material_name, prim_dict) in scene order."""
        for m in self.j.get("meshes", []):
            for p in m.get("primitives", []):
                mat = self.j["materials"][p["material"]]["name"] if "material" in p else "-"
                yield m.get("name", "?"), mat, p

    def node_transforms(self) -> dict[int, tuple[str, np.ndarray, np.ndarray]]:
        """mesh index -> (node name, 3x3 rotation, translation). Poly Haven kits lay their parts out with
        node transforms; ignoring them stacks every part on the origin."""
        out: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}

        def walk(ni: int, R: np.ndarray, T: np.ndarray):
            n = self.j["nodes"][ni]
            t = np.asarray(n.get("translation", [0, 0, 0]), np.float32)
            if "rotation" in n:
                x, y, z, w = n["rotation"]
                r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float32)
            else:
                r = np.eye(3, dtype=np.float32)
            s = np.asarray(n.get("scale", [1, 1, 1]), np.float32)
            R2 = R @ (r * s)
            T2 = T + R @ t
            if "mesh" in n:
                out[n["mesh"]] = (n.get("name", "?"), R2, T2)
            for c in n.get("children", []):
                walk(c, R2, T2)

        for si in self.j["scenes"][self.j.get("scene", 0)]["nodes"]:
            walk(si, np.eye(3, dtype=np.float32), np.zeros(3, np.float32))
        return out

    def geom(self, p):
        pos = self.accessor(p["attributes"]["POSITION"]).astype(np.float32)
        idx = (self.accessor(p["indices"]).reshape(-1).astype(np.int64) if "indices" in p
               else np.arange(len(pos), dtype=np.int64))
        uv = (self.accessor(p["attributes"]["TEXCOORD_0"]).astype(np.float32)
              if "TEXCOORD_0" in p["attributes"] else np.zeros((len(pos), 2), np.float32))
        nrm = (self.accessor(p["attributes"]["NORMAL"]).astype(np.float32)
               if "NORMAL" in p["attributes"] else np.zeros((len(pos), 3), np.float32))
        return pos, idx.reshape(-1, 3), uv, nrm

    def base_color_image(self, p):
        if "material" not in p:
            return None
        mat = self.j["materials"][p["material"]]
        t = (mat.get("pbrMetallicRoughness") or {}).get("baseColorTexture")
        if not t:
            return None
        img = self.j["images"][self.j["textures"][t["index"]]["source"]]
        uri = img.get("uri")
        return self.path.parent / uri if uri else None


def stats(g: Gltf) -> list[dict]:
    out = []
    for name, mat, p in g.primitives():
        pos, tri, _uv, _n = g.geom(p)
        lo, hi = pos.min(0), pos.max(0)
        out.append(dict(mesh=name, material=mat, verts=len(pos), tris=len(tri),
                        size_m=[round(float(x), 3) for x in (hi - lo)]))
    return out


# ---------------------------------------------------------------------------------------------- render
def _texture(path: Path | None, uv: np.ndarray) -> np.ndarray:
    if path is None or not path.exists():
        return np.full((len(uv), 3), 0.55, np.float32)
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((512, 512))
    a = np.asarray(im, np.float32) / 255.0
    h, w = a.shape[:2]
    x = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int32)
    y = np.clip((1.0 - uv[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(np.int32)
    return a[y, x]


def render(g: Gltf, out: Path, px: int = 720, samples: int = 2_200_000, azim: float = 35.0,
           elev: float = 12.0, seed: int = 7) -> dict:
    """Surface-sample the real geometry, project, painter-sort, shade with the model's own base colour."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    P, N, C = [], [], []
    xf = g.node_transforms()
    prims = [(mi, p) for mi, m in enumerate(g.j.get("meshes", [])) for p in m.get("primitives", [])]
    areas = []
    cache = []
    for mi, p in prims:
        pos, tri, uv, nrm = g.geom(p)
        if mi in xf:
            _nm, R, T = xf[mi]
            pos = pos @ R.T + T
            nrm = nrm @ R.T
        e1 = pos[tri[:, 1]] - pos[tri[:, 0]]
        e2 = pos[tri[:, 2]] - pos[tri[:, 0]]
        ar = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        cache.append((pos, tri, uv, nrm, ar, g.base_color_image(p)))
        areas.append(ar.sum())
    tot_area = float(sum(areas)) or 1.0
    for (pos, tri, uv, nrm, ar, tex), a_sum in zip(cache, areas):
        k = int(samples * (a_sum / tot_area))
        if k < 1 or ar.sum() <= 0:
            continue
        w = ar / ar.sum()
        pick = rng.choice(len(tri), size=k, p=w)
        u = rng.random(k, dtype=np.float32)
        v = rng.random(k, dtype=np.float32)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        t = tri[pick]
        b0, b1, b2 = (1 - u - v)[:, None], u[:, None], v[:, None]
        P.append(pos[t[:, 0]] * b0 + pos[t[:, 1]] * b1 + pos[t[:, 2]] * b2)
        nn = nrm[t[:, 0]] * b0 + nrm[t[:, 1]] * b1 + nrm[t[:, 2]] * b2
        N.append(nn)
        C.append(_texture(tex, uv[t[:, 0]] * b0 + uv[t[:, 1]] * b1 + uv[t[:, 2]] * b2))
    if not P:
        raise RuntimeError("no geometry to render")
    P = np.concatenate(P).astype(np.float32)
    N = np.concatenate(N).astype(np.float32)
    C = np.concatenate(C).astype(np.float32)
    # glTF is Y-up; the scene (and Unreal) are Z-up. Convert so "up" in the picture is really up.
    P = P[:, [0, 2, 1]] * np.array([1, -1, 1], np.float32)
    N = N[:, [0, 2, 1]] * np.array([1, -1, 1], np.float32)

    lo, hi = P.min(0), P.max(0)
    ctr = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    a, e = np.radians(azim), np.radians(elev)
    eye = ctr + diag * 1.55 * np.array([np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)], np.float32)
    fwd = ctr - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0, 0, 1], np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = P - eye
    z = rel @ fwd
    ok = z > 1e-4
    rel, z, Nv, Cv = rel[ok], z[ok], N[ok], C[ok]
    f = px / (2 * np.tan(np.radians(32) / 2))
    sx = (rel @ right) / z * f + px / 2
    sy = -(rel @ up) / z * f + px / 2
    on = (sx >= 0) & (sx < px) & (sy >= 0) & (sy < px)
    sx, sy, z, Nv, Cv = sx[on].astype(np.int32), sy[on].astype(np.int32), z[on], Nv[on], Cv[on]

    ldir = np.array([0.45, -0.55, 0.70], np.float32)
    ldir /= np.linalg.norm(ldir)
    nl = np.linalg.norm(Nv, axis=1, keepdims=True)
    Nn = Nv / np.where(nl == 0, 1, nl)
    lam = np.abs(Nn @ ldir)                       # two-sided: leaves are single-sided cards
    shade = (0.35 + 0.75 * lam)[:, None]
    col = np.clip(Cv * shade, 0, 1)

    img = np.full((px, px, 3), 0.86, np.float32)
    img[:, :, 2] = 0.90
    order = np.argsort(-z)                        # painter's algorithm: far first
    yy, xx, cc = sy[order], sx[order], col[order]
    for dy in (0, 1):
        for dx in (0, 1):
            y2, x2 = np.clip(yy + dy, 0, px - 1), np.clip(xx + dx, 0, px - 1)
            img[y2, x2] = cc
    im = Image.fromarray((img * 255).astype(np.uint8))

    # scale bar: 1/4 of the model's height, drawn against the real bbox
    d = ImageDraw.Draw(im)
    hgt = float(hi[2] - lo[2])
    bar_m = max(0.1, round(hgt / 4, 1))
    bar_px = int(bar_m / (2 * z.mean() * np.tan(np.radians(32) / 2) / px))
    bar_px = int(np.clip(bar_px, 20, px - 40))
    d.rectangle([20, px - 34, 20 + bar_px, px - 28], fill=(20, 20, 20))
    d.text((20, px - 24), f"{bar_m} m   |   bbox {hi[0]-lo[0]:.2f} x {hi[1]-lo[1]:.2f} x {hgt:.2f} m",
           fill=(20, 20, 20))
    d.text((20, 12), f"{g.path.name}   {sum(s['tris'] for s in stats(g)):,} tris", fill=(20, 20, 20))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return dict(png=str(out), points=int(len(sx)), bbox_m=[round(float(x), 3) for x in (hi - lo)])


# ---------------------------------------------------------------------------------------------- lite
def components(tri: np.ndarray, nvert: int) -> np.ndarray:
    """Connected-component label per vertex (one label per leaf / twig / detached part)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e0 = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    e1 = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    g = coo_matrix((np.ones(len(e0), np.uint8), (e0, e1)), shape=(nvert, nvert))
    _n, lab = connected_components(g, directed=False)
    return lab


def _pad(b: bytearray) -> None:
    while len(b) % 4:
        b.append(0)


def cluster_decimate(pos, tri, uv, nrm, cell):
    """Grid vertex-clustering decimation: snap vertices onto a `cell`-metre lattice, merge, drop the
    triangles that collapse. The right tool for solid organic shapes (trunks, hulls) - it keeps the
    silhouette, and it does not leave the spikes a quadric collapse leaves in leaf geometry."""
    key = np.floor(pos / cell).astype(np.int64)
    _u, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n).astype(np.float32)[:, None]
    pos2 = np.zeros((n, 3), np.float32); np.add.at(pos2, inv, pos); pos2 /= cnt
    uv2 = np.zeros((n, 2), np.float32); np.add.at(uv2, inv, uv); uv2 /= cnt
    nrm2 = np.zeros((n, 3), np.float32); np.add.at(nrm2, inv, nrm); nrm2 /= cnt
    t = inv[tri]
    good = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    t = t[good]
    if len(t):
        t = np.unique(np.sort(t, axis=1), axis=0)
    used = np.unique(t) if len(t) else np.zeros(0, np.int64)
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    return pos2[used], remap[t], uv2[used], nrm2[used]


def _fit_cluster(pos, tri, uv, nrm, want):
    """Pick the lattice size that lands nearest to `want` triangles."""
    diag = float(np.linalg.norm(pos.max(0) - pos.min(0))) or 1.0
    cell = diag * float(np.sqrt(max(want, 1) / max(len(tri), 1))) * 0.5
    best = None
    for _ in range(12):
        r = cluster_decimate(pos, tri, uv, nrm, cell)
        got = len(r[1])
        if best is None or abs(got - want) < abs(best[1] - want):
            best = (r, got, cell)
        if 0.75 * want <= got <= 1.25 * want:
            break
        cell *= (got / want) ** 0.5 if got > 0 else 2.0
    return best


LEAFY = ("leaf", "leaves")
TWIGGY = ("branch", "twig")
FOLIAGE = LEAFY + TWIGGY


def lite(src: Path, target: int, drop_materials: tuple[str, ...] = (), seed: int = 3,
         solid_share: float = 0.22, leaf_share: float = 0.82, leaf_grow: float = 2.4) -> dict:
    """Write `<stem>_lite.gltf` next to the original (same `textures/` URIs), at about `target` triangles.

    Three rules, chosen from what the geometry measurably IS:
      * a primitive whose material matches `drop_materials` is removed outright (masts, sails, rigging);
      * FOLIAGE primitives (material contains leaf/leaves/branch) are thinned by deleting whole connected
        components - one component is one leaf (~21 tris) or one twig (~37 tris). The largest components
        are kept first so structural limbs survive, and the survivors are grown about their own centroid by
        1/sqrt(keep), capped at 1.9x, so the crown keeps its coverage instead of going see-through;
      * everything else (trunks, bark, hulls) is decimated by vertex clustering, which keeps the shape.
    `solid_share` is the fraction of the triangle budget reserved for the non-foliage parts.
    """
    g = Gltf(src)
    rng = np.random.default_rng(seed)
    prims = [(mi, mesh.get("name", "?"),
              g.j["materials"][p["material"]]["name"] if "material" in p else "-", p)
             for mi, mesh in enumerate(g.j.get("meshes", [])) for p in mesh.get("primitives", [])]
    dropped = sorted({mt for _mi, _nm, mt, _p in prims if any(d in mt.lower() for d in drop_materials)})
    keep = [t for t in prims if not any(d in t[2].lower() for d in drop_materials)]

    parsed = [t + g.geom(t[3]) for t in keep]
    kind = ["leaf" if any(f in t[2].lower() for f in LEAFY)
            else "twig" if any(f in t[2].lower() for f in TWIGGY) else "solid" for t in parsed]
    tris_of = {k: sum(len(t[5]) for t, kk in zip(parsed, kind) if kk == k) for k in ("leaf", "twig", "solid")}
    has_foliage = (tris_of["leaf"] + tris_of["twig"]) > 0
    sol_want = (min(tris_of["solid"], max(int(target * solid_share), 8_000)) if has_foliage
                else min(tris_of["solid"], target)) if tris_of["solid"] else 0
    fol_want = max(0, target - sol_want)
    # Spend the foliage budget on LEAVES, not on bare twigs: in a healthy crown the twigs are hidden under
    # the leaves, and a tree thinned evenly reads as a dead tree (verified by eye at target=90k on
    # island_tree_01 - _artifacts/env_assets/cmp_island_tree_01.png).
    leaf_want, twig_want = int(fol_want * leaf_share), fol_want - int(fol_want * leaf_share)
    frac = {"leaf": 1.0 if not tris_of["leaf"] else min(1.0, leaf_want / tris_of["leaf"]),
            "twig": 1.0 if not tris_of["twig"] else min(1.0, twig_want / tris_of["twig"]),
            "solid": 1.0}

    bin_bytes = bytearray()
    accessors, views, prim_out, report = [], [], [], []

    def add_acc(arr, ctype, atype, tgt, minmax=False):
        _pad(bin_bytes)
        off = len(bin_bytes)
        arr = np.ascontiguousarray(arr)
        bin_bytes.extend(arr.tobytes())
        views.append(dict(buffer=0, byteOffset=off, byteLength=int(arr.nbytes), target=tgt))
        a = dict(bufferView=len(views) - 1, componentType=ctype, count=int(len(arr)), type=atype)
        if minmax:
            a["min"] = [float(x) for x in arr.min(0)]
            a["max"] = [float(x) for x in arr.max(0)]
        accessors.append(a)
        return len(accessors) - 1

    for (mi, nm, mt, p, pos, tri, uv, nrm), kd in zip(parsed, kind):
        n0 = len(tri)
        how = "kept"
        fol_frac = frac[kd]
        if kd != "solid" and fol_frac < 1.0:
            lab = components(tri, len(pos))
            tri_lab = lab[tri[:, 0]]
            ids, sizes = np.unique(tri_lab, return_counts=True)
            order = np.argsort(-sizes)                       # structural limbs first
            keep_n = max(1, int(round(len(ids) * fol_frac)))
            big = max(1, int(0.10 * keep_n))                 # always keep the 10% biggest components
            rest = rng.permutation(order[big:])[:max(0, keep_n - big)]
            chosen = ids[np.concatenate([order[:big], rest]).astype(np.int64)]
            tmask = np.isin(tri_lab, chosen)
            tri2 = tri[tmask]
            used = np.unique(tri2)
            remap = np.full(len(pos), -1, np.int64)
            remap[used] = np.arange(len(used))
            pos2, uv2, nrm2, tri2 = pos[used].copy(), uv[used], nrm[used], remap[tri2]
            # grow the surviving LEAVES to hold the canopy's coverage; growing twigs would just make
            # fat sticks, so they keep their size
            s = float(np.clip(1.0 / np.sqrt(max(fol_frac, 1e-3)), 1.0, leaf_grow)) if kd == "leaf" else 1.0
            lab2 = lab[used]
            o = np.argsort(lab2, kind="stable")
            uniq, start = np.unique(lab2[o], return_index=True)
            sums = np.add.reduceat(pos2[o], start, axis=0)
            cnt = np.diff(np.append(start, len(o)))[:, None]
            ctr = sums / cnt
            ind = ctr[np.searchsorted(uniq, lab2)]
            pos2 = ind + (pos2 - ind) * s
            how = f"{kd}: components {len(chosen):,}/{len(ids):,} kept, survivors x{s:.2f}"
        elif kd == "solid" and sol_want < tris_of["solid"]:
            want = max(2_000, int(sol_want * n0 / max(tris_of["solid"], 1)))
            (pos2, tri2, uv2, nrm2), got, cell = _fit_cluster(pos, tri, uv, nrm, want)
            how = f"vertex-cluster {cell*100:.1f} cm"
        else:
            pos2, tri2, uv2, nrm2 = pos, tri, uv, nrm
        idx = tri2.reshape(-1).astype(np.uint16 if len(pos2) < 65536 else np.uint32)
        prim = dict(attributes=dict(
            POSITION=add_acc(pos2.astype(np.float32), 5126, "VEC3", 34962, minmax=True),
            NORMAL=add_acc(nrm2.astype(np.float32), 5126, "VEC3", 34962),
            TEXCOORD_0=add_acc(uv2.astype(np.float32), 5126, "VEC2", 34962)),
            indices=add_acc(idx.reshape(-1, 1), 5123 if idx.dtype == np.uint16 else 5125, "SCALAR", 34963))
        if "material" in p:
            prim["material"] = p["material"]
        prim_out.append((mi, nm, prim))
        report.append(dict(mesh=nm, material=mt, tris_in=n0, tris_out=int(len(tri2)), how=how))

    _pad(bin_bytes)
    stem = src.name.replace(".gltf", "")
    binname = f"{stem}_lite.bin"
    (src.parent / binname).write_bytes(bytes(bin_bytes))
    # keep the original mesh/node split so the Unreal importer still makes one asset per part
    by_mesh: dict[int, list] = {}
    for mi, nm, prim in prim_out:
        by_mesh.setdefault(mi, []).append(prim)
    xf = g.node_transforms()
    meshes, nodes = [], []
    for mi, plist in sorted(by_mesh.items()):
        name = xf[mi][0] if mi in xf else g.j["meshes"][mi].get("name", f"mesh_{mi}")
        meshes.append(dict(name=name, primitives=plist))
        node = dict(mesh=len(meshes) - 1, name=name)
        src_node = next((n for n in g.j["nodes"] if n.get("mesh") == mi), None)
        if src_node:
            for k in ("translation", "rotation", "scale"):
                if k in src_node:
                    node[k] = src_node[k]
        nodes.append(node)
    out = dict(asset=g.j["asset"], scene=0, scenes=[dict(name="Scene", nodes=list(range(len(nodes))))],
               nodes=nodes, meshes=meshes, materials=g.j["materials"], textures=g.j.get("textures", []),
               images=g.j.get("images", []), samplers=g.j.get("samplers", [{}]),
               accessors=accessors, bufferViews=views,
               buffers=[dict(byteLength=len(bin_bytes), uri=binname)])
    dst = src.parent / f"{stem}_lite.gltf"
    dst.write_text(json.dumps(out), encoding="utf-8")
    return dict(gltf=str(dst), tris_in=sum(r["tris_in"] for r in report),
                tris_out=sum(r["tris_out"] for r in report), dropped_materials=dropped,
                keep_fraction={k: round(v, 4) for k, v in frac.items()},
                bin_mb=round(len(bin_bytes) / 1e6, 2),
                primitives=report)


def sheet(paths: list[str], out: Path, cell: int = 420, samples: int = 700_000, cols: int = 5) -> dict:
    """Render every model and paste the results into one contact sheet, so they can all be looked at."""
    from PIL import Image
    out.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for p in paths:
        png = out.parent / (Path(p).stem + ".png")
        n = samples
        try:
            render(Gltf(p), png, px=cell, samples=n)
            tiles.append((png, Path(p).stem))
        except Exception as e:                                    # a broken file must be visible, not silent
            print(f"RENDER FAILED {p}: {e}")
    rows = (len(tiles) + cols - 1) // cols
    sh = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i, (png, _name) in enumerate(tiles):
        sh.paste(Image.open(png), ((i % cols) * cell, (i // cols) * cell))
    sh.save(out)
    return dict(sheet=str(out), tiles=len(tiles))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats"); s.add_argument("gltf", nargs="+")
    r = sub.add_parser("render"); r.add_argument("gltf"); r.add_argument("-o", "--out", required=True)
    r.add_argument("--azim", type=float, default=35.0); r.add_argument("--elev", type=float, default=12.0)
    r.add_argument("--samples", type=int, default=2_200_000); r.add_argument("--px", type=int, default=720)
    l = sub.add_parser("lite"); l.add_argument("gltf"); l.add_argument("--target", type=int, default=90_000)
    l.add_argument("--drop", nargs="*", default=[])
    sh = sub.add_parser("sheet"); sh.add_argument("gltf", nargs="+"); sh.add_argument("-o", "--out", required=True)
    sh.add_argument("--cell", type=int, default=420); sh.add_argument("--samples", type=int, default=700_000)
    a = ap.parse_args()
    if a.cmd == "sheet":
        print(json.dumps(sheet(a.gltf, Path(a.out), a.cell, a.samples), indent=1))
        return 0
    if a.cmd == "stats":
        for f in a.gltf:
            g = Gltf(f)
            st = stats(g)
            print(f"== {Path(f).name}: {sum(x['tris'] for x in st):,} tris, "
                  f"{sum(x['verts'] for x in st):,} verts, {len(st)} primitives")
            for x in st:
                print(f"   {x['material'][:38]:38s} {x['tris']:>9,} tris  {x['verts']:>9,} v  size {x['size_m']}")
    elif a.cmd == "render":
        print(json.dumps(render(Gltf(a.gltf), Path(a.out), px=a.px, samples=a.samples,
                                azim=a.azim, elev=a.elev), indent=1))
    else:
        print(json.dumps(lite(Path(a.gltf), a.target, tuple(d.lower() for d in a.drop)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
