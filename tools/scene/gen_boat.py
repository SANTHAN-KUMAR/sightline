"""Generate a Kerala country boat (vallam) as a glTF, because no free-without-login photoreal small boat exists.

    uv run python tools/scene/gen_boat.py
    uv run python tools/scene/gltf_tools.py render _downloads/assets/generated/country_boat_01/country_boat_01.gltf -o out.png

Why generate one
----------------
The reference photo has small boats moving between the houses. Poly Haven's only vessel is `ship_pinnace`, a
three-masted square-rigged ship 39.8 m long and 32.6 m tall (measured: _artifacts/env_assets/sheet_utility_water.png);
scaling that down to 7 m would put a 17th-century warship in a Kerala village. ambientCG ships materials, not
models. So the boat is built the same way the 73 houses were built - procedurally, to real dimensions - and
skinned with the weathered-plank texture set already downloaded (Poly Haven, CC0).

Dimensions come from the small inland vallam used on the Kerala backwaters: about 7.5 m long, 1.55 m beam,
0.62 m depth amidships, fine raised ends, three thwarts. Open hull with a visible interior, so it reads as a
boat from directly above - which is the only view the drone ever has.
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "_downloads" / "assets" / "generated" / "country_boat_01"
TEX_SRC = REPO / "_downloads" / "assets" / "polyhaven" / "weathered_planks"   # flat: texture-only asset

L, B, D = 7.5, 1.55, 0.62          # length, beam, depth amidships (m)
SHEER, ROCKER, THICK = 0.34, 0.16, 0.035
NS, NT = 33, 9                      # stations along the length, points from keel to gunwale
TILE = 2.0                          # metres per texture repeat


def section(s: float):
    """Half-section at fore-aft fraction s in [0,1]: (half-beam, keel z, gunwale z)."""
    b = (B / 2) * float(np.sin(np.pi * s)) ** 0.55
    k = (2 * s - 1) ** 2
    return max(b, 0.012), ROCKER * k, D + SHEER * k


def hull_grid(inset: float) -> np.ndarray:
    """(NS, 2*NT, 3) grid of hull points, port and starboard, pulled `inset` metres inward."""
    pts = np.zeros((NS, 2 * NT, 3), np.float32)
    for i in range(NS):
        s = i / (NS - 1)
        b, zk, zg = section(s)
        for j in range(NT):
            t = j / (NT - 1)
            y = b * float(np.sin(t * np.pi / 2)) ** 0.9
            z = zk + (zg - zk) * t ** 1.55
            # inward normal in the section plane, roughly
            ny, nz = -float(np.sin(t * np.pi / 2)), -float(np.cos(t * np.pi / 2))
            n = np.hypot(ny, nz) or 1.0
            y += inset * ny / n
            z += inset * nz / n
            pts[i, NT - 1 - j] = (s * L - L / 2, -max(y, 0.004), z)     # port
            pts[i, NT + j] = (s * L - L / 2, max(y, 0.004), z)          # starboard
    return pts


def grid_tris(a: int, ns: int, nc: int, flip: bool) -> np.ndarray:
    t = []
    for i in range(ns - 1):
        for j in range(nc - 1):
            p0, p1 = a + i * nc + j, a + i * nc + j + 1
            p2, p3 = a + (i + 1) * nc + j, a + (i + 1) * nc + j + 1
            t += [(p0, p2, p1), (p1, p2, p3)] if not flip else [(p0, p1, p2), (p1, p3, p2)]
    return np.array(t, np.int64)


def build() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer, inner = hull_grid(0.0), hull_grid(THICK)
    P = [outer.reshape(-1, 3), inner.reshape(-1, 3)]
    n_grid = NS * 2 * NT
    T = [grid_tris(0, NS, 2 * NT, False), grid_tris(n_grid, NS, 2 * NT, True)]
    # gunwale rim: close outer and inner along both sheer edges
    rim = []
    for col_o, col_i, flip in ((0, 0, True), (2 * NT - 1, 2 * NT - 1, False)):
        for i in range(NS - 1):
            o0, o1 = i * 2 * NT + col_o, (i + 1) * 2 * NT + col_o
            i0, i1 = n_grid + i * 2 * NT + col_i, n_grid + (i + 1) * 2 * NT + col_i
            rim += [(o0, i0, o1), (o1, i0, i1)] if flip else [(o0, o1, i0), (o1, i1, i0)]
    T.append(np.array(rim, np.int64))
    # three thwarts (seats) across the hull
    base = 2 * n_grid
    for s in (0.28, 0.5, 0.72):
        b, _zk, zg = section(s)
        x = s * L - L / 2
        w, th = 0.16, 0.035
        c = np.array([[x - w / 2, -b, zg - 0.10], [x + w / 2, -b, zg - 0.10],
                      [x + w / 2, b, zg - 0.10], [x - w / 2, b, zg - 0.10]], np.float32)
        box = np.vstack([c, c + np.array([0, 0, th], np.float32)])
        P.append(box)
        q = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
             (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
        T.append(np.array(q, np.int64) + base)
        base += 8
    pos = np.vstack(P).astype(np.float32)
    tri = np.vstack(T).astype(np.int64)
    # UVs: planks run fore-and-aft (u along the length, v around the girth)
    uv = np.zeros((len(pos), 2), np.float32)
    # the weathered_planks texture runs its planks along V, so girth->U and length->V puts the planks
    # fore-and-aft, the way a boat is actually built (checked by eye in _artifacts/env_assets/country_boat_01.png)
    uv[:, 0] = (np.abs(pos[:, 1]) * 0.6 + pos[:, 2]) / TILE
    uv[:, 1] = pos[:, 0] / TILE
    # area-weighted vertex normals
    e1 = pos[tri[:, 1]] - pos[tri[:, 0]]
    e2 = pos[tri[:, 2]] - pos[tri[:, 0]]
    fn = np.cross(e1, e2)
    nrm = np.zeros_like(pos)
    for k in range(3):
        np.add.at(nrm, tri[:, k], fn)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(ln == 0, 1, ln)
    return pos, tri, uv, nrm


def write_gltf(pos, tri, uv, nrm) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "textures").mkdir(exist_ok=True)
    tex = {}
    for src_kind, kind, name in (("diff", "diff", "baseColor"), ("nor_dx", "nor_gl", "normal"),
                                 ("arm", "arm", "arm")):
        src = TEX_SRC / f"weathered_planks_{src_kind}_2k.jpg"
        if not src.exists():
            raise SystemExit(f"missing source texture {src}: the weathered_planks texture set is required")
        dst = OUT / "textures" / f"country_boat_01_{kind}_2k.jpg"
        if not dst.exists():
            if src_kind == "nor_dx":
                # Poly Haven ships this texture set DirectX-convention only; glTF wants OpenGL, so flip green
                from PIL import Image
                im = Image.open(src).convert("RGB")
                r, gch, b = im.split()
                from PIL import ImageChops
                Image.merge("RGB", (r, ImageChops.invert(gch), b)).save(dst, quality=95)
            else:
                shutil.copyfile(src, dst)
        tex[name] = f"textures/{dst.name}"

    # glTF is Y-up; the model is built Z-up, so swap here (x, y, z) -> (x, z, -y)
    pos = pos[:, [0, 2, 1]] * np.array([1, 1, -1], np.float32)
    nrm = nrm[:, [0, 2, 1]] * np.array([1, 1, -1], np.float32)
    tri = tri[:, ::-1].copy()                       # winding follows the handedness flip

    blob = bytearray()
    views, accs = [], []

    def acc(arr, ctype, atype, target, minmax=False):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        arr = np.ascontiguousarray(arr)
        blob.extend(arr.tobytes())
        views.append(dict(buffer=0, byteOffset=off, byteLength=int(arr.nbytes), target=target))
        a = dict(bufferView=len(views) - 1, componentType=ctype, count=int(len(arr)), type=atype)
        if minmax:
            a["min"] = [float(x) for x in arr.min(0)]
            a["max"] = [float(x) for x in arr.max(0)]
        accs.append(a)
        return len(accs) - 1

    prim = dict(attributes=dict(POSITION=acc(pos.astype(np.float32), 5126, "VEC3", 34962, True),
                                NORMAL=acc(nrm.astype(np.float32), 5126, "VEC3", 34962),
                                TEXCOORD_0=acc(uv.astype(np.float32), 5126, "VEC2", 34962)),
                indices=acc(tri.reshape(-1, 1).astype(np.uint16), 5123, "SCALAR", 34963), material=0)
    (OUT / "country_boat_01.bin").write_bytes(bytes(blob))
    doc = dict(
        asset=dict(generator="Sightline tools/scene/gen_boat.py", version="2.0"),
        scene=0, scenes=[dict(name="Scene", nodes=[0])],
        nodes=[dict(mesh=0, name="country_boat_01")],
        meshes=[dict(name="country_boat_01", primitives=[prim])],
        materials=[dict(name="country_boat_01", doubleSided=True,
                        normalTexture=dict(index=1),
                        pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0,
                                                  metallicRoughnessTexture=dict(index=2)))],
        textures=[dict(sampler=0, source=0), dict(sampler=0, source=1), dict(sampler=0, source=2)],
        images=[dict(mimeType="image/jpeg", name="country_boat_01_diff", uri=tex["baseColor"]),
                dict(mimeType="image/jpeg", name="country_boat_01_nor_gl", uri=tex["normal"]),
                dict(mimeType="image/jpeg", name="country_boat_01_arm", uri=tex["arm"])],
        samplers=[dict(magFilter=9729, minFilter=9987)],
        accessors=accs, bufferViews=views,
        buffers=[dict(byteLength=len(blob), uri="country_boat_01.bin")])
    (OUT / "country_boat_01.gltf").write_text(json.dumps(doc), encoding="utf-8")
    size = pos.max(0) - pos.min(0)
    return dict(gltf=str(OUT / "country_boat_01.gltf"), verts=int(len(pos)), tris=int(len(tri)),
                size_m_ue=[round(float(size[0]), 3), round(float(size[2]), 3), round(float(size[1]), 3)])


if __name__ == "__main__":
    pos, tri, uv, nrm = build()
    r = write_gltf(pos, tri, uv, nrm)
    # a boat that is not boat-shaped must fail here, not in the scene
    x, y, z = r["size_m_ue"]
    assert 7.0 <= x <= 8.0, f"length {x} m is not a vallam"
    assert 1.3 <= y <= 1.8, f"beam {y} m is not a vallam"
    assert 0.7 <= z <= 1.2, f"depth over sheer {z} m is not a vallam"
    print(json.dumps(r, indent=1))
