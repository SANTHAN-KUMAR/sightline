"""Author the project's real tree foliage materials and assign them to the 5 tree meshes (editor, PIE OFF).

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_foliage_materials.py').read())"
    FOLIAGE_TAG='after'; exec(open(r'D:\\Sightline\\tools\\scene\\qa_foliage.py').read())
    uv run python tools/scene/check_foliage_materials.py after --compare before

Idempotent: it rebuilds both master graphs from scratch and re-derives every instance's textures from what the
slot currently resolves to (falling back to the glTF's own material->image mapping), so re-running it cannot
lose a texture.

THIS SCRIPT OWNS ONLY `/Game/Sightline/Vegetation/Materials/` AND THE 5 TREE MESHES' MATERIAL SLOTS.
It does not place, move or delete a single instance. The 5,508 HISM instances placed by `build_vegetation.py`
are verified against the layout and against `actors.json` crown clearance; nothing here touches them.

THE DEFECT THIS FIXES
---------------------
An independent audit found that all 4,342 trees rendered through the glTF importer's STOCK DEFAULT material:
every leaf slot was a material instance whose parent chain ran `MI_Default_Blend_DS` ->
`/InterchangeAssets/gltf/M_Default`, i.e. an asset inside an ENGINE PLUGIN. No project foliage material had
ever been authored, so:

  * the shading model was the plain default-lit one. A leaf is a THIN TRANSLUCENT SHEET: most of what you see
    of a crown from underneath, from inside, or against the sun is light that went THROUGH a leaf, not off it.
    Default-lit has no term for that at all, so a leaf whose normal faces away from the sun returns nothing and
    renders black. That is exactly the reported symptom -- "crown interiors and backlit crowns render
    near-black at survey altitude" -- and `MSM_TWO_SIDED_FOLIAGE` plus a subsurface colour is the specific fix.
  * the leaves were declared TRANSLUCENT (the glTF's `"alphaMode": "BLEND"`), which is wrong three times over:
    translucency is not rendered by Nanite (the mesh silently drops to its fallback, i.e. bald trees), it does
    not write the depth/stencil that Cosys-AirSim's instance segmentation is read from -- so it would corrupt
    the labels the whole auto-labelling pipeline depends on -- and 4,342 translucent crowns is unbounded
    overdraw.

BLEND MODE: MASKED. THE PREMISE THAT THERE IS NOTHING TO MASK IS WRONG, AND THE PIXELS SAY SO.
----------------------------------------------------------------------------------------------
`build_vegetation.py` forced every leaf material to OPAQUE, on the reasoning that the Poly Haven leaf textures
are JPEGs, JPEG cannot carry alpha, and the leaves are modelled geometry rather than alpha cards. Every step of
that is wrong here, and the renders in `_artifacts/foliage/` are the evidence:

  * `jacaranda_tree_leaves_diff_2k.jpg` is an ATLAS OF THREE COMPOUND FRONDS ON A PURE BLACK BACKGROUND --
    measured median luma 0.000, only 27.2 % of texels green-dominant. That is an alpha-card atlas. The leaf
    geometry is flat quads; the black is meant to be cut away.
  * the imported texture is named `<pid>_leaves_diff-<pid>_leaves_alpha`, i.e. the Interchange importer
    composed an ALPHA CHANNEL into it. An opacity source therefore does exist in the project.
  * and it is demonstrably live: in `fol_3_crown_frontlit_beauty.png` BLUE SKY SHOWS THROUGH THE CANOPY. If
    the blend mode really were opaque and the alpha really were ignored, those gaps would be filled with the
    atlas's black background instead. They are not.

So the material renders TRANSLUCENT no matter what the instance's `base_property_overrides` claims -- the
dump reads back `BLEND_OPAQUE` and the override does not reach the renderer. That is also what produces the
white haze smeared over every crown in both crown views: many alpha-blended card layers accumulating towards
the bright sky behind them. It is not lighting; the frontlit and backlit frames show the identical wash.

MASKED is therefore the correct answer, cutting on the texture's own alpha. It keeps the leaf silhouettes,
removes the unbounded overdraw, restores a surface that Nanite renders and that writes the depth/stencil
Cosys-AirSim's instance segmentation is read from, and it does not invent a mask from luminance.
(The ferns are out of scope here and are still on an engine-plugin default -- see the report.)

SUBSURFACE COLOUR: MEASURED-ISH, NOT PICKED BY EYE
--------------------------------------------------
`MSM_TWO_SIDED_FOLIAGE` uses the Subsurface Color input as the tint of light transmitted through the surface
from behind. Published leaf transmittance for a typical broadleaf runs roughly 0.05 at 660 nm (red), 0.13 at
550 nm (green) and 0.03 at 450 nm (blue) -- normalised to green, that is R 0.38 : G 1.0 : B 0.23. The default
`SubsurfaceTint` of (1.0, 2.6, 0.6) reproduces those ratios exactly (1.0/2.6 = 0.385, 0.6/2.6 = 0.231) while
scaling the whole thing up to a useful level. It multiplies the leaf's own base colour rather than replacing
it, so per-leaf variation in the scan survives into the transmission.

WIND: DELIBERATELY NONE.
------------------------
No World Position Offset is authored, and this is a decision, not an omission. The dataset is captured as
STILL FRAMES whose RGB and instance-segmentation masks are separate render passes. Any vertex animation
advances between those two passes, so the geometry that produced the RGB is not the geometry that produced the
mask, and every box and every mask silhouette goes stale by however far a leaf moved. A canopy that looks alive
in a flythrough is worth nothing next to labels that are wrong. If wind is ever wanted for a demo it belongs
behind a scalar parameter set to 0.0 for every capture run.
"""

import ctypes
import json
import os
import time

import unreal

REPO = r"D:\Sightline"
MATDIR = "/Game/Sightline/Vegetation/Materials"
LEAF_MASTER = f"{MATDIR}/M_Sightline_Foliage_Leaf"
BARK_MASTER = f"{MATDIR}/M_Sightline_Foliage_Bark"
OUT = REPO + r"\_artifacts\foliage"

# Leaf transmittance ratios, normalised to green (see the docstring): R 0.38 : G 1.0 : B 0.23.
#
# The first attempt used (1.0, 2.6, 0.6) at strength 1.0, which carries the same RATIOS but scaled up ~2.6x.
# It was measurably too strong and the render said so: subsurface is BaseColor * Tint * Strength, a lit leaf's
# base green is around 0.5, and 0.5 * 2.6 = 1.3 CLIPS the green channel to 1.0 everywhere. The backlit crown
# came out at luma p50 0.608 against the frontlit crown's 0.593 -- a canopy brighter with the sun behind it
# than with the sun on it, which is backwards -- and the whole crown read as an electric lime rather than the
# mid-green of docs/SCENE_REFERENCE.md Reference A.
#
# So the tint is now the measured ratios themselves and the level is set by one honest scalar: 0.55 puts a
# base-0.5 leaf at a green transmission of 0.275, well clear of clipping.
SUBSURFACE_TINT = (0.38, 1.0, 0.23, 1.0)
SUBSURFACE_STRENGTH = 0.55
# AO is used exactly as authored. The guess worth recording is the one that turned out to be WRONG: the baked
# AO in a photogrammetry leaf atlas was assumed to be dark enough to need dialling back, so this defaulted to
# 0.70. Measured on `jacaranda_tree_leaves_arm_2k.jpg` restricted to the 25.7 % of texels that are actually
# leaf (the rest is the black cut-out background), the AO channel is mean 0.930, p10 0.886, p50 0.937 -- very
# nearly white. There is nothing to compensate for, so compensating would be inventing a correction. It stays
# a parameter so the lighting lane can retune from a render, but its default is now 1.0.
LEAF_AO_STRENGTH = 1.00
BARK_AO_STRENGTH = 1.00
# Where the leaf cut-out falls. The atlas background is pure black with an alpha to match, so the alpha is
# close to binary and the exact threshold barely matters; 0.33 is the usual foliage value and keeps the thin
# leaflet tips that a higher clip would nibble away.
OPACITY_CLIP = 0.33

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
MST = unreal.MaterialSamplerType
S = unreal.SamplerSourceMode.SSM_WRAP_WORLD_GROUP_SETTINGS

if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: material recompiles and asset saves are unreliable while PIE runs")


class _MEMSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memline(tag):
    G = 1024.0 ** 3
    ms = _MEMSTATUSEX()
    ms.dwLength = ctypes.sizeof(ms)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    m = {"AvailablePhysical": ms.ullAvailPhys / G, "AvailableVirtual": ms.ullAvailPageFile / G,
         "CommitLimit": ms.ullTotalPageFile / G}
    print(f"  [mem] {tag:14s} AvailablePhysical {m['AvailablePhysical']:.2f} GiB  "
          f"AvailableVirtual {m['AvailableVirtual']:.2f} GiB  (commit limit {m['CommitLimit']:.2f} GiB)")
    return m


MEM0 = memline("before")
os.makedirs(OUT, exist_ok=True)

with open(REPO + r"\data\scene\vegetation.json") as fh:
    plan = json.load(fh)
CAT = plan["catalogue"]
TREE_PIDS = sorted({it["pid"] for it in plan["items"]})
print(f"build_foliage_materials: {len(TREE_PIDS)} tree meshes -> {MATDIR}")
print(f"  species: {TREE_PIDS}")


# =============================================================================================================
# 1. find the meshes, and find the textures each slot must keep using
# =============================================================================================================
def bbox_volume(sm):
    b = sm.get_bounding_box()
    return (b.max.x - b.min.x) * (b.max.y - b.min.y) * (b.max.z - b.min.z)


def resolve_mesh(pid):
    hint = CAT[pid].get("asset_hint")
    if hint and eal.does_asset_exist(hint):
        a = unreal.load_asset(hint)
        if isinstance(a, unreal.StaticMesh):
            return a
    folder = f"/Game/Sightline/Props/{pid}"
    found = [unreal.load_asset(p) for p in eal.list_assets(folder, recursive=True, include_folder=False)]
    found = [a for a in found if isinstance(a, unreal.StaticMesh)]
    if not found:
        raise RuntimeError(f"{pid}: no static mesh under {folder}")
    return max(found, key=bbox_volume)


def gltf_texture_names(pid):
    """material name -> {'D': basename, 'ARM': basename, 'N': basename} straight from the glTF that was
    imported. This is the authoritative mapping; the importer names each Texture2D after the image file."""
    path = CAT[pid].get("gltf")
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        g = json.load(fh)
    imgs = [i.get("uri") for i in g.get("images", [])]

    def base(d):
        if not d:
            return None
        src = g["textures"][d["index"]].get("source")
        if src is None or imgs[src] is None:
            return None
        return os.path.splitext(os.path.basename(imgs[src]))[0]

    out = {}
    for m in g.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        out[m["name"]] = {"D": base(pbr.get("baseColorTexture")),
                          "ARM": base(pbr.get("metallicRoughnessTexture")),
                          "N": base(m.get("normalTexture"))}
    return out


def classify(tex):
    """Which of the three maps a texture asset is, from its name.

    The Poly Haven source files are <material>_diff_2k / _arm_2k / _nor_gl_2k, but the Interchange importer
    RENAMES them on the way in, and not predictably: the packed AO/Roughness/Metallic map arrives as
    `<material>_rough` (it is named after its glTF metallicRoughness role, not its Poly Haven filename), and
    the leaf base colour arrives as `<material>_diff-<material>_alpha` because the importer composed an alpha
    channel into it. Both spellings have to be matched or the map is silently not found."""
    n = tex.get_name().lower()
    if "_diff" in n or n.endswith("_d"):
        return "D"
    if "_arm" in n or "_orm" in n or "_rough" in n or "_occlusion" in n:
        return "ARM"
    if "_nor" in n or "_normal" in n or n.endswith("_n"):
        return "N"
    return None


def ename(v):
    """The bare name of a UE enum value. `str(v)` is '<BlendMode.BLEND_OPAQUE: 0>', so splitting on '.' alone
    leaves 'BLEND_OPAQUE: 0>' -- which then never compares equal to anything and quietly fails every check."""
    n = getattr(v, "name", None)
    if n:
        return str(n)
    return str(v).strip("<>").split(":")[0].split(".")[-1]


def textures_for_slot(pid, mat, gmap, folder_index):
    """The three maps this slot must keep using. Tries what the slot currently resolves to first, then the
    glTF's own mapping. Refuses to guess: a missing map is a hard error naming what WAS found."""
    got = {}
    if isinstance(mat, unreal.MaterialInstanceConstant):
        try:
            for pv in mat.get_editor_property("texture_parameter_values"):
                v = pv.get_editor_property("parameter_value")
                if v is None:
                    continue
                k = classify(v)
                if k and k not in got:
                    got[k] = v
        except Exception as exc:                                        # noqa: BLE001 - editor API varies
            print(f"    ! {mat.get_name()}: could not read texture_parameter_values ({exc})")

    want = gmap.get(mat.get_name(), {})
    for k in ("D", "ARM", "N"):
        if k in got:
            continue
        base = want.get(k)
        if base and base in folder_index:
            got[k] = folder_index[base]
    missing = [k for k in ("D", "ARM", "N") if k not in got]
    if missing:
        raise RuntimeError(
            f"{pid}/{mat.get_name()}: no {missing} texture. glTF says {want}; the prop folder holds "
            f"{sorted(folder_index)[:40]}. Refusing to author a material with a map missing.")
    return got


MESHES, SLOTS = {}, []
for pid in TREE_PIDS:
    sm = resolve_mesh(pid)
    MESHES[pid] = sm
    gmap = gltf_texture_names(pid)
    index = {}
    for p in eal.list_assets(f"/Game/Sightline/Props/{pid}", recursive=True, include_folder=False):
        a = unreal.load_asset(p)
        if isinstance(a, unreal.Texture2D):
            index[a.get_name()] = a
    print(f"  {pid:16s} mesh={sm.get_name()} textures_in_folder={len(index)}")
    for i, slot in enumerate(sm.get_editor_property("static_materials")):
        mi = slot.get_editor_property("material_interface")
        if mi is None:
            raise RuntimeError(f"{pid}: material slot {i} is EMPTY")
        name = mi.get_name()
        role = "leaf" if ("leaves" in name.lower() or "leaf" in name.lower()) else "bark"
        tx = textures_for_slot(pid, mi, gmap, index)
        SLOTS.append({"pid": pid, "index": i, "src_name": name, "role": role, "tex": tx,
                      "slot_name": str(slot.get_editor_property("material_slot_name")),
                      "was": mi.get_path_name()})
        print(f"      [{i}] {role:4s} {name:32s} D={tx['D'].get_name()} ARM={tx['ARM'].get_name()} "
              f"N={tx['N'].get_name()}")

n_leaf = sum(1 for s in SLOTS if s["role"] == "leaf")
if n_leaf != len(TREE_PIDS):
    raise RuntimeError(f"expected exactly one leaf slot per tree ({len(TREE_PIDS)}), found {n_leaf}: "
                       f"{[s['src_name'] for s in SLOTS if s['role'] == 'leaf']}")
print(f"  {len(SLOTS)} slots total, {n_leaf} leaf")

# --- the mesh being edited must be the mesh the 5,508 placed instances actually render --------------------
# `resolve_mesh` follows the catalogue's asset_hint, which points at `<pid>_2k_lite`; the asset that is really
# in the project is `<pid>_2k`, so the hint misses and the folder scan decides. That is fine as long as it
# lands on the SAME mesh the HISM components hold -- otherwise this would author perfect materials onto an
# asset nothing in the level draws, and every render would be unchanged with every flag green.
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
hism_meshes = {}
for act in eas.get_all_level_actors():
    if str(act.get_folder_path()) != "Vegetation":
        continue
    h = act.get_component_by_class(unreal.HierarchicalInstancedStaticMeshComponent)
    if h is None:
        continue
    sm = h.get_editor_property("static_mesh")
    hism_meshes[act.get_actor_label()] = (sm.get_path_name() if sm else None, h.get_instance_count())
print("\nwhat the placed instances actually render:")
mismatch = []
for label, (path, n) in sorted(hism_meshes.items()):
    mine = next((p for p, m in MESHES.items() if m.get_path_name() == path), None)
    tag = f"<- this lane edits it as {mine}" if mine else "(understorey / not a tree: out of scope)"
    print(f"  {label:34s} {n:5d} instances  {path}  {tag}")
for pid, sm in MESHES.items():
    want = sm.get_path_name()
    hits = [lbl for lbl, (p, _n) in hism_meshes.items() if p == want]
    if not hits:
        mismatch.append(f"{pid}: nothing in the level renders {want}")
if mismatch:
    raise RuntimeError("THE MESHES RESOLVED ARE NOT THE MESHES BEING RENDERED - materials would be authored "
                       "onto assets nothing draws:\n  " + "\n  ".join(mismatch))
print(f"  all {len(MESHES)} tree meshes are confirmed in use by the placed canopy")

# --- sampler types, chosen from what the textures ACTUALLY are ------------------------------------------------
# A TextureSampleParameter2D whose sampler type disagrees with its texture makes the WHOLE material fail to
# compile, and a failed compile reports zeroed statistics and renders as the grey WorldGridMaterial checker
# with no error anywhere (docs/CONTEXT.md section 7; it has already happened once on this project). So the
# sampler type is DERIVED from the imported textures instead of assumed, and every texture that will ever be
# fed to a given parameter is required to agree.
def sampler_for(kind, texs):
    srgb = {bool(t.get_editor_property("srgb")) for t in texs}
    comp = {ename(t.get_editor_property("compression_settings")) for t in texs}
    if len(srgb) != 1:
        raise RuntimeError(f"{kind}: textures disagree on sRGB ({srgb}) - one master parameter cannot serve "
                           f"both. Textures: {[t.get_name() for t in texs]}")
    is_srgb = srgb.pop()
    if kind == "N":
        st = MST.SAMPLERTYPE_NORMAL if comp == {"TC_NORMALMAP"} else MST.SAMPLERTYPE_LINEAR_COLOR
    elif kind == "D":
        st = MST.SAMPLERTYPE_COLOR if is_srgb else MST.SAMPLERTYPE_LINEAR_COLOR
    else:
        st = MST.SAMPLERTYPE_MASKS if comp == {"TC_MASKS"} else (
            MST.SAMPLERTYPE_COLOR if is_srgb else MST.SAMPLERTYPE_LINEAR_COLOR)
    print(f"  sampler {kind:3s}: srgb={is_srgb} compression={sorted(comp)} -> {str(st).split('.')[-1]}")
    return st


SAMPLER, DEFAULT_TEX = {}, {}
for kind in ("D", "ARM", "N"):
    texs = [s["tex"][kind] for s in SLOTS]
    SAMPLER[kind] = sampler_for(kind, texs)
    DEFAULT_TEX[kind] = [s["tex"][kind] for s in SLOTS if s["role"] == "leaf"][0]

# The leaf cut-out depends entirely on the base colour texture's ALPHA channel. `compression_no_alpha` throws
# that channel away at build time, so a leaf texture carrying it would mask nothing and the crowns would fill
# with the atlas's black background -- a failure that is obvious in a render and invisible in every flag.
print("\nleaf base-colour alpha (the mask source):")
for s in SLOTS:
    if s["role"] != "leaf":
        continue
    t = s["tex"]["D"]
    no_alpha = bool(t.get_editor_property("compression_no_alpha"))
    print(f"  {s['pid']:16s} {t.get_name():58s} compression={ename(t.get_editor_property('compression_settings'))}"
          f" no_alpha={no_alpha}")
    if no_alpha:
        raise RuntimeError(
            f"{s['pid']}: {t.get_name()} has compression_no_alpha=True, so it carries no alpha channel and a "
            f"MASKED leaf material would cut nothing. Fix the texture's compression before running this.")


# =============================================================================================================
# 2. the two master materials
# =============================================================================================================
class G:
    """Tiny graph builder over MaterialEditingLibrary; same shape as build_roof_materials.py's."""

    def __init__(self, m):
        self.m, self.y = m, 0

    def node(self, cls, x=-1200, **props):
        self.y += 90
        e = mel.create_material_expression(self.m, cls, x, self.y)
        for k, v in props.items():
            e.set_editor_property(k, v)
        return e

    def link(self, a, a_out, b, b_in):
        if not mel.connect_material_expressions(a, a_out, b, b_in):
            raise RuntimeError(f"connect failed: {a.get_name()}.{a_out or 'out'} -> {b.get_name()}.{b_in}")

    def op(self, cls, a, b, a_out="", b_out="", x=-600):
        e = self.node(cls, x)
        for src, out, pin in ((a, a_out, "A"), (b, b_out, "B")):
            if isinstance(src, (int, float)):
                e.set_editor_property("const_" + pin.lower(), float(src))
            else:
                self.link(src, out, e, pin)
        return e

    def mul(self, a, b, a_out="", b_out=""):
        return self.op(unreal.MaterialExpressionMultiply, a, b, a_out, b_out)

    def lerp(self, a, b, alpha, a_out="", b_out="", alpha_out=""):
        e = self.node(unreal.MaterialExpressionLinearInterpolate, -300)
        for src, out, pin in ((a, a_out, "A"), (b, b_out, "B"), (alpha, alpha_out, "Alpha")):
            if isinstance(src, (int, float)):
                e.set_editor_property("const_" + ("alpha" if pin == "Alpha" else pin.lower()), float(src))
            else:
                self.link(src, out, e, pin)
        return e

    def sat(self, a, a_out=""):
        e = self.node(unreal.MaterialExpressionSaturate, -450)
        self.link(a, a_out, e, "")
        return e

    def mask(self, a, r=False, g=False, b=False, a_out=""):
        e = self.node(unreal.MaterialExpressionComponentMask, -450, r=r, g=g, b=b, a=False)
        self.link(a, a_out, e, "")
        return e

    def scalar(self, name, val, x=-1700):
        return self.node(unreal.MaterialExpressionScalarParameter, x, parameter_name=name, default_value=val)

    def vector(self, name, rgba, x=-1700):
        return self.node(unreal.MaterialExpressionVectorParameter, x, parameter_name=name,
                         default_value=unreal.LinearColor(*rgba))

    def tex(self, name, kind, x=-1400):
        return self.node(unreal.MaterialExpressionTextureSampleParameter2D, x, parameter_name=name,
                         texture=DEFAULT_TEX[kind], sampler_type=SAMPLER[kind], sampler_source=S)


def reuse_or_create(path, cls, factory):
    """Reuse an existing asset and clear its graph. create_asset over an asset a level actor references is
    refused, and deleting one is worse, so never delete: rebuild in place (build_materials.py does the same)."""
    d, n = path.rsplit("/", 1)
    if eal.does_asset_exist(path):
        a = unreal.load_asset(path)
        for e in list(mel.get_material_expressions(a)):
            mel.delete_material_expression(a, e)
        mel.delete_all_material_expressions(a)
        left = mel.get_num_material_expressions(a)
        if left:
            raise RuntimeError(f"{path}: {left} expressions survived clearing; the graph would accumulate junk")
        return a
    a = tools.create_asset(n, d, cls, factory)
    if a is None:
        raise RuntimeError(f"create_asset returned None for {path}")
    return a


def assert_compiles(mat, where):
    """A material that fails to compile reports ZERO instructions and nothing else, and renders as the grey
    WorldGridMaterial checker with no error in any log. Fail here, loudly, instead of in a capture."""
    s = mel.get_statistics(mat)
    if s.num_pixel_shader_instructions == 0:
        raise RuntimeError(f"{where}: {mat.get_name()} FAILED TO COMPILE (0 instructions). Check each "
                           f"sampler type against its texture's sRGB / compression settings.")
    if s.num_pixel_texture_samples == 0:
        print(f"  !! {where}: {mat.get_name()} compiles but samples NO texture - it would render flat")
    return s


def build_master(path, leaf):
    m = reuse_or_create(path, unreal.Material, unreal.MaterialFactoryNew())
    # Set the material-level properties BEFORE building the graph: MP_SUBSURFACE_COLOR and MP_OPACITY_MASK are
    # only active inputs once the shading model / blend mode make them so, and connecting to an inactive input
    # silently does nothing at all.
    m.set_editor_property("blend_mode",
                          unreal.BlendMode.BLEND_MASKED if leaf else unreal.BlendMode.BLEND_OPAQUE)
    m.set_editor_property("two_sided", True)
    m.set_editor_property("shading_model",
                          unreal.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE if leaf
                          else unreal.MaterialShadingModel.MSM_DEFAULT_LIT)
    if leaf:
        m.set_editor_property("opacity_mask_clip_value", OPACITY_CLIP)

    g = G(m)
    col = g.tex("BaseColor", "D")
    nrm = g.tex("Normal", "N")
    arm = g.tex("ARM", "ARM")

    tint = g.vector("Tint", (1.0, 1.0, 1.0, 1.0))
    mel.connect_material_property(g.mul(col, tint), "", unreal.MaterialProperty.MP_BASE_COLOR)
    mel.connect_material_property(nrm, "", unreal.MaterialProperty.MP_NORMAL)

    ao_str = g.scalar("AOStrength", LEAF_AO_STRENGTH if leaf else BARK_AO_STRENGTH)
    mel.connect_material_property(g.lerp(1.0, g.mask(arm, r=True), ao_str), "",
                                  unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
    mel.connect_material_property(g.sat(g.mul(g.mask(arm, g=True), g.scalar("RoughnessScale", 1.0))), "",
                                  unreal.MaterialProperty.MP_ROUGHNESS)
    # Leaves and bark are dielectrics. The ARM blue channel is nominally metallic and is 0 for these scans,
    # but wiring a constant is one fewer thing that can be wrong in an asset nobody re-checks.
    mel.connect_material_property(g.scalar("Metallic", 0.0), "", unreal.MaterialProperty.MP_METALLIC)

    if leaf:
        # The cut-out. The atlas is fronds on black and the importer put the cut-out in the texture's alpha,
        # so the mask is that alpha, scaled by a parameter so the clip can be tuned without a rebuild.
        mel.connect_material_property(g.sat(g.mul(col, g.scalar("OpacityBoost", 1.0), a_out="A")), "",
                                      unreal.MaterialProperty.MP_OPACITY_MASK)
        # Transmission: the leaf's own albedo, pushed towards the green ratios real leaves transmit.
        ss = g.sat(g.mul(g.mul(col, g.vector("SubsurfaceTint", SUBSURFACE_TINT)),
                         g.scalar("SubsurfaceStrength", SUBSURFACE_STRENGTH)))
        mel.connect_material_property(ss, "", unreal.MaterialProperty.MP_SUBSURFACE_COLOR)

    mel.recompile_material(m)
    eal.save_asset(m.get_path_name())
    st = assert_compiles(m, "master")

    # Read the flags back off the SAVED asset. set_editor_property returning without raising is not evidence.
    m2 = unreal.load_asset(m.get_path_name())
    got = (ename(m2.get_editor_property("blend_mode")),
           bool(m2.get_editor_property("two_sided")),
           ename(m2.get_editor_property("shading_model")))
    want = ("BLEND_MASKED" if leaf else "BLEND_OPAQUE", True,
            "MSM_TWO_SIDED_FOLIAGE" if leaf else "MSM_DEFAULT_LIT")
    if got != want:
        raise RuntimeError(f"{path}: flags did not stick. got {got}, want {want}")
    clip = float(m2.get_editor_property("opacity_mask_clip_value")) if leaf else None
    print(f"  {path.rsplit('/', 1)[-1]:28s} {got[0]} two_sided={got[1]} {got[2]}"
          + (f" clip={clip:.2f}" if leaf else "")
          + f"  {st.num_pixel_shader_instructions} instr, {st.num_pixel_texture_samples} samples, "
            f"{mel.get_num_material_expressions(m)} nodes")
    return m


print("\nmaster materials:")
leaf_master = build_master(LEAF_MASTER, leaf=True)
bark_master = build_master(BARK_MASTER, leaf=False)
memline("masters")


# =============================================================================================================
# 3. one instance per slot, then assign it to the mesh
# =============================================================================================================
def canonical(src_name):
    """The instance name for a slot, derived so that RE-RUNNING IS IDEMPOTENT.

    The first version of this was `f"MI_{s['src_name']}"`, where src_name is whatever material the slot holds
    right now. That is fine on the first run and wrong on every run after it: the second pass reads back
    `MI_jacaranda_tree_leaves`, prefixes it again, and creates `MI_MI_jacaranda_tree_leaves` -- a full
    duplicate set of 15 materials, with the previous set left orphaned in the folder. Strip any prefix this
    script itself added before adding it back."""
    n = src_name
    while n.startswith("MI_"):
        n = n[3:]
    if not n:
        raise RuntimeError(f"cannot derive a canonical name from {src_name!r}")
    return f"MI_{n}"


print("\ninstances and slot assignment:")
report = []
EXPECTED_NAMES = {canonical(s["src_name"]) for s in SLOTS}
for s in SLOTS:
    path = f"{MATDIR}/{canonical(s['src_name'])}"
    if eal.does_asset_exist(path):
        inst = unreal.load_asset(path)
    else:
        inst = tools.create_asset(canonical(s["src_name"]), MATDIR, unreal.MaterialInstanceConstant,
                                  unreal.MaterialInstanceConstantFactoryNew())
        if inst is None:
            raise RuntimeError(f"create_asset returned None for {path}")
    mel.set_material_instance_parent(inst, leaf_master if s["role"] == "leaf" else bark_master)
    for pname, kind in (("BaseColor", "D"), ("Normal", "N"), ("ARM", "ARM")):
        mel.set_material_instance_texture_parameter_value(inst, pname, s["tex"][kind])
    mel.update_material_instance(inst)
    eal.save_asset(inst.get_path_name())

    # the instance must resolve to the textures we asked for, by PATH - a lost override silently falls back to
    # the master's default, which is a real texture, so checking for None would not catch it
    for pname, kind in (("BaseColor", "D"), ("Normal", "N"), ("ARM", "ARM")):
        v = mel.get_material_instance_texture_parameter_value(inst, pname)
        if v is None or v.get_path_name() != s["tex"][kind].get_path_name():
            raise RuntimeError(f"{path}.{pname} resolves to {v and v.get_path_name()}, expected "
                               f"{s['tex'][kind].get_path_name()}")
    st = assert_compiles(inst, s["pid"])

    sm = MESHES[s["pid"]]
    sm.set_material(s["index"], inst)
    eal.save_asset(sm.get_path_name())
    s["now"] = inst.get_path_name()
    report.append({"pid": s["pid"], "index": s["index"], "role": s["role"], "was": s["was"],
                   "now": inst.get_path_name(), "instructions": int(st.num_pixel_shader_instructions),
                   "samples": int(st.num_pixel_texture_samples)})
    print(f"  {s['pid']:16s} [{s['index']}] {s['role']:4s} {inst.get_name():34s} "
          f"{st.num_pixel_shader_instructions:4d} instr {st.num_pixel_texture_samples:2d} samples")

# --- verify from DISK, not from the objects we just wrote ----------------------------------------------------
# Reload every mesh from its package and read the slot back. An assignment that did not serialise is exactly
# the class of silent failure this project keeps being bitten by.
print("\nverifying the slots from disk:")
bad = []
for pid in TREE_PIDS:
    p = MESHES[pid].get_path_name()
    eal.save_asset(p)
    sm = unreal.load_asset(p)
    for i, slot in enumerate(sm.get_editor_property("static_materials")):
        mi = slot.get_editor_property("material_interface")
        got = mi.get_path_name() if mi else None
        want = next(x["now"] for x in SLOTS if x["pid"] == pid and x["index"] == i)
        chain, cur = [], mi
        while cur is not None:
            chain.append(cur.get_path_name())
            cur = cur.get_editor_property("parent") if isinstance(cur, unreal.MaterialInstanceConstant) else None
        engine = [c for c in chain if c.startswith("/InterchangeAssets/") or c.startswith("/Engine/")]
        if got != want:
            bad.append(f"{pid}[{i}] is {got}, expected {want}")
        if engine:
            bad.append(f"{pid}[{i}] chain still reaches an engine asset: {engine}")
        print(f"  {pid:16s} [{i}] -> {got}")
        for c in chain[1:]:
            print(f"        parent -> {c}")
if bad:
    raise RuntimeError("SLOT ASSIGNMENT DID NOT STICK:\n  " + "\n  ".join(bad))

# --- sweep the folder ----------------------------------------------------------------------------------------
# Only the two masters and the 15 canonical instances belong here. Anything else is debris from an earlier run
# of this script (the MI_MI_* duplicate set the non-idempotent naming produced), and leaving it means the next
# person cannot tell which of two identically-named-ish materials the canopy actually uses. Deletion is done
# ONLY after every slot has been verified from disk to point at a canonical instance, so nothing referenced by
# the level can be removed.
KEEP = EXPECTED_NAMES | {LEAF_MASTER.rsplit("/", 1)[-1], BARK_MASTER.rsplit("/", 1)[-1]}
stray = []
for p in eal.list_assets(MATDIR, recursive=False, include_folder=False):
    name = p.rsplit("/", 1)[-1].split(".")[0]
    if name not in KEEP:
        stray.append((name, p))
if stray:
    print(f"\nsweeping {len(stray)} stray asset(s) left in {MATDIR} by an earlier run:")
    for name, p in stray:
        in_use = [x for x in SLOTS if x["now"].rsplit("/", 1)[-1].split(".")[0] == name]
        if in_use:
            raise RuntimeError(f"refusing to delete {p}: a slot still uses it")
        ok = eal.delete_asset(p)
        print(f"  {'deleted' if ok else 'COULD NOT DELETE'}  {name}")
        if not ok:
            print(f"    ! {p} is still referenced by something; it must be removed by hand")
else:
    print(f"\n{MATDIR} holds only the 2 masters and the {len(EXPECTED_NAMES)} canonical instances")

saved = les.save_current_level()
print(f"\nlevel saved: {saved}")

with open(os.path.join(OUT, "build_report.json"), "w") as fh:
    json.dump({"when": time.time(), "when_local": time.strftime("%Y-%m-%d %H:%M:%S"),
               "leaf_master": LEAF_MASTER, "bark_master": BARK_MASTER,
               "subsurface_tint": SUBSURFACE_TINT, "subsurface_strength": SUBSURFACE_STRENGTH,
               "leaf_ao_strength": LEAF_AO_STRENGTH, "wind": "none (see the module docstring)",
               "slots": report}, fh, indent=1)

MEM1 = memline("after")
print(f"\n{len(report)} slots on {len(TREE_PIDS)} tree meshes now use project materials under {MATDIR}")
print("NOW RUN: FOLIAGE_TAG='after'; exec(open(r'D:\\Sightline\\tools\\scene\\qa_foliage.py').read())")
print("THEN:    uv run python tools/scene/check_foliage_materials.py after --compare before")
