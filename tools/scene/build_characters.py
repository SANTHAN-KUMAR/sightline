"""Import the Rocketbox character textures and bind them to the imported materials (running editor, PIE off).

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_characters.py').read())"

Why this is needed: the FBX import created one `FBXLegacyPhongSurfaceMaterial` instance per material slot but
bound **no textures at all** - the instances compile cleanly (418 instructions, 7 samples) and report no error,
yet every character renders as a plain white mannequin. The textures ship beside the FBX as TGAs
(`<prefix>_body_color.tga`, `_normal`, `_specular`, and `<prefix>_opacity_color.tga`) and are wired here.

`DiffuseColorMapWeight` matters as much as the texture: the parent multiplies the map by that weight, so a bound
texture with weight 0 still renders flat. Every weight this script sets is asserted afterwards.
"""

import os

import unreal

REPO = r"D:\Sightline"
SRC = REPO + r"\_downloads\assets\rocketbox"
DEST_ROOT = "/Game/Sightline/Characters/Rocketbox"

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: texture imports come back as 32 px placeholders while PIE runs")

MAPS = {"color": ("DiffuseColorMap", "DiffuseColorMapWeight", "color"),
        "normal": ("NormalMap", "NormalMapWeight", "normal"),
        "specular": ("SpecularColorMap", "SpecularColorMapWeight", "color")}


def import_tex(path, dest, name, kind):
    existing = unreal.load_asset(f"{dest}/{name}")
    if existing is None:
        t = unreal.AssetImportTask()
        for k, v in (("filename", path), ("destination_path", dest), ("destination_name", name),
                     ("automated", True), ("replace_existing", True), ("save", False)):
            t.set_editor_property(k, v)
        tools.import_asset_tasks([t])
        existing = unreal.load_asset(f"{dest}/{name}")
    if existing is None:
        return None
    if kind == "normal":
        existing.set_editor_property("srgb", False)
        existing.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
    eal.save_asset(existing.get_path_name())
    return existing


def character_dirs():
    for grp in ("Adults", "Children"):
        d = os.path.join(SRC, grp)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "Textures")):
                yield name, os.path.join(p, "Textures")


total_bound, checked = 0, []
for char, texdir in character_dirs():
    dest = f"{DEST_ROOT}/{char}"
    if not eal.does_directory_exist(dest):
        print(f"  {char}: not imported into the project, skipped")
        continue
    files = os.listdir(texdir)
    # material slots are named <prefix>_<part>, e.g. m006_body; textures are <prefix>_<part>_<map>.tga
    for f in sorted(files):
        stem, ext = os.path.splitext(f)
        if ext.lower() != ".tga":
            continue
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in MAPS:
            continue
        slot, mapkind = parts
        mat = unreal.load_asset(f"{dest}/{slot}")
        if mat is None or not isinstance(mat, unreal.MaterialInstanceConstant):
            continue
        param, weight, kind = MAPS[mapkind]
        tex = import_tex(os.path.join(texdir, f), dest, stem, kind)
        if tex is None:
            print(f"  {char}: FAILED to import {f}")
            continue
        mel.set_material_instance_texture_parameter_value(mat, param, tex)
        mel.set_material_instance_scalar_parameter_value(mat, weight, 1.0)
        if slot.endswith("_opacity"):
            # hair/eyelash cards: the colour map's alpha is the mask
            mel.set_material_instance_texture_parameter_value(mat, "OpacityMaskMap", tex)
            mel.set_material_instance_scalar_parameter_value(mat, "OpacityMaskMapWeight", 1.0)
        mel.update_material_instance(mat)
        eal.save_asset(mat.get_path_name())
        total_bound += 1
        checked.append((char, slot, param, tex.get_name()))

# --- verify: a bound texture with weight 0 still renders flat, so check both ------------------------------
bad = []
for char, slot, param, texname in checked:
    mat = unreal.load_asset(f"{DEST_ROOT}/{char}/{slot}")
    got = mel.get_material_instance_texture_parameter_value(mat, param)
    if got is None or got.get_name() != texname:
        bad.append(f"{char}/{slot}.{param} = {got}")
    w = MAPS[[k for k, v in MAPS.items() if v[0] == param][0]][1]
    if abs(mel.get_material_instance_scalar_parameter_value(mat, w) - 1.0) > 1e-6:
        bad.append(f"{char}/{slot}.{w} != 1.0")
if bad:
    raise RuntimeError(f"{len(bad)} texture bindings did not stick, first: {bad[:5]}")

print(f"bound and verified {total_bound} texture parameters across "
      f"{len({c for c, _, _, _ in checked})} characters")
print("NOW LOOK AT A RENDER: run tools/scene/qa_shots.py and open the images.")
