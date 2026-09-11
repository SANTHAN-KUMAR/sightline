"""F2 + F5: fly a real boustrophedon coverage pattern and capture the dataset from the actual flight.

    uv run python -m sightline.mission.survey --alt 45 --speed 10 --out _artifacts/dataset/flight_alt45

Why this replaces waypoint teleporting. The first capture runner moved the drone with `simSetVehiclePose` /
`simSetKinematics` at 731 waypoints. That is using a physics-simulated multirotor as a camera dolly, and every
failure it produced came from fighting the solver rather than from the scene:

    * teleporting a rigid body each frame drove it to 2.2e8 rad/s and a 17 m/s descent;
    * the survey camera sits 30 cm BELOW the body, so the tumbling airframe swung into shot and filled frames
      with propellers - it silently ruined a 731-frame dataset that passed every programmatic check;
    * a pose set while paused is ignored; set while running it lands one call late;
    * even after zeroing the motion, each teleport impulse rolled the body up to 25.4 deg.

Flying the pattern removes all of it at once, and it is what SOLUTION_DOC F2 asks for anyway. The telemetry
then describes a real trajectory - real attitude, real velocity, real motion between frames - which is what
sections 5.4 and 5.7 assume and what the tracker's camera-motion compensation is designed for.

The drone holds constant AGL over the higher of ground or flood surface: the valley walls reach 1169.9 m ASL
while the flood surface is 1061.7, so a fixed height above the water flies into the hillside.

The plan, the terrain follower, the cadence gate and the AirSim capture helper live in
`sightline/mission/pattern.py` and are SHARED with `sightline/mission/live.py` (the real-time demo loop).
They used to be local closures in `main()` below, which meant a second runner had to copy them.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "scene"))
from sightline.mission.pattern import (MIN_HITS, MIN_HITS_WINDOW_S, Scenario, body_euler_deg,  # noqa: E402
                                       build_plan, cadence_verdict, connect, grab)
from tools.capture.labels import apply_depth_visibility, attach_truth, labels_from_mask, to_yolo  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import cosysairsim as airsim  # noqa: E402  (weather / drivetrain / yaw enums only)

__all__ = ["MIN_HITS", "MIN_HITS_WINDOW_S", "main"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alt", type=float, default=45.0, help="metres AGL")
    ap.add_argument("--speed", type=float, default=10.0, help="m/s ground speed")
    ap.add_argument("--out", default="_artifacts/dataset/flight")
    ap.add_argument("--quantile", type=float, default=0.05,
                    help="box plan only: trim the survivor bounding box to a sane flight length")
    ap.add_argument("--plan", choices=("box", "patches"), default="patches",
                    help="box: one lawnmower over the whole survivor extent. patches: one small "
                         "lawnmower over each cluster of survivors, skipping the empty valley between "
                         "them. The box plan spends most of its frames on ground with nobody on it: "
                         "measured on the 2026-09-11 run, 209 frames returned 42 boxes.")
    ap.add_argument("--time-of-day", default="",
                    help="YYYY-MM-DD HH:MM:SS to drive the sun, or empty to leave the level alone. "
                         "Sun angle changes the image far more than another 300 frames of the same "
                         "light does, and it costs no extra flying.")
    ap.add_argument("--rain", type=float, default=None, help="0..1")
    ap.add_argument("--fog", type=float, default=None, help="0..1")
    ap.add_argument("--dust", type=float, default=None, help="0..1")
    ap.add_argument("--condition", default="clear_midday",
                    help="name for this lighting/weather condition; it is written to the data card "
                         "and is a slice axis, so every metric can be reported per condition")
    ap.add_argument("--no-track-check", action="store_true",
                    help="fly even if the cadence cannot confirm a track (detector-only dataset)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the flight plan and exit without arming or flying")
    ap.add_argument("--patch-link-m", type=float, default=90.0,
                    help="patches plan: survivors closer than this are flown as one patch")
    ap.add_argument("--jpeg", type=int, default=0, metavar="Q",
                    help="save frames as JPEG at this quality instead of PNG (masks stay lossless "
                         "PNG - a lossy mask would corrupt every label). 4K PNG is 11.9 MB a frame.")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--shutter-m", type=float, default=26.0, help="capture every N metres of track")
    ap.add_argument("--thermal", action="store_true", help="also capture the Infrared pass")
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--alt-tol-m", type=float, default=8.0,
                    help="do not capture while more than this far off the commanded AGL")
    ap.add_argument("--max-tilt-deg", type=float, default=8.0,
                    help="do not capture while the airframe is banked beyond this")
    a = ap.parse_args()

    # Scenario, terrain follower, plan and cadence gate all come from the SHARED module: `live.py` flies the
    # identical plan, so tuning either runner tunes both (sightline/mission/pattern.py).
    scn = Scenario.load()
    home, water = scn.home, scn.water_level_m
    truth, cal, f_px = scn.truth, scn.cal, scn.f_px
    surface_asl = scn.terrain.surface_asl
    det = scn.detectable_actors()

    sp = build_plan(scn, alt_m=a.alt, speed_ms=a.speed, shutter_m=a.shutter_m, plan=a.plan,
                    overlap=a.overlap, quantile=a.quantile, patch_link_m=a.patch_link_m)
    legs, inside, W, Hm = sp.legs, sp.survivors_in_plan, sp.frame_w_m, sp.frame_h_m
    print(sp.summary())

    # --- will this cadence produce a TRACK? -------------------------------------------------------------
    # SOLUTION_DOC 5.6 rule 3: a track is emitted only after `min_hits=3` inside `min_hits_window_s=2.0`.
    # A survey that cannot deliver that yields detections and geolocations and then nothing at all.
    verdict = cadence_verdict(a.alt, a.speed, a.shutter_m, scn.hfov_deg)
    if not a.no_track_check and not verdict.ok:
        print(verdict.message())
        return 2

    if a.dry_run:
        est = sp.est_minutes + sp.est_frames * 0.6 / 60
        print(f"dry run: {sp.est_frames} frames, ~{est:.1f} min including "
              f"capture overhead. Nothing was armed or flown.")
        return 0

    out = REPO / a.out
    for d in ("images", "labels", "masks") + (("ir",) if a.thermal else ()):
        (out / d).mkdir(parents=True, exist_ok=True)

    n_hidden = 0        # survivors present in the mask but fully hidden in RGB by foliage/rubble
    c = connect()
    names = c.simListInstanceSegmentationObjects()
    cmap = c.simGetSegmentationColorMap()
    grab(c, want_ir=a.thermal, want_depth=True)             # warm-up frame, discarded (unconverted buffers)

    # --- PRE-FLIGHT: is the instance segmentation healthy? -------------------------------------------------
    # A capture is only as good as its instance colours. `tools/capture/thermal_ids.py` assigns a SHARED
    # segmentation id per temperature (331 objects got one id), which collapses distinct instances onto one
    # colour: the flood plane then renders in a survivor's colour, and 78 % of the resulting labels are
    # physically impossible - a "person" spanning 67.8 x 38.1 m of ground. Nothing downstream noticed until a
    # human looked at a frame. Refuse to fly rather than produce that again.
    _pal: dict[tuple, list[str]] = {}
    for _i, _n in enumerate(names):
        _pal.setdefault(tuple(int(v) for v in cmap[_i]), []).append(_n)
    _dupes = {c_: ns for c_, ns in _pal.items()
              if len(ns) > 1 and any(x.startswith(("Human_", "Animal_")) for x in ns)}
    if _dupes:
        print("\nFATAL: instance segmentation is degenerate - actors share a colour with other objects:")
        for c_, ns in list(_dupes.items())[:5]:
            print(f"   colour {c_} used by {len(ns)} objects: {ns[:4]}")
        print("\n   Restart PIE to regenerate unique instance ids (InitialInstanceSegmentation in settings),")
        print("   and never run tools/capture/thermal_ids.py on a PIE session you intend to capture from.")
        raise SystemExit(2)

    # --- record the palette WITH the dataset ---------------------------------------------------------------
    # `tools/capture/validate.py` check C is the only one that confirms a labelled box actually contains its
    # actor in the mask, and it needs actor_id -> colour. Reading that from a live simulator meant it was
    # skipped every time anyone validated a dataset with the editor shut, which is almost always - and on
    # 2026-09-11 the tool then printed "all checks passed - dataset is clean" and exited 0 anyway. Writing
    # the palette next to the frames makes the dataset self-describing: the check runs for ever, on any
    # machine, with nothing running.
    from tools.capture.labels import actor_index as _actor_index, palette_rgb as _palette_rgb  # noqa: PLC0415
    _pal_rgb = _palette_rgb(cmap)
    _idx = _actor_index(names)
    _pal_out = {str(aid): [int(v) for v in _pal_rgb[i]] for i, (aid, _nm, _cls) in _idx.items()}
    (out / "segmentation_palette.json").write_text(json.dumps({
        "by": "sightline/mission/survey.py",
        "note": "actor_id -> RGB in the instance mask, as written by the capture. RGB, not BGR: the raw "
                "AirSim buffer is RGB and cv2.imread returns BGR, so a reader must reverse the channels.",
        "objects_registered": len(names), "actors": len(_pal_out),
        "actor_rgb": _pal_out,
        # The full map, not just the survivors. Without it the dataset is self-describing for people and
        # mute about everything else, so nothing downstream can read a mask for what a pixel IS - terrain,
        # water, a roof - and has to re-derive it geometrically from the scene JSONs instead. It costs ~40 KB.
        # data/scene/thermal_table.json is NOT a substitute: it holds 1,162 objects against this capture's
        # 1,252, so the indices do not line up.
        # CAVEAT, and it is the whole reason the depth gate exists: this map lists what the annotator
        # REGISTERED, not what it RENDERS. Cosys-AirSim disables the InstancedFoliage/InstancedGrass show
        # flags in its annotation pass, so every HISM plant here has an entry below and never appears in a
        # single mask pixel. A reader must not infer "absent from the mask" means "absent from the scene".
        "object_rgb": {nm: [int(v) for v in _pal_rgb[i]] for i, nm in enumerate(names)},
    }, indent=1), encoding="utf-8")
    print(f"  palette recorded: {len(_pal_out)} actors, {len(names)} objects "
          f"-> {out / 'segmentation_palette.json'}")
    print(f"pre-flight: {len(names)} instances, every actor colour unique")

    if a.time_of_day or a.rain is not None or a.fog is not None or a.dust is not None:
        if a.time_of_day:
            c.simSetTimeOfDay(True, a.time_of_day, False, 1.0, 60.0, True)
        if a.rain is not None or a.fog is not None or a.dust is not None:
            c.simEnableWeather(True)
            for p_, v_ in ((airsim.WeatherParameter.Rain, a.rain),
                           (airsim.WeatherParameter.Fog, a.fog),
                           (airsim.WeatherParameter.Dust, a.dust)):
                if v_ is not None:
                    c.simSetWeatherParameter(p_, float(v_))
        # Look at the result rather than trusting the call: a level with no sky sphere accepts
        # simSetTimeOfDay and changes nothing, and that would silently label every frame with a
        # condition it does not have.
        _probe = grab(c)["scene"]
        _tod = a.time_of_day or "level default"
        print(f"condition {a.condition!r}: time_of_day={_tod} rain={a.rain} fog={a.fog} "
              f"dust={a.dust}; frame mean brightness {float(_probe.mean()):.1f}/255")
    c.enableApiControl(True)
    c.armDisarm(True)
    print("\ntaking off...")
    c.takeoffAsync(timeout_sec=20).join()

    clip_id = f"seed{truth['seed']}_alt{int(a.alt)}_{a.condition}"
    tf = (out / "telemetry.csv").open("w", newline="", encoding="utf-8")
    tw = csv.writer(tf)
    tw.writerow(["frame_idx", "t_utc", "clip_id", "east_m", "north_m", "alt_msl_m", "agl_m", "lat", "lon",
                 "q_w", "q_x", "q_y", "q_z", "gimbal_pitch_deg", "hfov_deg", "width_px", "height_px",
                 "gsd_cm_px", "mode", "flood_level_asl_m", "n_labels", "speed_ms"])

    k = 0
    total = 0
    agls: list[float] = []
    seen: set[int] = set()
    t_start = time.time()
    last_shot = None
    try:
        for li, leg in enumerate(legs):
            if (time.time() - t_start) / 60.0 > a.max_minutes:
                print(f"  stopping: {a.max_minutes} min budget reached at line {li}/{len(legs)}")
                break
            # Fly the WHOLE leg in one command and shoot on the move. Stopping at every shutter point made the
            # drone accelerate and brake 22 times per line (5 min for two lines) and removed the very motion the
            # tracker's camera-motion compensation exists to handle. A continuous leg is faster AND a truer pass.
            e, na, nb = leg.east_m, leg.north_start_m, leg.north_end_m
            asl_a = surface_asl(e, na) + a.alt
            c.moveToPositionAsync(float(na - home["north_m"]), float(e - home["east_m"]),
                                  float(-(asl_a - home["ground_asl_m"])), a.speed, timeout_sec=120).join()
            asl_b = surface_asl(e, nb) + a.alt
            fut = c.moveToPositionAsync(float(nb - home["north_m"]), float(e - home["east_m"]),
                                        float(-(asl_b - home["ground_asl_m"])), a.speed, timeout_sec=400,
                                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                                        yaw_mode=airsim.YawMode(False, 0))
            print(f"  line {li + 1}/{len(legs)} east {e:7.1f}  north {na:.0f} -> {nb:.0f}")
            shots, last_n, t_leg = 0, na, time.time()
            while True:
                st = c.simGetGroundTruthKinematics()
                cur_n = home["north_m"] + st.position.x_val
                cur_e = home["east_m"] + st.position.y_val
                reached = (cur_n >= nb - 2.0) if nb > na else (cur_n <= nb + 2.0)
                # A real survey does not release the shutter mid-bank, and the validator flags any frame
                # captured beyond +-12 deg. The drone banks on acceleration and at leg ends, so gate on the
                # body attitude: wait for it to level rather than recording a tilted frame.
                _o = st.orientation
                _roll, _pitch, _ = body_euler_deg(_o)
                _agl = (home["ground_asl_m"] - st.position.z_val) - surface_asl(cur_e, cur_n)
                # and do not shoot when the drone is far off its commanded height: an audit found AGL ranging
                # 18.7-121.4 m in a run labelled "45 m", which corrupts GSD, scale and geolocation alike.
                on_alt = abs(_agl - a.alt) <= a.alt_tol_m
                level = max(abs(_roll), abs(_pitch)) <= a.max_tilt_deg and on_alt
                if level and (shots == 0 or abs(cur_n - last_n) >= a.shutter_m):
                    g = grab(c, want_ir=a.thermal, want_depth=True)
                    gp = c.getMultirotorState().gps_location
                    o, v = st.orientation, st.linear_velocity
                    spd = math.sqrt(v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2)
                    asl = home["ground_asl_m"] - st.position.z_val
                    # The instance mask is blind to foliage (Cosys-AirSim disables the InstancedFoliage show
                    # flag in its annotation render), so it reports survivors under a canopy as fully
                    # visible. The depth buffer is not blind to it. Gate every box on depth before writing.
                    labs, hidden = apply_depth_visibility(
                        attach_truth(labels_from_mask(g["seg"], names, cmap), truth),
                        g["seg"], g["depth"], cam_alt_asl_m=asl, actors_json=truth)
                    n_hidden += len(hidden)
                    stem = f"{clip_id}_{k:05d}"
                    if a.jpeg:
                        cv2.imwrite(str(out / "images" / f"{stem}.jpg"),
                                    g["scene"][:, :, ::-1],
                                    [int(cv2.IMWRITE_JPEG_QUALITY), int(a.jpeg)])
                    else:
                        cv2.imwrite(str(out / "images" / f"{stem}.png"), g["scene"][:, :, ::-1])
                    cv2.imwrite(str(out / "masks" / f"{stem}.png"), g["seg"][:, :, ::-1])
                    if a.thermal and "ir" in g:
                        cv2.imwrite(str(out / "ir" / f"{stem}.png"), g["ir"][:, :, ::-1])
                    (out / "labels" / f"{stem}.txt").write_text(
                        "\n".join(to_yolo(labs, g["seg"].shape[1], g["seg"].shape[0])), encoding="utf-8")
                    (out / "labels" / f"{stem}.json").write_text(json.dumps(
                        [{"actor_id": m.actor_id, "name": m.name, "cls": m.cls, "bbox_px": list(m.bbox_px),
                          "visible_px": m.visible_px, "size_px": m.size_px, "pose": m.pose,
                          "submersion": m.submersion, "occlusion": m.occlusion, "zone": m.zone,
                          "group": m.group, "aerially_detectable": m.aerially_detectable,
                          "visible_fraction": m.visible_fraction,
                          "amodal_bbox_px": list(m.amodal_bbox_px) if m.amodal_bbox_px else None,
                          "occlusion_basis": "measured per observation from the DepthPlanar buffer"}
                         for m in labs],
                        indent=1), encoding="utf-8")
                    # GSD must come from the MEASURED height above the surface, not the commanded --alt. The
                    # drone does not hold the setpoint exactly over rising ground, and writing the constant
                    # made every GSD and every geolocation number wrong by however much it was off.
                    agl_now = asl - surface_asl(cur_e, cur_n)
                    tw.writerow([k, time.time(), clip_id, round(cur_e, 2), round(cur_n, 2), round(asl, 2),
                                 round(agl_now, 2), gp.latitude, gp.longitude,
                                 round(o.w_val, 6), round(o.x_val, 6), round(o.y_val, 6), round(o.z_val, 6),
                                 -90.0, round(float(cal["hfov_deg"]), 3), g["seg"].shape[1], g["seg"].shape[0],
                                 round(agl_now / f_px * 100.0, 4), "AUTO", round(water, 3), len(labs),
                                 round(spd, 2)])
                    agls.append(agl_now)
                    total += len(labs)
                    seen.update(m.actor_id for m in labs)
                    k += 1
                    shots += 1
                    last_shot = stem
                    last_n = cur_n
                if reached or time.time() - t_leg > 400:
                    break
                time.sleep(0.05)
            fut.join()
            print(f"    {shots} shots in {time.time() - t_leg:.0f}s, {len(seen)} unique survivors so far")
        print(f"\nreturning to launch...")
        c.moveToPositionAsync(0.0, 0.0, -(surface_asl(home["east_m"], home["north_m"]) + a.alt
                                          - home["ground_asl_m"]), a.speed, timeout_sec=120).join()
    finally:
        tf.close()
        try:
            c.landAsync(timeout_sec=30)
            c.armDisarm(False)
            c.enableApiControl(False)
        except Exception:
            pass

    detset = {x["id"] for x in det}
    card = {"clip_id": clip_id, "scenario_seed": truth["seed"], "altitude_m_agl": a.alt,
            "speed_ms": a.speed, "shutter_m": a.shutter_m, "frames": k, "total_boxes": total,
            "plan": a.plan, "patch_link_m": a.patch_link_m, "lines": len(legs),
            "condition": a.condition, "time_of_day": a.time_of_day or None,
            "weather": {"rain": a.rain, "fog": a.fog, "dust": a.dust},
            "image_format": ("jpeg", a.jpeg) if a.jpeg else ("png", None),
            "unique_actors_seen": len(seen), "detectable_in_box": inside,
            "detectable_seen": len(seen & detset), "buried_seen": sorted(seen - detset),
            "f_px": f_px,
            "gsd_cm_px_nominal": round(a.alt / f_px * 100.0, 4),
            "agl_m_measured": ({"min": round(min(agls), 1), "median": round(sorted(agls)[len(agls) // 2], 1),
                                "max": round(max(agls), 1)} if agls else None),
            "gsd_note": "per-frame gsd_cm_px in telemetry.csv is from MEASURED AGL, not the commanded altitude",
            "capture": "flown coverage pattern (F2), not teleported",
            # The instance mask cannot see foliage: Cosys-AirSim's annotation renderer disables the
            # InstancedFoliage/InstancedGrass show flags, and every plant here is a HISM instance. So
            # boxes are gated on the DepthPlanar buffer, which does render the canopy. This counts the
            # observations that gate removed: survivors the mask claimed but the camera cannot see.
            "depth_gated": True,
            "observations_hidden_by_occluders": n_hidden,
            "domain": "sim", "randomisation": "off (5.5c)", "split_rule": "by scenario seed",
            "minutes": round((time.time() - t_start) / 60.0, 1), "last": last_shot}
    (out / "data_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(f"\n{k} frames, {total} boxes, {len(seen & detset)}/{inside} survivors in the box seen, "
          f"{card['minutes']} min")
    print(f"depth gate: {n_hidden} observation(s) dropped as fully hidden by foliage or rubble")
    print(f"written to {out}")
    print("NOW LOOK: uv run python tools/capture/contact_sheet.py " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
