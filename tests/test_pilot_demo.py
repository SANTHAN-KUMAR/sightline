"""F3 judge-demo behaviour: the idle hand-back, the velocity manual path, the pad map, the HUD feed.

These cover the things `tests/test_live_mission.py` could not, because they did not exist: the machine had
no idle timer, MANUAL was an RC handover, the button map was four literals, and the HUD had no feed.

The bar for each test here is the same one the rest of the project uses: it must fail if the behaviour
regresses, and it must be about a thing a judge would notice.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sightline.api.control_feed import ControlState
from sightline.mission.manual import EnvelopeClamp, ManualLimits, ManualPilot, OrbitAssist
from sightline.mission.padmap import (CONTROL_AXES, CONTROL_BUTTONS, PadMap, XBOX360_SDL_WINDOWS,
                                      describe_unknown_pad, load_pad_map, mark_measured, pad_slug,
                                      save_pad_map)
from sightline.mission.takeover import ControlInput, TakeoverMachine, VehicleAuthority


# --- the idle hand-back ------------------------------------------------------------------------------------
def _fly(m: TakeoverMachine, t: float, **kw) -> float:
    m.poll(ControlInput(t=t, valid=True, **kw))
    return t


def test_a_judge_who_walks_away_gets_the_mission_back():
    """The behaviour the whole demo turns on: MANUAL does not last for ever."""
    m = TakeoverMachine(idle_resume_s=10.0, clock=lambda: 0.0)
    _fly(m, 0.0)
    _fly(m, 0.1, pitch=0.9)
    assert m.mode == "MANUAL"

    t = 0.2
    while t < 10.0:                       # sticks centred, no buttons, nobody there
        m.poll(ControlInput(t=t, valid=True))
        t += 0.1
    assert m.mode == "MANUAL", "handed back early - the timer must measure from the last INPUT"

    tr = m.poll(ControlInput(t=10.2, valid=True))
    assert m.mode == "AUTO" and tr is not None
    assert tr.reason == "idle hand-back" and m.idle_handbacks == 1
    assert m.resume_pending, "the mission must re-plan from the current pose, exactly as after RESUME"


def test_the_timer_resets_every_time_the_pilot_does_anything():
    """A judge lining up a shot for a minute must never be interrupted."""
    m = TakeoverMachine(idle_resume_s=5.0, clock=lambda: 0.0)
    _fly(m, 0.0)
    _fly(m, 0.1, roll=0.9)
    t = 0.2
    for _ in range(12):                   # nudge the stick every 4 s for the best part of a minute
        for _ in range(40):
            m.poll(ControlInput(t=t, valid=True))
            t += 0.1
        m.poll(ControlInput(t=t, valid=True, roll=0.8))
        t += 0.1
    assert m.mode == "MANUAL" and m.idle_handbacks == 0


def test_a_button_press_counts_as_being_present_even_when_it_changes_nothing():
    """MARK is not a mode button, but a judge pressing it is plainly still flying."""
    m = TakeoverMachine(idle_resume_s=3.0, clock=lambda: 0.0)
    _fly(m, 0.0)
    _fly(m, 0.1, yaw=0.9)
    t = 0.2
    for _ in range(25):
        m.poll(ControlInput(t=t, valid=True))
        t += 0.1
    m.poll(ControlInput(t=t, valid=True, mark=True))     # 2.5 s in: resets the clock
    t += 0.1
    for _ in range(25):
        m.poll(ControlInput(t=t, valid=True))
        t += 0.1
    assert m.mode == "MANUAL"


def test_a_pad_that_disappears_mid_flight_hands_the_mission_back():
    """Put the pad down, unplug it, or let it go flat: "nobody is holding this" is the whole point."""
    m = TakeoverMachine(idle_resume_s=4.0, clock=lambda: 0.0)
    _fly(m, 0.0)
    _fly(m, 0.1, pitch=0.9)
    t = 0.2
    while t < 5.0:
        m.poll(ControlInput(t=t, valid=False))          # the device is gone
        t += 0.1
    assert m.mode == "AUTO" and m.idle_handbacks == 1


def test_no_pad_was_ever_attached_and_the_mission_simply_flies():
    """The inverse case, which must NOT change: an absent controller cannot move the machine."""
    m = TakeoverMachine(idle_resume_s=2.0, clock=lambda: 0.0)
    for k in range(200):
        m.poll(ControlInput(t=k * 0.1, valid=False))
    assert m.mode == "AUTO" and m.idle_handbacks == 0 and not m.transitions


def test_rtl_is_never_cancelled_by_a_timer():
    """RTL means 'something is wrong, go home'. A convenience timer must not undo it."""
    m = TakeoverMachine(idle_resume_s=2.0, clock=lambda: 0.0)
    m.poll(ControlInput(t=0.0, valid=True, rtl=True))
    assert m.mode == "RTL"
    t = 0.1
    while t < 20.0:
        m.poll(ControlInput(t=t, valid=True))
        t += 0.1
    assert m.mode == "RTL"


def test_rtl_cannot_be_configured_into_the_idle_set():
    with pytest.raises(ValueError, match="RTL must never be idle-resumed"):
        TakeoverMachine(idle_modes=("MANUAL", "RTL"))


def test_hold_is_idle_resumed_because_it_is_also_a_human_asking_for_a_pause():
    m = TakeoverMachine(idle_resume_s=3.0, clock=lambda: 0.0)
    m.poll(ControlInput(t=0.0, valid=True, hold=True))
    assert m.mode == "HOLD"
    t = 0.1
    while t < 4.0:
        m.poll(ControlInput(t=t, valid=True))
        t += 0.1
    assert m.mode == "AUTO"


def test_the_timer_can_be_switched_off_to_restore_the_documented_behaviour():
    """§5.2 as written: only RESUME ends a takeover. Still reachable, and still tested."""
    m = TakeoverMachine(idle_resume_s=0.0, clock=lambda: 0.0)
    _fly(m, 0.0)
    _fly(m, 0.1, pitch=0.9)
    t = 0.2
    while t < 600.0:
        m.poll(ControlInput(t=t, valid=True))
        t += 1.0
    assert m.mode == "MANUAL" and m.idle_remaining() is None


def test_the_hud_countdown_is_none_when_nothing_can_fire_and_a_number_when_it_can():
    m = TakeoverMachine(idle_resume_s=10.0, clock=lambda: 0.0)
    _fly(m, 0.0)
    assert m.idle_remaining(0.0) is None, "AUTO has nothing to count down"
    _fly(m, 0.1, pitch=0.9)
    assert m.idle_remaining(0.2) is None, "the pilot is still holding the stick"
    m.poll(ControlInput(t=1.0, valid=True))
    assert m.idle_remaining(3.0) == pytest.approx(8.0, abs=0.01)


def test_an_explicit_resume_still_wins_over_the_timer():
    m = TakeoverMachine(idle_resume_s=30.0, clock=lambda: 0.0)
    _fly(m, 0.0)
    _fly(m, 0.1, pitch=0.9)
    m.poll(ControlInput(t=0.2, valid=True))
    tr = m.poll(ControlInput(t=2.0, valid=True, resume=True))
    assert m.mode == "AUTO" and tr.reason == "RESUME button" and m.idle_handbacks == 0


# --- the velocity manual path ------------------------------------------------------------------------------
class _FakeVehicle:
    def __init__(self):
        self.velocity_calls: list[tuple] = []
        self.api_calls: list[bool] = []
        self.cancelled = 0

    def moveByVelocityBodyFrameAsync(self, vx, vy, vz, duration, **kw):   # noqa: N802
        self.velocity_calls.append((vx, vy, vz, duration, kw))

    def enableApiControl(self, on, vehicle=""):                            # noqa: N802
        self.api_calls.append(bool(on))

    def cancelLastTask(self, vehicle=""):                                  # noqa: N802
        self.cancelled += 1

    def hoverAsync(self):                                                  # noqa: N802
        return None


class _FlatTerrain:
    def __init__(self, asl: float = 0.0):
        self.asl = asl

    def surface_asl(self, east_m, north_m):
        return self.asl


def _pilot(**kw) -> tuple[ManualPilot, _FakeVehicle]:
    c = _FakeVehicle()
    return ManualPilot(c, terrain=_FlatTerrain(), home={"east_m": 0.0, "north_m": 0.0}, **kw), c


def test_centred_sticks_command_zero_velocity_which_is_what_hold_station_means():
    """The whole reason this path exists: a released stick must not be 50 percent motor output."""
    p, c = _pilot()
    for k in range(40):                    # let the smoothing settle
        p.command(ControlInput(t=k * 0.05, valid=True), east_m=0, north_m=0, alt_asl_m=50, dt=0.05)
    vx, vy, vz, _, _ = c.velocity_calls[-1]
    assert abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(vz) < 1e-6
    assert c.api_calls == [], "the velocity path must never touch enableApiControl"


def test_the_sticks_map_the_way_a_person_expects():
    p, c = _pilot(limits=ManualLimits(max_speed_ms=10.0, max_climb_ms=4.0, smooth_tau_s=0.0))
    p.command(ControlInput(t=0.0, valid=True, pitch=1.0), east_m=0, north_m=0, alt_asl_m=50, dt=0.1)
    assert c.velocity_calls[-1][0] == pytest.approx(10.0), "stick forward -> fly forward"
    p.command(ControlInput(t=0.1, valid=True, roll=1.0), east_m=0, north_m=0, alt_asl_m=50, dt=0.1)
    assert c.velocity_calls[-1][1] == pytest.approx(10.0), "stick right -> slide right"
    p.command(ControlInput(t=0.2, valid=True, throttle=1.0), east_m=0, north_m=0, alt_asl_m=50, dt=0.1)
    assert c.velocity_calls[-1][2] == pytest.approx(-4.0), "throttle up -> climb (NED z is DOWN)"


def test_the_terrain_floor_refuses_to_let_a_judge_fly_into_the_ground():
    p, c = _pilot(limits=ManualLimits(min_agl_m=10.0, max_climb_ms=4.0, smooth_tau_s=0.0))
    out = p.command(ControlInput(t=0.0, valid=True, throttle=0.0),      # full descent
                    east_m=0, north_m=0, alt_asl_m=10.0, dt=0.1)        # exactly at the floor
    assert c.velocity_calls[-1][2] == 0.0
    assert out["clamp"]["floor"] is True and "floor" in out["clamp"]["reasons"][0]


def test_the_floor_bleeds_off_instead_of_hitting_a_wall():
    """Stopping dead at the limit feels broken; the last few metres should feel like a cushion."""
    p, c = _pilot(limits=ManualLimits(min_agl_m=10.0, max_climb_ms=4.0, smooth_tau_s=0.0))
    p.command(ControlInput(t=0.0, valid=True, throttle=0.0), east_m=0, north_m=0, alt_asl_m=12.5, dt=0.1)
    vz = c.velocity_calls[-1][2]
    assert 0.0 < vz < 4.0, f"expected a reduced descent halfway into the margin, got {vz}"


def test_the_bleed_off_is_reported_rather_than_silent():
    """Found by flying it: the envelope limited a real descent and `reasons` came back empty.

    Measured on the live simulator 2026-09-11 - floor 1 m under the aircraft, full-down stick, descent
    correctly reduced, HUD told nothing. A pilot who is being overridden without being told concludes the
    controller is broken, which is the one thing `EnvelopeClamp` is for.
    """
    p, _ = _pilot(limits=ManualLimits(min_agl_m=10.0, max_climb_ms=4.0, smooth_tau_s=0.0))
    out = p.command(ControlInput(t=0.0, valid=True, throttle=0.0),
                    east_m=0, north_m=0, alt_asl_m=11.0, dt=0.1)      # 1 m above the floor, inside the margin
    assert out["clamp"]["any"] is True and out["clamp"]["floor"] is True
    assert any("floor" in r for r in out["clamp"]["reasons"]), out["clamp"]

    # ...and well clear of the limit it must stay quiet, or the warning means nothing.
    quiet = p.command(ControlInput(t=0.1, valid=True, throttle=0.0),
                      east_m=0, north_m=0, alt_asl_m=60.0, dt=0.1)
    assert quiet["clamp"]["any"] is False, quiet["clamp"]


def test_the_ceiling_bleed_off_is_reported_too():
    p, _ = _pilot(limits=ManualLimits(max_agl_m=120.0, max_climb_ms=4.0, smooth_tau_s=0.0))
    out = p.command(ControlInput(t=0.0, valid=True, throttle=1.0),
                    east_m=0, north_m=0, alt_asl_m=118.0, dt=0.1)
    assert out["clamp"]["ceiling"] is True and any("ceiling" in r for r in out["clamp"]["reasons"])


def test_the_ceiling_is_enforced_on_a_human_too():
    p, c = _pilot(limits=ManualLimits(max_agl_m=120.0, max_climb_ms=4.0, smooth_tau_s=0.0))
    out = p.command(ControlInput(t=0.0, valid=True, throttle=1.0),
                    east_m=0, north_m=0, alt_asl_m=120.0, dt=0.1)
    assert c.velocity_calls[-1][2] == 0.0 and out["clamp"]["ceiling"] is True


def test_descending_is_allowed_at_the_ceiling_and_climbing_at_the_floor():
    """A clamp that traps the pilot at the limit is worse than no clamp."""
    p, c = _pilot(limits=ManualLimits(min_agl_m=10.0, max_agl_m=120.0, max_climb_ms=4.0, smooth_tau_s=0.0))
    p.command(ControlInput(t=0.0, valid=True, throttle=1.0), east_m=0, north_m=0, alt_asl_m=10.0, dt=0.1)
    assert c.velocity_calls[-1][2] < 0, "must still be able to climb off the floor"
    p.command(ControlInput(t=0.1, valid=True, throttle=0.0), east_m=0, north_m=0, alt_asl_m=120.0, dt=0.1)
    assert c.velocity_calls[-1][2] > 0, "must still be able to descend from the ceiling"


class _Fence:
    """Everything within 100 m of the origin is inside; nothing is a no-go."""

    def inside_geofence(self, ne):
        return math.hypot(ne[0], ne[1]) <= 100.0

    def inside_no_go(self, ne):
        return False


def test_the_geofence_turns_a_judge_back_instead_of_freezing_them():
    p, c = _pilot(limits=ManualLimits(max_speed_ms=10.0, smooth_tau_s=0.0))
    p.constraints = _Fence()
    # 95 m north of home, flying further north at 10 m/s: 2 s of lookahead puts it outside.
    out = p.command(ControlInput(t=0.0, valid=True, pitch=1.0),
                    east_m=0.0, north_m=95.0, alt_asl_m=50.0, heading_deg=0.0, dt=0.1)
    assert out["clamp"]["geofence"] is True
    assert c.velocity_calls[-1][0] <= 0.0, "outbound velocity must not survive the fence"

    # ...but flying back toward home from the same place is untouched.
    out2 = p.command(ControlInput(t=0.1, valid=True, pitch=-1.0),
                     east_m=0.0, north_m=95.0, alt_asl_m=50.0, heading_deg=0.0, dt=0.1)
    assert c.velocity_calls[-1][0] == pytest.approx(-10.0, abs=0.5), out2


def test_boost_raises_the_speed_limit_and_nothing_else():
    p, c = _pilot(limits=ManualLimits(max_speed_ms=9.0, boost_speed_ms=18.0, smooth_tau_s=0.0))
    p.command(ControlInput(t=0.0, valid=True, pitch=1.0), east_m=0, north_m=0, alt_asl_m=50, dt=0.1)
    slow = c.velocity_calls[-1][0]
    p.command(ControlInput(t=0.1, valid=True, pitch=1.0, boost=True),
              east_m=0, north_m=0, alt_asl_m=50, dt=0.1)
    assert c.velocity_calls[-1][0] == pytest.approx(2.0 * slow, rel=0.01)


def test_the_authority_never_releases_api_control_on_the_velocity_path():
    """The safety claim, asserted: simple_flight's RC channels are never reachable by the pilot."""
    c = _FakeVehicle()
    m = TakeoverMachine()
    auth = VehicleAuthority(c, manual_mode="velocity", hover_fn=c.hoverAsync)
    out = auth.apply(m.poll(ControlInput(t=1.0, roll=0.9, valid=True)))
    assert c.api_calls == [], "enableApiControl(False) would hand the judge the disarm gesture"
    assert out["did"] == ["api_control=True (velocity manual)"]


def test_leaving_manual_releases_the_velocity_pilot_before_the_mission_re_issues():
    """Otherwise the pilot and the waypoint fight for one poll and the aircraft lurches."""
    c = _FakeVehicle()
    released: list[int] = []
    m = TakeoverMachine()
    auth = VehicleAuthority(c, manual_mode="velocity", release_fn=lambda: released.append(1),
                            hover_fn=c.hoverAsync)
    auth.apply(m.poll(ControlInput(t=1.0, roll=0.9, valid=True)))
    m.poll(ControlInput(t=2.0, valid=True))
    auth.apply(m.poll(ControlInput(t=3.5, resume=True, valid=True)))
    assert released == [1]


def test_orbit_circles_the_mark_and_points_the_nose_at_it():
    o = OrbitAssist(radius_m=20.0, period_s=20.0)
    assert not o.engaged and o.command(0.0, 0.0, 0.0) is None
    o.engage(east_m=100.0, north_m=100.0, now=0.0)
    vn, ve, yaw = o.command(1.0, east_m=120.0, north_m=100.0)     # due east of the centre, on the circle
    assert math.hypot(vn, ve) == pytest.approx(2 * math.pi * 20.0 / 20.0, rel=0.2)
    assert yaw == pytest.approx(-90.0, abs=1.0), "nose must swing back toward the mark"
    o.release()
    assert not o.engaged


# --- the pad map -------------------------------------------------------------------------------------------
def test_the_shipped_xbox_map_is_self_consistent_against_its_own_measured_rest_state():
    assert XBOX360_SDL_WINDOWS.validate() == []
    assert set(XBOX360_SDL_WINDOWS.axes) == set(CONTROL_AXES)
    assert set(CONTROL_BUTTONS) <= set(XBOX360_SDL_WINDOWS.buttons)


def test_the_shipped_map_does_not_claim_to_be_measured():
    """The honesty rule. It becomes "measured" only when a human has pressed the buttons."""
    assert XBOX360_SDL_WINDOWS.provenance == "partial"
    assert XBOX360_SDL_WINDOWS.verified is False


def test_a_trigger_mapped_as_a_stick_is_refused():
    """The bug this project already shipped once: pitch on axis 4, which rests at -1.0."""
    m = PadMap(name="x", slug="x", axes={"roll": 2, "pitch": 4, "yaw": 0, "throttle": 1},
               buttons={"takeover": 1, "resume": 0, "hold": 2, "rtl": 3}, n_axes=6, n_buttons=11)
    problems = m.validate(rest_axes=(0.0, 0.0, 0.0, 0.0, -1.0, -1.0))
    assert any("TRIGGER" in p for p in problems), problems


def test_two_controls_on_one_button_is_refused():
    m = PadMap(name="x", slug="x", axes=dict(XBOX360_SDL_WINDOWS.axes),
               buttons={"takeover": 1, "resume": 0, "hold": 2, "rtl": 2}, n_axes=6, n_buttons=11)
    assert any("both index 2" in p for p in m.validate())


def test_a_missing_mode_button_is_refused():
    m = PadMap(name="x", slug="x", axes=dict(XBOX360_SDL_WINDOWS.axes),
               buttons={"takeover": 1, "resume": 0, "hold": 2}, n_axes=6, n_buttons=11)
    assert any("'rtl' is not bound" in p for p in m.validate())


def test_the_dead_zone_rescales_so_the_stick_still_reaches_full_travel():
    m = PadMap(dead_zone=0.2)
    assert m.apply_dead_zone(0.1) == 0.0
    assert m.apply_dead_zone(1.0) == pytest.approx(1.0)
    assert m.apply_dead_zone(-1.0) == pytest.approx(-1.0)
    assert m.apply_dead_zone(0.6) == pytest.approx(0.5)


def test_a_saved_map_round_trips_and_is_loaded_by_device_name(tmp_path):
    m = mark_measured(PadMap(name="Test Pad 9000", slug=pad_slug("Test Pad 9000"),
                             axes=dict(XBOX360_SDL_WINDOWS.axes), buttons={"takeover": 1, "resume": 0,
                                                                           "hold": 2, "rtl": 3},
                             invert=("pitch",), n_axes=6, n_buttons=11))
    p = save_pad_map(m, repo=tmp_path)
    assert json.loads(p.read_text())["provenance"] == "measured"
    back = load_pad_map("Test Pad 9000", repo=tmp_path, n_axes=6, n_buttons=11)
    assert back.verified and back.axes == m.axes and back.invert == ("pitch",)


def test_a_saved_map_for_a_different_device_shape_is_refused(tmp_path):
    """Same name, different hardware: applying it would put RESUME on whatever sits at that index."""
    m = mark_measured(PadMap(name="Xbox 360 Controller", slug="xbox-360-controller",
                             axes=dict(XBOX360_SDL_WINDOWS.axes),
                             buttons=dict(XBOX360_SDL_WINDOWS.buttons), n_axes=6, n_buttons=11))
    save_pad_map(m, repo=tmp_path)
    back = load_pad_map("Xbox 360 Controller", repo=tmp_path, n_axes=8, n_buttons=14)
    assert not back.verified, "a map for a 6-axis pad must not be flown on an 8-axis one"
    assert "6-axis" in back.note


def test_an_unknown_pad_says_so_rather_than_pretending(tmp_path):
    m = load_pad_map("Some Unknown Pad", repo=tmp_path, n_axes=6, n_buttons=10)
    assert m.provenance == "default-guess" and not m.verified
    assert "NO MAP EXISTS" in m.note and "pad_calibrate" in m.note


def test_a_corrupt_map_file_falls_back_instead_of_flying_garbage(tmp_path):
    d = tmp_path / "data" / "controller"
    d.mkdir(parents=True)
    (d / "xbox-360-controller.json").write_text("{not json", encoding="utf-8")
    m = load_pad_map("Xbox 360 Controller", repo=tmp_path, n_axes=6, n_buttons=11)
    assert m.axes == XBOX360_SDL_WINDOWS.axes and not m.verified


# --- the HUD feed ------------------------------------------------------------------------------------------
def test_the_control_snapshot_stays_inside_the_radio_budget():
    """§5.8: a live-feed frame must fit in MAX_MESSAGE_BYTES. A 200-event backlog would not."""
    from sightline.api.live import MAX_MESSAGE_BYTES, envelope

    st = ControlState()
    for k in range(400):
        st.event("mode", f"AUTO -> MANUAL {k}", detail="stick deflection, a reasonably long explanation")
    st.update(mode="MANUAL", sticks={"roll": 0.5, "pitch": -0.2, "yaw": 0.1, "throttle": 0.8},
              buttons={n: False for n in ("takeover", "resume", "hold", "rtl", "mark", "boost")})
    wire = json.dumps(envelope("control", 1, **st.snapshot()))
    assert len(wire.encode()) <= MAX_MESSAGE_BYTES, f"{len(wire.encode())} bytes"


def test_the_http_route_can_still_return_the_whole_log():
    st = ControlState()
    for k in range(50):
        st.event("note", f"line {k}")
    assert len(st.snapshot(max_events=0)["events"]) == 50
    assert len(st.snapshot()["events"]) == 12


def test_events_are_append_only_and_resume_from_a_sequence():
    st = ControlState()
    st.event("mode", "one")
    st.event("mark", "two")
    seq = st.snapshot()["event_seq"]
    st.event("note", "three")
    fresh = st.snapshot(since_seq=seq)["events"]
    assert [e["text"] for e in fresh] == ["three"]


def test_the_feed_never_claims_a_guessed_pad_was_measured():
    st = ControlState()
    st.update(pad={"attached": True, "device": "Xbox 360 Controller", "verified": False,
                   "provenance": "partial"})
    snap = st.snapshot()
    assert snap["pad"]["verified"] is False and snap["pad"]["provenance"] == "partial"


def test_the_api_publishes_control_state_and_serves_it_back():
    from fastapi.testclient import TestClient

    from sightline.api.app import create_app
    from sightline.store import RecordStore

    app = create_app(store=RecordStore(":memory:"), serve_static=False)
    with TestClient(app) as client:
        r = client.post("/api/mission/control", json={
            "mode": "MANUAL", "idle_remaining_s": 7.5, "idle_resume_s": 12.0,
            "sticks": {"roll": 0.4, "pitch": -0.9, "yaw": 0.0, "throttle": 0.7},
            "buttons": {"mark": True}, "free_flight": True,
            "events": [{"kind": "mode", "text": "AUTO -> MANUAL", "detail": "stick deflection"}],
        })
        assert r.status_code == 200 and r.json()["ok"]
        got = client.get("/api/mission/control").json()
        assert got["mode"] == "MANUAL" and got["free_flight"] is True
        assert got["idle_remaining_s"] == 7.5 and got["sticks"]["pitch"] == -0.9
        assert [e["text"] for e in got["events"]] == ["AUTO -> MANUAL"]
        assert client.get("/health").json()["control"]["mode"] == "MANUAL"


def test_a_hud_that_connects_mid_flight_is_not_blank():
    """The reason the control frame is part of the WebSocket handshake at all."""
    from fastapi.testclient import TestClient

    from sightline.api.app import create_app
    from sightline.store import RecordStore

    app = create_app(store=RecordStore(":memory:"), serve_static=False)
    with TestClient(app) as client:
        client.post("/api/mission/control", json={"mode": "MANUAL"})
        with client.websocket_connect("/ws") as ws:
            types = []
            for _ in range(4):
                m = ws.receive_json()
                types.append(m["type"])
                if m["type"] == "control":
                    assert m["mode"] == "MANUAL"
                    break
            assert "control" in types, types


def test_a_fresh_server_does_not_send_a_control_frame_nobody_asked_for():
    from fastapi.testclient import TestClient

    from sightline.api.app import create_app
    from sightline.store import RecordStore

    app = create_app(store=RecordStore(":memory:"), serve_static=False)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            assert ws.receive_json()["type"] == "snapshot"


# --- R10 -----------------------------------------------------------------------------------------------
def test_nothing_in_the_pilot_path_can_delete_or_clear():
    """Guardrail R10 over the modules this lane added, using the PROJECT's scanner, not a token list.

    `sightline/triage/guardrails.py` already knows the difference between a record deletion and a dict's
    `kwargs.pop("valid")`, and it carries a per-site allowance table that is itself audited by
    `tests/test_guardrails_scan.py`. Re-implementing a cruder version here would only teach us that
    `.pop(` appears in Python.
    """
    from sightline.triage.guardrails import scan_source

    for name in ("sightline/mission/manual.py", "sightline/mission/padmap.py",
                 "sightline/mission/takeover.py", "sightline/api/control_feed.py"):
        found = scan_source(REPO / name, repo_root=REPO)
        assert found == [], f"{name}: {[str(v) for v in found]}"


def test_the_new_pilot_tools_carry_no_record_deletion_either():
    """`tools/` is outside the lane dirs the scanner walks, so point it at these files explicitly."""
    from sightline.triage.guardrails import scan_source

    for name in ("tools/live/demo_controller.py", "tools/live/pad_calibrate.py"):
        found = scan_source(REPO / name, repo_root=REPO)
        assert found == [], f"{name}: {[str(v) for v in found]}"


def test_a_mark_only_ever_adds():
    """There is deliberately no unmark: a human saying 'I saw something here' is evidence."""
    src = (REPO / "sightline" / "mission" / "live.py").read_text(encoding="utf-8")
    assert "def place_mark" in src
    assert "def unmark" not in src and "self.marks.remove" not in src and "self.marks.clear" not in src
