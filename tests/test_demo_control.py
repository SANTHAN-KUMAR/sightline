"""The demo launcher: what it will run, where it sends the results, and when it refuses.

These are reproducibility tests. The failures they cover are all of one kind - a button that launches
something subtly different from what the operator expects - and all of them have actually happened here:

* a flag was added to the launch line that the launcher did not accept, so every demo died instantly on
  argparse (caught before shipping, but only just);
* the two launchers name the C2 differently (`--port` vs `--c2`), so one correct fix is one broken demo;
* the launch line hardcoded port 8781 while the server answered somewhere else, which does not error - it
  quietly streams the flight into a different dashboard than the one being watched.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sightline.api import demo_control as dc

REPO = Path(__file__).resolve().parents[1]


def test_every_catalog_entry_is_launchable() -> None:
    """Ids are the whole API surface, so an entry with no id or no command is a 500 waiting to happen."""
    assert dc.CATALOG, "an empty catalog means a dashboard with no buttons"
    for e in dc.CATALOG:
        assert e["id"] and e["title"] and e["sub"], e
        assert isinstance(e["cmd"], list) and len(e["cmd"]) > 4, e
    ids = [e["id"] for e in dc.CATALOG]
    assert len(ids) == len(set(ids)), f"duplicate demo ids: {ids}"


def test_unknown_id_is_refused_not_run() -> None:
    code, payload = dc.RUNNER.start("../../etc/passwd")
    assert code == 404
    assert "unknown demo" in payload["error"]


@pytest.mark.parametrize("entry", dc.CATALOG, ids=[e["id"] for e in dc.CATALOG])
def test_launch_line_points_at_this_server(entry: dict) -> None:
    """The C2 that launches a flight must be the C2 that receives it - with each launcher's own flag."""
    dc.set_serve_port(8899)
    try:
        cmd = list(entry["cmd"]) + dc._c2_args(entry)
        if entry.get("c2_flag") == "url":
            assert "--c2" in cmd and "http://127.0.0.1:8899" in cmd
            assert "--port" not in cmd, "demo_controller.py has no --port; it would die on argparse"
        else:
            assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8899"
            assert "--c2" not in cmd, "demo.py has no --c2; it would die on argparse"
    finally:
        dc.set_serve_port(8781)


@pytest.mark.parametrize("entry", dc.CATALOG, ids=[e["id"] for e in dc.CATALOG])
def test_launcher_accepts_every_flag_we_send_it(entry: dict) -> None:
    """Parse the real launch line with the real parser.

    A demo that dies on `unrecognized arguments` two seconds after the judge presses Start is the exact
    failure this suite exists to prevent, and only the launcher's own argparse can rule it out. `--help`
    is not enough: it proves the flag exists, not that this combination parses.
    """
    dc.set_serve_port(8899)
    try:
        cmd = list(entry["cmd"]) + dc._c2_args(entry)
    finally:
        dc.set_serve_port(8781)
    script = cmd[4]                                   # [uv, run, python, -u, <script>, ...]
    args = cmd[5:]
    probe = (
        "import argparse,runpy,sys,types\n"
        "sys.argv=[%r]+%r\n"
        "real=argparse.ArgumentParser.parse_args\n"
        "def stop(self,*a,**k):\n"
        "    real(self,*a,**k)\n"
        "    raise SystemExit(17)\n"                  # parsed cleanly; do not then fly anything
        "argparse.ArgumentParser.parse_args=stop\n"
        "runpy.run_path(%r,run_name='__main__')\n" % (script, args, script)
    )
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO),
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 17, (
        f"{entry['id']}: launcher rejected its own launch line\n"
        f"  cmd: {script} {' '.join(args)}\n  rc={r.returncode}\n{r.stdout[-800:]}{r.stderr[-800:]}"
    )


def test_preflight_reports_every_check_and_blocks_on_failure() -> None:
    """A check that cannot run counts as a failure, and the reason has to be readable by a person."""
    ok, checks = dc.preflight("nominal")
    names = {c["name"] for c in checks}
    assert {"Detector", "Simulator"} <= names, names
    for c in checks:
        assert c["detail"] and len(c["detail"]) > 10, c
    assert ok == all(c["ok"] for c in checks)


def test_preflight_refuses_to_start_when_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal must arrive as a 412 with words, not as a launched process that fails later."""
    monkeypatch.setattr(dc, "preflight",
                        lambda _id: (False, [{"name": "Simulator", "ok": False,
                                              "detail": "AirSim not listening on 41451"}]))
    code, payload = dc.RUNNER.start("nominal")
    assert code == 412
    assert "41451" in payload["error"]
    assert not dc.RUNNER.running(), "a blocked preflight must not leave a process behind"


def test_pad_check_is_advisory_never_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    """The takeover demos are hand-flown and belong to the controller lane.

    A judge is entitled to open the page, press Start and plug the pad in afterwards, so the gamepad is
    reported and never blocks. Only the simulator does.
    """
    monkeypatch.setattr(dc, "_vehicle_check", lambda: (True, "vehicles=['SimpleFlight'], clock ticking"))
    ok, checks = dc.preflight("pilot_free")
    pad = [c for c in checks if c["name"] == "Gamepad"]
    assert pad and pad[0].get("advisory") is True
    assert ok, "a missing pad must not block the demo it is optional for"


def test_a_shell_that_grepped_for_a_flight_is_not_a_flight() -> None:
    """The exclusive-use probe must identify a PROGRAM, not a string.

    "mission.live" appears in the command line of any shell that ever searched for it - including the
    session that built this - and matching those made `preflight` report "another flight is already
    running" for a shell, which blocked every demo from starting. This is that regression.
    """
    assert dc._is_interpreter("D:/Sightline/.venv/Scripts/python.exe")
    assert dc._is_interpreter("uv.exe")
    assert dc._is_interpreter("python")
    assert not dc._is_interpreter("C:/Program Files/Git/bin/bash.exe")
    assert not dc._is_interpreter("powershell.exe")
    assert not dc._is_interpreter("UnrealEditor.exe")
    assert not dc._is_interpreter("")


def test_foreign_flight_does_not_find_this_process() -> None:
    """The scan must never report the runner's own child, or Start would refuse right after succeeding."""
    pid, _cmd = dc.foreign_flight()
    assert pid != __import__("os").getpid()


def test_chain_reports_every_link_with_a_reason() -> None:
    """A dashboard that says 'no data' is useless; one that says WHICH link is down is a diagnosis."""
    c = dc.chain(8801, records=3, ws_clients=2)
    names = [l["name"] for l in c["links"]]
    assert names == ["Unreal", "Flight", "Backend", "Camera"], names
    for link in c["links"]:
        assert isinstance(link["ok"], bool)
        assert link["detail"], f"{link['name']} gave no reason"
    assert c["dashboard"].endswith(":8801/app/map/index.html")
    backend = next(l for l in c["links"] if l["name"] == "Backend")
    assert "3 record(s)" in backend["detail"] and ":8801" in backend["detail"]


def test_status_names_the_port_it_will_launch_against() -> None:
    """A flight streaming to one port while another page is watched looks exactly like a dead pipeline."""
    dc.set_serve_port(8801)
    try:
        assert dc.RUNNER.status()["serve_port"] == 8801
    finally:
        dc.set_serve_port(8781)
