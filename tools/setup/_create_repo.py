"""Create the GitHub repository for this project using the credential git already has.

    uv run python tools/setup/_create_repo.py <name> [--private]

`gh` is not installed on this machine, but git stores a GitHub credential (`credential.helper = store`), so
the token is already on disk for exactly this host. This reads it, calls the REST API once, and prints only
the resulting URL - the token is never echoed, never logged and never written anywhere.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

NAME = sys.argv[1] if len(sys.argv) > 1 else "sightline"
PRIVATE = "--private" in sys.argv
DESC = ("Aerial survivor triage for flood disasters: a simulation-first computer-vision system. "
        "UE 5.8 + Cosys-AirSim, tiled YOLO26s detection, geolocation, tracking, dedup, triage, "
        "and an offline command map.")


def token_and_user() -> tuple[str, str]:
    """Pull the github.com credential out of ~/.git-credentials. Returns (user, token); never prints them."""
    p = Path.home() / ".git-credentials"
    if not p.exists():
        sys.exit("no ~/.git-credentials; run `git credential fill` or install gh and `gh auth login`")
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if "github.com" not in line or "://" not in line:
            continue
        rest = line.split("://", 1)[1]
        if "@" not in rest:
            continue
        creds, host = rest.rsplit("@", 1)
        if not host.startswith("github.com"):
            continue
        user, _, tok = creds.partition(":")
        if tok:
            return user, tok
    sys.exit("no github.com entry with a token found in ~/.git-credentials")


def main() -> int:
    user, tok = token_and_user()
    print(f"authenticating as {user} (token not shown)")
    body = json.dumps({
        "name": NAME, "description": DESC, "private": PRIVATE,
        "has_issues": True, "has_wiki": False, "auto_init": False,
    }).encode()
    req = urllib.request.Request(
        "https://api.github.com/user/repos", data=body, method="POST",
        headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
                 "User-Agent": "sightline-setup", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        if e.code == 422 and "already exists" in detail:
            print(f"repository {NAME} already exists; reusing it")
            print(f"REMOTE https://github.com/{user}/{NAME}.git")
            return 0
        sys.exit(f"GitHub API {e.code}: {detail}")
    print(f"created {'private' if d.get('private') else 'PUBLIC'} repo: {d['html_url']}")
    print(f"REMOTE {d['clone_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
