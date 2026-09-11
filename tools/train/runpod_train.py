"""Run the F8b fine-tune on a rented RunPod GPU instead of the laptop, and always give the GPU back.

    uv run python tools/train/runpod_train.py plan     --data _artifacts/yolo/sim   # costs nothing
    uv run python tools/train/runpod_train.py up       --gpu "NVIDIA GeForce RTX 4090"
    uv run python tools/train/runpod_train.py push     --data _artifacts/yolo/sim
    uv run python tools/train/runpod_train.py train    --epochs 40 --imgsz 1024
    uv run python tools/train/runpod_train.py pull
    uv run python tools/train/runpod_train.py down                                  # ALWAYS
    uv run python tools/train/runpod_train.py status

Why not train locally: the RTX 4060 Laptop has 8 GB and the Unreal editor wants most of it, so
`sightline/detect/train.py` refuses to start while the editor is up (`editor_is_running`). Renting a 24 GB
card removes that conflict entirely and finishes in a fraction of the time.

The one thing this file is really about is **not leaving a pod running**. A rented GPU bills by the second
whether or not anything is training, so:

  * `up` records the pod id, the price and the start time in `_artifacts/runpod_pod.json`;
  * every subcommand prints the accrued cost so far;
  * `down` is idempotent and safe to run at any time, including after a crash, and `status` will tell you a
    pod is alive even if this process has forgotten it;
  * `plan` does the whole cost arithmetic without renting anything.

Credentials come from `_secrets/runpod.env` (gitignored). The API key is sent to api.runpod.io and nowhere
else, and is never printed - not in the command echo, not in an error message.

Transport is plain OpenSSH: RunPod injects `PUBLIC_KEY` into the pod's authorized_keys, and exposes port 22
on a public host/port pair. The keypair lives in `_secrets/` and is generated on first use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SECRETS = REPO / "_secrets"
STATE = REPO / "_artifacts" / "runpod_pod.json"
KEY = SECRETS / "runpod_ed25519"
API = "https://api.runpod.io/graphql"

#: A CUDA image with PyTorch already built; ultralytics is pip-installed on top at train time.
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
REMOTE = "/workspace/sightline"


# --- credentials and API ---------------------------------------------------------------------------------
def api_key() -> str:
    env = SECRETS / "runpod.env"
    if not env.exists():
        sys.exit(f"missing {env}. Put RUNPOD_API_KEY=... in it (the file is gitignored).")
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("RUNPOD_API_KEY"):
            return line.split("=", 1)[1].strip()
    sys.exit("RUNPOD_API_KEY not found in _secrets/runpod.env")


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    # A default `Python-urllib/3.x` User-Agent is refused by Cloudflare in front of api.runpod.io with
    # HTTP 403 "error code: 1010" - the key is fine, the client string is not.
    req = urllib.request.Request(API, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key()}",
        "User-Agent": "sightline-f8b/1.0 (+https://github.com/SANTHAN-KUMAR)",
        "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            out = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:                     # never let the key reach a traceback
        sys.exit(f"RunPod API HTTP {e.code}: {e.read().decode()[:400]}")
    if "errors" in out:
        sys.exit(f"RunPod API error: {json.dumps(out['errors'])[:600]}")
    return out["data"]


# --- local state -----------------------------------------------------------------------------------------
def load() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(d, indent=1), encoding="utf-8")


def accrued(st: dict) -> str:
    if not st.get("t_start"):
        return ""
    h = (time.time() - st["t_start"]) / 3600.0
    return f"  [pod up {h * 60:.0f} min, about ${h * st.get('price_hr', 0):.2f} so far]"


def ensure_key() -> str:
    SECRETS.mkdir(parents=True, exist_ok=True)
    if not KEY.exists():
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "sightline-runpod",
                        "-f", str(KEY)], check=True)
        print(f"generated {KEY} (gitignored)")
    return (KEY.with_suffix(".pub")).read_text(encoding="utf-8").strip()


def ssh_target(st: dict) -> list[str]:
    return ["-i", str(KEY), "-p", str(st["ssh_port"]), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]


def scp_target(st: dict) -> list[str]:
    """The same options, but with the port flag scp actually understands.

    `ssh` takes `-p <port>`; **`scp` takes `-P <port>`, and its own `-p` means "preserve mtimes"**. Passing
    the ssh form to scp made it read the port number as a source path and fail with
    `scp: stat local "19730": No such file or directory` - which is what it did on the first real upload this
    project ever attempted. Both transfer directions were affected, so neither push nor pull had ever run.
    """
    return ["-i", str(KEY), "-P", str(st["ssh_port"]), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]


def ssh(st: dict, cmd: str, check: bool = True) -> int:
    full = ["ssh", *ssh_target(st), f"{st.get('ssh_user', 'root')}@{st['ssh_host']}", cmd]
    print(f"  ssh: {cmd[:110]}{'...' if len(cmd) > 110 else ''}")
    return subprocess.run(full, check=check).returncode


# --- pricing ---------------------------------------------------------------------------------------------
def gpu_price(gpu_id: str) -> float:
    d = gql("query($id:String){ gpuTypes(input:{id:$id}) { id displayName "
            "lowestPrice(input:{gpuCount:1}) { uninterruptablePrice } } }", {"id": gpu_id})
    rows = d.get("gpuTypes") or []
    if not rows or not (rows[0].get("lowestPrice") or {}).get("uninterruptablePrice"):
        sys.exit(f"no on-demand price for {gpu_id!r}; run the `plan` subcommand to list GPUs")
    return float(rows[0]["lowestPrice"]["uninterruptablePrice"])


def balance() -> float:
    return float(gql("query { myself { clientBalance } }")["myself"]["clientBalance"])


# --- subcommands -----------------------------------------------------------------------------------------
def cmd_plan(a) -> int:
    d = gql("query { gpuTypes { id displayName memoryInGb communityCloud secureCloud "
            "lowestPrice(input:{gpuCount:1}) { uninterruptablePrice } } }")
    rows = [g for g in d["gpuTypes"] if (g.get("lowestPrice") or {}).get("uninterruptablePrice")]
    rows.sort(key=lambda g: g["lowestPrice"]["uninterruptablePrice"])
    print(f"balance ${balance():.2f}\n")
    print(f"{'gpuTypeId':34s}{'GPU':20s}{'VRAM':>5}{'$/h':>8}")
    for g in rows[:12]:
        print(f"{g['id'][:33]:34s}{g['displayName'][:19]:20s}{g['memoryInGb']:>5}"
              f"{g['lowestPrice']['uninterruptablePrice']:>8.3f}")
    data = Path(a.data) if a.data and Path(a.data).is_absolute() else (REPO / a.data if a.data else None)
    if data and data.exists():
        n = sum(1 for _ in data.rglob("*.png")) + sum(1 for _ in data.rglob("*.jpg"))
        mb = sum(f.stat().st_size for f in data.rglob("*") if f.is_file()) / 1e6
        print(f"\ndataset {data}: {n} images, {mb:.0f} MB to upload")
        if n:
            price = 0.34
            hrs = max(0.25, n * a.epochs / 55000.0)     # ~55k tile-epochs an hour on a 4090, measured order
            print(f"rough estimate at ${price}/h for {a.epochs} epochs: {hrs:.1f} h, about ${hrs * price:.2f}")
    else:
        print("\n(pass --data <tiled yolo dir> to size the upload and the run)")
    return 0


def cmd_up(a) -> int:
    st = load()
    if st.get("pod_id"):
        print(f"pod {st['pod_id']} already recorded; `down` it first or use `status`{accrued(st)}")
        return 1
    pub = ensure_key()
    price = gpu_price(a.gpu)
    bal = balance()
    print(f"renting {a.gpu} at ${price:.3f}/h; balance ${bal:.2f} "
          f"(~{bal / max(price, 1e-6):.0f} h of runway)")
    d = gql(
        "mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input:$input) "
        "{ id imageName machineId } }",
        {"input": {
            "cloudType": a.cloud, "gpuCount": 1, "gpuTypeId": a.gpu,
            "name": "sightline-f8b", "imageName": IMAGE,
            "containerDiskInGb": a.disk_gb, "volumeInGb": 0,
            "minVcpuCount": 8, "minMemoryInGb": 24,
            "ports": "22/tcp", "startSsh": True,
            "env": [{"key": "PUBLIC_KEY", "value": pub}],
            "dockerArgs": "",
        }})
    pod = d["podFindAndDeployOnDemand"]
    if not pod:
        sys.exit(f"RunPod had no {a.gpu} available in {a.cloud}. Try another GPU from `plan`.")
    st = {"pod_id": pod["id"], "gpu": a.gpu, "price_hr": price, "t_start": time.time(), "image": IMAGE}
    save(st)
    print(f"pod {pod['id']} created; waiting for SSH...")
    return wait_ready(st, a.wait_min)


def wait_ready(st: dict, wait_min: float) -> int:
    """Wait for a SHELL, by either route RunPod offers.

    Two things learned the hard way on 2026-09-11:

    * `desiredStatus: RUNNING` with `runtime: null` means the pod is allocated but the CONTAINER has not
      started. The PyTorch image is ~20 GB and a cold community-cloud node needs well over 8 minutes to pull
      it, so a short timeout reports failure on a pod that is merely still downloading - and leaves it
      billing.
    * A community-cloud pod usually gets NO public TCP port, so waiting for `isIpPublic` on port 22 waits
      for ever. RunPod also exposes an SSH PROXY at `<podHostId>@ssh.runpod.io`, which works with the same
      injected PUBLIC_KEY. That is the route that actually connects here.
    """
    deadline = time.time() + wait_min * 60
    while time.time() < deadline:
        d = gql("query($id:String!){ pod(input:{podId:$id}) { id desiredStatus machine { podHostId } "
                "runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }",
                {"id": st["pod_id"]})
        pod = d.get("pod") or {}
        rt = pod.get("runtime") or {}
        for p in rt.get("ports") or []:
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                st["ssh_host"], st["ssh_port"], st["ssh_user"] = p["ip"], p["publicPort"], "root"
                save(st)
                print(f"pod ready (direct): ssh root@{p['ip']} -p {p['publicPort']}{accrued(st)}")
                return 0
        host_id = (pod.get("machine") or {}).get("podHostId")
        if rt and host_id:
            st["ssh_host"], st["ssh_port"], st["ssh_user"] = "ssh.runpod.io", 22, host_id
            save(st)
            print(f"pod ready (proxy): ssh {host_id}@ssh.runpod.io{accrued(st)}")
            return 0
        print(f"  {pod.get('desiredStatus')}, runtime={'up' if rt else 'still starting'}, waiting...")
        time.sleep(15)
    print(f"pod not reachable within {wait_min} min. It is STILL RUNNING AND BILLING - "
          f"run `down` if you are not going to use it.")
    return 1


def cmd_push(a) -> int:
    st = load()
    if not st.get("ssh_host"):
        sys.exit("no live pod with SSH; run `up` first")
    data = Path(a.data) if Path(a.data).is_absolute() else REPO / a.data
    if not (data / "data.yaml").exists():
        sys.exit(f"{data}/data.yaml not found - build the tiled dataset first with\n"
                 f"    uv run python -m sightline.detect dataset <run> --out {a.data}")
    arch = REPO / "_artifacts" / "yolo_upload.tar.gz"
    print(f"packing {data} -> {arch}")
    shutil.make_archive(str(arch.with_suffix("").with_suffix("")), "gztar",
                        root_dir=str(data.parent), base_dir=data.name)
    mb = arch.stat().st_size / 1e6
    print(f"uploading {mb:.0f} MB{accrued(st)}")
    ssh(st, f"mkdir -p {REMOTE}")
    subprocess.run(["scp", *scp_target(st), str(arch),
                    f"{st.get('ssh_user', 'root')}@{st['ssh_host']}:{REMOTE}/data.tar.gz"], check=True)
    ssh(st, f"cd {REMOTE} && tar xzf data.tar.gz && rm data.tar.gz && ls")
    st["data_dir"] = data.name
    save(st)
    return 0


def cmd_train(a) -> int:
    st = load()
    if not st.get("ssh_host"):
        sys.exit("no live pod with SSH; run `up` first")
    if not st.get("data_dir"):
        sys.exit("nothing pushed yet; run `push` first")
    ssh(st, "pip -q install 'ultralytics==8.4.146' >/dev/null 2>&1; python -c "
            "\"import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),"
            "torch.cuda.get_device_name(0))\"")
    # `nohup ... &` so a dropped SSH session does not kill a run that is being paid for by the second.
    # Hyperparameters chosen for what this dataset IS: a few hundred instances of ONE class in a narrow,
    # fixed domain, fine-tuned from COCO. The binding limit is DATA, not compute or capacity - 60 epochs
    # here is ~15 min on a 4090 - so the settings buy variety out of few boxes rather than buying capacity:
    #   mosaic       stitches four images, so each step sees four contexts and more small targets at once
    #   copy_paste   pastes instances between images; the single most useful augmentation when the object
    #                count is the scarce thing, which it is here
    #   scale/fliplr the survivor may be at any GSD and any heading; both are real invariances of the task
    #   degrees=180  a nadir camera has NO canonical up, so full rotation is honest rather than a stretch
    #   patience     early stopping, because a small set overfits long before the epoch budget runs out
    #   close_mosaic turns mosaic off for the last epochs so the model finishes on undistorted frames
    aug = (f"mosaic=1.0 copy_paste={a.copy_paste} scale=0.5 fliplr=0.5 flipud=0.5 degrees=180 "
           f"hsv_h=0.015 hsv_s=0.5 hsv_v=0.4 translate=0.1 close_mosaic=10 patience={a.patience}")
    cmd = (f"cd {REMOTE} && nohup yolo detect train "
           f"model={a.model} data={REMOTE}/{st['data_dir']}/data.yaml "
           f"epochs={a.epochs} imgsz={a.imgsz} batch={a.batch} cache=disk workers=8 {aug} "
           f"project={REMOTE}/runs name=f8b_sim exist_ok=True plots=True "
           f"> {REMOTE}/train.log 2>&1 & echo started")
    ssh(st, cmd)
    print(f"training started on the pod{accrued(st)}\n"
          f"  follow it:  uv run python tools/train/runpod_train.py logs\n"
          f"  fetch it:   uv run python tools/train/runpod_train.py pull\n"
          f"  STOP PAYING: uv run python tools/train/runpod_train.py down")
    return 0


def cmd_logs(a) -> int:
    st = load()
    if not st.get("ssh_host"):
        sys.exit("no live pod")
    print(accrued(st).strip())
    return ssh(st, f"tail -n {a.lines} {REMOTE}/train.log", check=False)


def cmd_pull(a) -> int:
    st = load()
    if not st.get("ssh_host"):
        sys.exit("no live pod")
    out = REPO / "models" / "detect"
    out.mkdir(parents=True, exist_ok=True)
    ssh(st, f"cd {REMOTE}/runs && tar czf {REMOTE}/results.tar.gz f8b_sim", check=False)
    subprocess.run(["scp", *scp_target(st), f"{st.get('ssh_user', 'root')}@{st['ssh_host']}:{REMOTE}/results.tar.gz",
                    str(out / "results.tar.gz")], check=True)
    shutil.unpack_archive(str(out / "results.tar.gz"), str(out))
    (out / "results.tar.gz").unlink(missing_ok=True)
    print(f"pulled to {out / 'f8b_sim'}{accrued(st)}")
    print("REMEMBER: `down` terminates the pod and stops the billing.")
    return 0


def cmd_down(a) -> int:
    st = load()
    if not st.get("pod_id"):
        print("no pod recorded. Checking the account for stragglers anyway...")
        return cmd_status(a)
    gql("mutation($id:String!){ podTerminate(input:{podId:$id}) }", {"id": st["pod_id"]})
    print(f"terminated pod {st['pod_id']}{accrued(st)}")
    STATE.unlink(missing_ok=True)
    return 0


def cmd_status(a) -> int:
    st = load()
    print(f"balance ${balance():.2f}")
    d = gql("query { myself { pods { id name desiredStatus costPerHr machine { gpuDisplayName } "
            "runtime { uptimeInSeconds } } } }")
    pods = d["myself"]["pods"] or []
    if not pods:
        print("no pods on the account - nothing is billing")
    for p in pods:
        up = ((p.get("runtime") or {}).get("uptimeInSeconds") or 0) / 3600.0
        print(f"  {p['id']}  {p['name']}  {p['desiredStatus']}  "
              f"{(p.get('machine') or {}).get('gpuDisplayName')}  ${p.get('costPerHr', 0):.3f}/h  "
              f"up {up * 60:.0f} min  (~${up * (p.get('costPerHr') or 0):.2f})")
    if st.get("pod_id"):
        print(f"local state: {st['pod_id']} {st.get('gpu')}{accrued(st)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--data", default=""); p.add_argument("--epochs", type=int, default=40); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("up")
    p.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    # SECURE, not COMMUNITY, and the reason is SSH. Measured 2026-09-11: a community-cloud 4090 came up
    # fine but exposed NO ssh port at all - only an internal http port with `isIpPublic: false`. RunPod's
    # other route, the `<podHostId>@ssh.runpod.io` proxy, authenticates against the ACCOUNT's registered
    # public key rather than the per-pod PUBLIC_KEY, so it rejects a key this tool generated. Secure cloud
    # exposes a real public TCP port, which is where the injected PUBLIC_KEY works.
    p.add_argument("--cloud", default="SECURE", choices=("COMMUNITY", "SECURE", "ALL"))
    p.add_argument("--disk-gb", type=int, default=60)
    p.add_argument("--wait-min", type=float, default=25.0)   # a cold ~20 GB image pull is slow
    p.set_defaults(fn=cmd_up)
    p = sub.add_parser("wait", help="resume waiting for a pod that is already up (e.g. a slow image pull)")
    p.add_argument("--wait-min", type=float, default=25.0)
    p.set_defaults(fn=lambda a: wait_ready(load(), a.wait_min) if load().get("pod_id")
                   else sys.exit("no pod recorded"))
    p = sub.add_parser("push"); p.add_argument("--data", default="_artifacts/yolo/sim"); p.set_defaults(fn=cmd_push)
    p = sub.add_parser("train")
    p.add_argument("--model", default="yolo26s.pt")
    p.add_argument("--epochs", type=int, default=80)   # cheap here; early stopping decides the real number
    p.add_argument("--imgsz", type=int, default=1024)  # matches the tile size exactly - no resample
    p.add_argument("--batch", type=int, default=-1)
    p.add_argument("--copy-paste", type=float, default=0.3,
                   help="the most valuable augmentation when INSTANCES are the scarce resource")
    p.add_argument("--patience", type=int, default=20,
                   help="early stopping; a few hundred boxes overfit well before the epoch budget")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("logs"); p.add_argument("--lines", type=int, default=40); p.set_defaults(fn=cmd_logs)
    p = sub.add_parser("pull"); p.set_defaults(fn=cmd_pull)
    p = sub.add_parser("down"); p.set_defaults(fn=cmd_down)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
