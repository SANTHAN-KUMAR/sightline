"""F12: one living being, one record — across frames, passes and re-visits (SOLUTION_DOC §5.6, §5.8).

    "Every confirmed track yields observations (lat, lon, sigma, conf, frame). Take the weighted-median position
     and the per-track CE90 from the geolocation budget for that geometry. Cluster all tracks from this and
     previous passes with DBSCAN at radius 2 x CE90 ... One cluster = one record."

Why the identity lives in **geo space** and not in the tracker: a 20-60 px blob has no usable appearance, so a
re-visit two minutes later cannot be re-identified from pixels. Position can. The consequence, which is the whole
value of this stage, is that **tracker ID switches stop mattering for the output**: two fragments of one survivor
land in one cluster and become one record.

## Guardrail R10, in this module specifically

There is no delete and no "cleared". Concretely:

* `Deduplicator` never removes a `Record` from its registry. `all_records()` only ever grows.
* A record whose cell is re-imaged without the record being seen again becomes `status = "stale"`. It keeps its
  id, its position, its evidence and its history.
* When two records turn out to be the same being (a later track bridges them), the **older** record survives and
  the newer is marked `status = "dismissed"` with `dismissed_reason = "merged_into:<record_id>"` and
  `source["merged_into"]`. It stays in the registry and in the export, on its own layer, per §5.8.
* **Record ids are never renumbered.** A second pass merges into the existing record; it does not mint a new one.

## Deviations from the letter of §5.6, all deliberate and all switchable

1. `motion_state` is `"unknown"`, not `"still"`, when no member track was observed over a window of at least
   `motion_window_s`. §5.6 says "moving if ... else still", but a track seen for 1.2 s has not earned the claim
   that its subject is stationary. Set `still_when_window_short=True` for the doc's literal behaviour.
2. The record's `h_acc_m` is the **best** member track's, never a `1/sqrt(N)` combination. §5.7 is explicit that
   averaging shrinks the random terms but not the yaw and boresight biases, so combining tracks must not be
   allowed to invent accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sightline.common import geodesy
from sightline.schemas import CE90_FACTOR, Evidence, Observation, Record, Track, ce90_m

#: How a single clustering radius is chosen when member tracks disagree about their accuracy.
EpsPolicy = Literal["median", "max", "mean", "min"]


@dataclass(slots=True)
class DedupConfig:
    """§5.6's deduplication parameters. Defaults are the doc's."""

    #: DBSCAN radius = `eps_ce90_multiple` x CE90. §5.6: "DBSCAN at radius 2 x CE90 (about 6-8 m at 60 m nadir
    #: with consumer GNSS; ~10 m oblique)".
    eps_ce90_multiple: float = 2.0
    #: Which per-track `h_acc_m` becomes the cluster radius when tracks disagree. The median is robust to a
    #: single bad geometry; `max` merges more aggressively, `min` less.
    eps_policy: EpsPolicy = "median"
    #: Hard floor and ceiling on the radius in metres, so one absurd `h_acc_m` cannot collapse or explode the map.
    eps_min_m: float = 1.0
    eps_max_m: float = 50.0
    #: Fixed radius in metres, overriding everything above. For ablations and for the evaluation lane.
    eps_override_m: float | None = None
    #: §5.6: "motion.state = moving if any member track's geo-displacement over >= 5 s exceeds 3 x CE90".
    motion_window_s: float = 5.0
    motion_ce90_multiple: float = 3.0
    #: See the module docstring, deviation 1.
    still_when_window_short: bool = False
    #: Tracks whose observations carry no valid `GeoFix` cannot be clustered. Raise instead of dropping silently.
    raise_on_unlocated: bool = False
    #: A record is only created from a track that passed §5.6 rule 3. Unconfirmed tracks may be admitted for
    #: debugging; they produce `status = "candidate"` records.
    accept_unconfirmed: bool = False


@dataclass(slots=True)
class TrackSummary:
    """One confirmed track reduced to the five numbers dedup actually clusters on."""

    key: tuple[str, int, int]  # (clip_id, pass_id, track_id) — unique across passes and clips
    track: Track
    lat: float
    lon: float
    h_acc_m: float
    conf: float
    t_first: float
    t_last: float
    pass_ids: tuple[int, ...]
    n_observations: int
    #: Largest geo displacement seen between two observations at least `motion_window_s` apart, and its span.
    max_displacement_m: float = 0.0
    displacement_window_s: float = 0.0

    @property
    def ce90_m(self) -> float:
        return ce90_m(self.h_acc_m)


def _valid_observations(track: Track) -> list[Observation]:
    return [o for o in track.observations if o.fix.valid and math.isfinite(o.fix.lat) and math.isfinite(o.fix.lon)]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def summarise_track(track: Track, cfg: DedupConfig) -> TrackSummary | None:
    """Reduce a track to its clustering position and accuracy, or `None` when it cannot be located.

    Position is the **confidence-weighted median** of the observations (§5.6), which `Track` already implements;
    it is recomputed here over the *valid* fixes only, because an invalid fix carries the aircraft's position and
    would drag the median off the target.
    """
    obs = _valid_observations(track)
    if not obs:
        if cfg.raise_on_unlocated:
            raise ValueError(f"track {track.track_id} has no valid GeoFix; run the geolocation stage (F13) first")
        return None

    lats = np.array([o.fix.lat for o in obs], dtype=float)
    lons = np.array([o.fix.lon for o in obs], dtype=float)
    w = np.array([max(o.det.score, 1e-6) for o in obs], dtype=float)
    lat = _weighted_median(lats, w)
    lon = _weighted_median(lons, w)

    accs = np.array([o.fix.h_acc_m for o in obs], dtype=float)
    accs = accs[np.isfinite(accs)]
    # The track's representative accuracy is the median of its observations': robust to the one frame taken at
    # the edge of the swath, and never better than a single real measurement (§5.7's bias floor).
    h_acc = float(np.median(accs)) if accs.size else float("inf")

    times = np.array([o.t_utc for o in obs], dtype=float)
    disp, window = _max_displacement(lats, lons, times, cfg.motion_window_s)
    pass_ids = tuple(sorted({o.pass_id for o in obs}))
    clip = track.clip_id or (obs[0].clip_id if obs else "")
    return TrackSummary(
        key=(clip, pass_ids[0] if pass_ids else 0, track.track_id),
        track=track,
        lat=lat,
        lon=lon,
        h_acc_m=h_acc,
        conf=float(max(o.det.score for o in obs)),
        t_first=float(times.min()),
        t_last=float(times.max()),
        pass_ids=pass_ids,
        n_observations=len(obs),
        max_displacement_m=disp,
        displacement_window_s=window,
    )


def _max_displacement(
    lats: np.ndarray, lons: np.ndarray, times: np.ndarray, min_window_s: float
) -> tuple[float, float]:
    """Largest distance between two observations separated by at least `min_window_s`, and that separation.

    §5.6 measures motion over a window rather than between consecutive frames because per-frame geolocation
    noise (metres) dwarfs a survivor's per-frame movement (centimetres).
    """
    n = len(times)
    if n < 2:
        return 0.0, 0.0
    order = np.argsort(times)
    lats, lons, times = lats[order], lons[order], times[order]
    best_d, best_w = 0.0, 0.0
    j = 0
    for i in range(n):
        # first index whose time is at least min_window_s after times[i]
        while j < n and times[j] - times[i] < min_window_s:
            j += 1
        for k in range(j, n):
            d = geodesy.haversine_m(lats[i], lons[i], lats[k], lons[k])
            if d > best_d:
                best_d, best_w = d, float(times[k] - times[i])
    return best_d, best_w


def _cluster(summaries: Sequence[TrackSummary], eps_m: float) -> np.ndarray:
    """`sklearn.cluster.DBSCAN(eps = 2 x CE90, min_samples = 1, metric = 'haversine')`, per §5.6.

    sklearn's haversine metric works in **radians** and returns radians, so the metre radius is divided by the
    Earth radius at this latitude (`geodesy.earth_radius_m`) before it is handed over. Getting this wrong by the
    6.37e6 factor either merges the whole map into one record or merges nothing at all.
    """
    from sklearn.cluster import DBSCAN

    lat = np.array([s.lat for s in summaries], dtype=float)
    lon = np.array([s.lon for s in summaries], dtype=float)
    eps_rad = eps_m / geodesy.earth_radius_m(float(np.mean(lat)))
    labels = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit_predict(geodesy.radians_stack(lat, lon))
    return np.asarray(labels, dtype=int)


class Deduplicator:
    """The record registry. Feed it confirmed tracks, one pass at a time; it hands back records.

        dedup = Deduplicator()
        records = dedup.ingest(tracker_pass0.close())          # pass 0
        records = dedup.ingest(tracker_pass1.close())          # pass 1 merges into the SAME record ids
        dedup.mark_stale(pass_id=1, was_reimaged=Deduplicator.radius_predicate(lat, lon, 100.0))

    `ingest` is idempotent in the sense that re-ingesting a track it already holds replaces that track's
    contribution rather than doubling it: tracks are keyed by `(clip_id, pass_id, track_id)`.
    """

    def __init__(self, cfg: DedupConfig | None = None) -> None:
        self.cfg = cfg or DedupConfig()
        self._summaries: dict[tuple[str, int, int], TrackSummary] = {}
        self._records: dict[str, Record] = {}
        self._order: list[str] = []  # record ids in creation order; never reordered, never shortened
        self._order_index: dict[str, int] = {}
        self._track_to_record: dict[tuple[str, int, int], str] = {}
        self._next_cluster_id = 0
        self.last_eps_m: float = float("nan")
        self.skipped_unlocated: list[int] = []

    # --- ingestion ----------------------------------------------------------------------------------------
    def ingest(self, tracks: Iterable[Track]) -> list[Record]:
        """Add a pass's confirmed tracks and return the **active** records (creation order, ids stable)."""
        for track in tracks:
            if not track.confirmed and not self.cfg.accept_unconfirmed:
                continue
            summary = summarise_track(track, self.cfg)
            if summary is None:
                self.skipped_unlocated.append(track.track_id)
                continue
            self._summaries[summary.key] = summary
        self._recluster()
        return self.active_records()

    def _eps_m(self) -> float:
        if self.cfg.eps_override_m is not None:
            return float(self.cfg.eps_override_m)
        accs = np.array([s.h_acc_m for s in self._summaries.values() if math.isfinite(s.h_acc_m)], dtype=float)
        if accs.size == 0:
            return self.cfg.eps_max_m
        agg = {
            "median": np.median,
            "max": np.max,
            "mean": np.mean,
            "min": np.min,
        }[self.cfg.eps_policy](accs)
        eps = self.cfg.eps_ce90_multiple * ce90_m(float(agg))
        return float(min(max(eps, self.cfg.eps_min_m), self.cfg.eps_max_m))

    def _recluster(self) -> None:
        """Re-run DBSCAN over every track ever seen and re-attach clusters to their existing record ids."""
        if not self._summaries:
            return
        keys = sorted(self._summaries)  # deterministic order in, deterministic labels out
        summaries = [self._summaries[k] for k in keys]
        eps_m = self._eps_m()
        self.last_eps_m = eps_m
        labels = _cluster(summaries, eps_m)

        clusters: dict[int, list[TrackSummary]] = {}
        for label, s in zip(labels, summaries, strict=True):
            clusters.setdefault(int(label), []).append(s)

        claimed: set[str] = set()
        for label in sorted(clusters):
            members = clusters[label]
            record = self._claim_record(members, claimed)
            claimed.add(record.record_id)
            self._fill_record(record, members, eps_m)
            for s in members:
                self._track_to_record[s.key] = record.record_id

    def _claim_record(self, members: list[TrackSummary], claimed: set[str]) -> Record:
        """Find which existing record this cluster IS, or mint one. Ids are never reassigned or renumbered.

        Three cases:
          * no member track has a record yet -> a new record;
          * exactly one record is represented -> that record, updated in place;
          * several -> the **oldest** survives and the others are dismissed as merged into it (R10: they stay).
        """
        existing: list[str] = []
        for s in members:
            rid = self._track_to_record.get(s.key)
            if rid is not None and rid not in claimed and rid not in existing:
                existing.append(rid)
        if not existing:
            record = Record(cluster_id=self._next_cluster_id)
            self._next_cluster_id += 1
            self._records[record.record_id] = record
            self._order_index[record.record_id] = len(self._order)
            self._order.append(record.record_id)
            return record

        existing.sort(key=lambda rid: (self._records[rid].first_seen_utc, self._order_index[rid]))
        survivor = self._records[existing[0]]
        for rid in existing[1:]:
            other = self._records[rid]
            if other.status == "dismissed" and other.source.get("merged_into") == survivor.record_id:
                continue
            other.status = "dismissed"
            other.dismissed_reason = f"merged_into:{survivor.record_id}"
            other.dismissed_by = "dedup"
            other.dismissed_utc = max(other.last_seen_utc, survivor.last_seen_utc)
            other.source["merged_into"] = survivor.record_id
            other.version += 1
            merged_from = survivor.source.setdefault("merged_from", [])
            if rid not in merged_from:
                merged_from.append(rid)
        return survivor

    def _fill_record(self, record: Record, members: list[TrackSummary], eps_m: float) -> None:
        """Rebuild every derived field of a record from its member tracks (§5.6 dedup mechanism, §5.8 schema)."""
        lats = np.array([s.lat for s in members], dtype=float)
        lons = np.array([s.lon for s in members], dtype=float)
        accs = np.array([s.h_acc_m for s in members], dtype=float)

        # Inverse-variance weighted mean position ...
        w = np.where(np.isfinite(accs) & (accs > 0), 1.0 / np.maximum(accs, 1e-6) ** 2, 0.0)
        if not np.any(w > 0):
            w = np.ones_like(accs)
        record.lat = float(np.average(lats, weights=w))
        record.lon = float(np.average(lons, weights=w))
        # ... but NOT an inverse-variance accuracy: §5.7's yaw and boresight biases are common to every track in
        # a pass, so combining tracks may not claim better than the best single geometry.
        finite = accs[np.isfinite(accs)]
        record.h_acc_m = float(finite.min()) if finite.size else float("inf")

        all_obs = [o for s in members for o in _valid_observations(s.track)]
        best = max(all_obs, key=lambda o: o.det.score) if all_obs else None

        record.cls = members[0].track.cls
        record.n_tracks_merged = len(members)
        record.n_observations = sum(s.n_observations for s in members)
        record.first_seen_utc = float(min(s.t_first for s in members))
        record.last_seen_utc = float(max(s.t_last for s in members))
        record.seen_in_passes = sorted({p for s in members for p in s.pass_ids})

        # §5.6: confidence = 1 - prod(1 - conf_track). Tracks are treated as independent evidence.
        record.confidence = float(1.0 - np.prod([1.0 - min(max(s.conf, 0.0), 1.0) for s in members]))
        record.confidence_max_det = float(max(s.conf for s in members))

        record.count_estimate, record.count_max = self._count(members)
        record.count_min = 1
        record.count_basis = "max_simultaneous_tracks"

        record.motion_state, record.motion_displacement_m, record.motion_window_s = self._motion(members)

        if best is not None:
            record.alt_msl_m = best.fix.alt_msl_m
            record.h_acc_basis = best.fix.h_acc_basis
            record.method = best.fix.method
            record.dem_source = best.fix.dem_source
            record.agl_m = best.fix.agl_m
            record.off_nadir_deg = best.fix.off_nadir_deg
            record.modality = best.det.modality
            record.pixel_size_px = float(best.det.size_px)
            record.thermal_hot = any(o.det.thermal_hot for o in all_obs)
            hot = [o.det.thermal_c for o in all_obs if o.det.thermal_c is not None]
            record.thermal_c = float(max(hot)) if hot else None
            record.posture, record.posture_conf = _best_attribute(all_obs, "posture", "posture_conf")
            record.submersion, record.submersion_conf = _best_attribute(all_obs, "submersion", "submersion_conf")
            occl = [o.det.occlusion for o in all_obs if o.det.occlusion is not None]
            record.occlusion = int(min(occl)) if occl else None
            record.evidence = [
                Evidence(
                    thumb_uri="",  # filled by the export lane (F14/F17); never invented here
                    clip_id=best.clip_id,
                    frame_idx=best.frame_idx,
                    frame_time_utc=best.t_utc,
                    bbox_px=best.det.bbox_px,
                    det_conf=best.det.score,
                    camera=best.det.modality,
                )
            ]

        if record.status not in ("dismissed",):
            confirmed = any(s.track.confirmed for s in members)
            record.status = "confirmed" if confirmed else "candidate"
        record.source["dedup_eps_m"] = eps_m
        record.source["dedup_eps_ce90_multiple"] = self.cfg.eps_ce90_multiple
        record.source["track_keys"] = [list(s.key) for s in members]
        record.version += 1

    def _count(self, members: list[TrackSummary]) -> tuple[int, int]:
        """§5.6: `count_estimate` = the maximum number of **simultaneous** distinct tracks inside the cluster.

        Simultaneity is measured per FRAME (`clip_id`, `frame_idx`), not per time interval, and that choice is
        what makes the rule work: two fragments of one survivor caused by an ID switch never share a frame, so
        they count once; two people standing on the same roof do share frames, so they count twice.

        The interval-overlap count is returned as `count_max` — the looser upper bound, kept because tracks from
        different clips (two aircraft over the same roof) never share a frame key and would otherwise read 1.
        """
        per_frame: dict[tuple[str, int], set[int]] = {}
        for s in members:
            for o in _valid_observations(s.track):
                per_frame.setdefault((o.clip_id, o.frame_idx), set()).add(o.track_id)
        estimate = max((len(v) for v in per_frame.values()), default=1)

        events: list[tuple[float, int]] = []
        for s in members:
            events.append((s.t_first, 1))
            events.append((s.t_last, -1))
        events.sort(key=lambda e: (e[0], -e[1]))
        live = overlap = 0
        for _, delta in events:
            live += delta
            overlap = max(overlap, live)
        return max(1, estimate), max(1, estimate, overlap)

    def _motion(self, members: list[TrackSummary]) -> tuple[str, float, float]:
        """§5.6: "moving" if any member track's geo-displacement over >= 5 s exceeds 3 x CE90, else "still"."""
        best_d, best_w = 0.0, 0.0
        moving = False
        observed_window = False
        for s in members:
            if s.displacement_window_s >= self.cfg.motion_window_s:
                observed_window = True
                if s.max_displacement_m > self.cfg.motion_ce90_multiple * s.ce90_m:
                    moving = True
            if s.max_displacement_m > best_d:
                best_d, best_w = s.max_displacement_m, s.displacement_window_s
        if moving:
            return "moving", best_d, best_w
        if observed_window or self.cfg.still_when_window_short:
            return "still", best_d, best_w
        return "unknown", best_d, best_w

    # --- staleness (R10: never a delete) ------------------------------------------------------------------
    def mark_stale(self, *, pass_id: int, was_reimaged: Callable[[Record], bool] | None = None) -> list[Record]:
        """§5.6: "A record not re-observed when its cell is re-imaged becomes `stale`, never deleted."

        Args:
            pass_id: the pass that has just finished.
            was_reimaged: `Record -> bool`, true when this record's location was covered by `pass_id`. The
                coverage lane (F16) owns the real answer; `bbox_predicate` / `radius_predicate` cover the simple
                cases. `None` means "the whole area was re-imaged".

        Returns the records whose status changed. A record seen again in `pass_id` is restored to "confirmed" —
        staleness is a statement about the last look, not a permanent mark.
        """
        changed: list[Record] = []
        for rid in self._order:
            record = self._records[rid]
            if record.status == "dismissed":
                continue
            seen_now = pass_id in record.seen_in_passes
            covered = True if was_reimaged is None else bool(was_reimaged(record))
            if seen_now and record.status == "stale":
                record.status = "confirmed"
                record.version += 1
                changed.append(record)
            elif not seen_now and covered and record.status != "stale":
                record.status = "stale"
                record.notes = (record.notes + " " if record.notes else "") + f"not re-observed in pass {pass_id}"
                record.version += 1
                changed.append(record)
        return changed

    @staticmethod
    def radius_predicate(lat: float, lon: float, radius_m: float) -> Callable[[Record], bool]:
        """"Everything within `radius_m` of this point was re-imaged"."""

        def inside(record: Record) -> bool:
            return geodesy.haversine_m(lat, lon, record.lat, record.lon) <= radius_m

        return inside

    @staticmethod
    def bbox_predicate(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> Callable[[Record], bool]:
        def inside(record: Record) -> bool:
            return lat_min <= record.lat <= lat_max and lon_min <= record.lon <= lon_max

        return inside

    # --- access -------------------------------------------------------------------------------------------
    def all_records(self) -> list[Record]:
        """Every record ever created, in creation order, including dismissed ones. This list only grows (R10)."""
        return [self._records[rid] for rid in self._order]

    def active_records(self) -> list[Record]:
        """The records a commander should see: everything except the ones dismissed as duplicates."""
        return [r for r in self.all_records() if r.status != "dismissed"]

    def record_for_track(self, key: tuple[str, int, int]) -> Record | None:
        rid = self._track_to_record.get(key)
        return self._records.get(rid) if rid else None

    def stats(self) -> dict[str, object]:
        active = self.active_records()
        return {
            "n_tracks": len(self._summaries),
            "n_records_total": len(self._order),
            "n_records_active": len(active),
            "n_records_dismissed": len(self._order) - len(active),
            "n_stale": sum(1 for r in active if r.status == "stale"),
            "eps_m": self.last_eps_m,
            "eps_ce90_multiple": self.cfg.eps_ce90_multiple,
            "ce90_factor": CE90_FACTOR,
            "skipped_unlocated_tracks": len(self.skipped_unlocated),
        }


def _best_attribute(observations: list[Observation], attr: str, conf_attr: str) -> tuple[str, float]:
    """Take the attribute of the observation that was most confident about it, not of the biggest box."""
    best_val, best_conf = "unknown", 0.0
    for o in observations:
        val = getattr(o.det, attr)
        conf = float(getattr(o.det, conf_attr))
        if val != "unknown" and conf >= best_conf:
            best_val, best_conf = val, conf
    return best_val, best_conf


def records_from_tracks(tracks: Iterable[Track], cfg: DedupConfig | None = None) -> list[Record]:
    """One-shot convenience for a single pass. Use `Deduplicator` when there is more than one."""
    return Deduplicator(cfg).ingest(tracks)


__all__ = [
    "DedupConfig",
    "Deduplicator",
    "EpsPolicy",
    "TrackSummary",
    "records_from_tracks",
    "summarise_track",
]
