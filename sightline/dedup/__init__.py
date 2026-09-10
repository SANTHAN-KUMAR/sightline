"""F12 deduplication: one living being, one record, across passes (SOLUTION_DOC §5.6, §5.8).

    from sightline.dedup import Deduplicator
    dedup = Deduplicator()
    records = dedup.ingest(tracker.close())          # pass 0
    records = dedup.ingest(tracker2.close())         # pass 1 merges into the SAME record ids
    dedup.mark_stale(pass_id=1)                      # unseen records go "stale" — never deleted (R10)

Owned by lane B3 (see docs/CONTRACTS.md). Consumes `Track`, produces `Record`.
"""

from sightline.dedup.cluster import (
    DedupConfig,
    Deduplicator,
    EpsPolicy,
    TrackSummary,
    records_from_tracks,
    summarise_track,
)
from sightline.dedup.metrics import (
    DEFAULT_MATCH_RADIUS_M,
    MOT_METRICS,
    DedupAccuracy,
    DedupMatch,
    FrameAssociations,
    GroundTruthSurvivor,
    dedup_accuracy,
    fp_per_min,
    frames_from_tracks,
    track_metrics,
)

__all__ = [
    "DEFAULT_MATCH_RADIUS_M",
    "MOT_METRICS",
    "DedupAccuracy",
    "DedupConfig",
    "DedupMatch",
    "Deduplicator",
    "EpsPolicy",
    "FrameAssociations",
    "GroundTruthSurvivor",
    "TrackSummary",
    "dedup_accuracy",
    "fp_per_min",
    "frames_from_tracks",
    "records_from_tracks",
    "summarise_track",
    "track_metrics",
]
