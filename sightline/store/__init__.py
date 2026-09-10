"""F18: the local record log and the offline outbox (SOLUTION_DOC §5.8, §5.10).

Guardrail R10 lives here: this package contains no ``DELETE`` and no ``DROP`` on any record table, and the
schema installs BEFORE DELETE triggers that abort one anyway. Dismissal is ``status = "dismissed"`` with a
non-empty reason; the row and every earlier version stay in the log.
"""

from sightline.store.db import (
    COMPONENT_COLUMNS,
    RECORD_COLUMNS,
    STATUSES,
    GuardrailError,
    RecordStore,
    StaleVersionError,
)
from sightline.store.outbox import HttpTransport, Outbox, Transport, Uploader, clip_id_of, job_key

__all__ = [
    "RecordStore",
    "Outbox",
    "Uploader",
    "HttpTransport",
    "Transport",
    "job_key",
    "clip_id_of",
    "GuardrailError",
    "StaleVersionError",
    "RECORD_COLUMNS",
    "COMPONENT_COLUMNS",
    "STATUSES",
]
