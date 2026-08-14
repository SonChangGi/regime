"""Public data-layer API for collectors, snapshots, and PIT joins."""

from .alfred import (
    ALFRED_RIGHTS_ACK_ENV,
    FRED_API_KEY_ENV,
    AlfredClient,
    AlfredConfig,
)
from .alpha_vantage import (
    ALPHA_VANTAGE_API_KEY_ENV,
    AlphaVantageClient,
    AlphaVantageConfig,
    DailyRequestBudget,
)
from .asof import AsOfValue, weekly_asof_frame, weekly_asof_join
from .contracts import (
    CollectionResult,
    HealthStatus,
    Observation,
    PreparedSnapshot,
    RealtimeCollectionWindow,
    SnapshotMode,
    SnapshotProvenance,
    combine_health,
    merge_collection_results,
    normalize_revision_sequences,
    observation_natural_key,
    plan_incremental_realtime_window,
    prepare_incremental_snapshot,
    provenance_safe_result,
    snapshot_mode_from_provenance,
)
from .security import REDACTED, redact_text, sanitize_mapping
from .store import SQLiteSnapshotStore
from .transport import (
    HttpStatusError,
    JsonTransport,
    ProviderRequestError,
    RetryPolicy,
    UrllibJsonTransport,
)

__all__ = [
    "ALFRED_RIGHTS_ACK_ENV",
    "ALPHA_VANTAGE_API_KEY_ENV",
    "FRED_API_KEY_ENV",
    "AlfredClient",
    "AlfredConfig",
    "AlphaVantageClient",
    "AlphaVantageConfig",
    "AsOfValue",
    "CollectionResult",
    "DailyRequestBudget",
    "HealthStatus",
    "HttpStatusError",
    "JsonTransport",
    "Observation",
    "PreparedSnapshot",
    "ProviderRequestError",
    "REDACTED",
    "RetryPolicy",
    "RealtimeCollectionWindow",
    "SQLiteSnapshotStore",
    "SnapshotProvenance",
    "SnapshotMode",
    "UrllibJsonTransport",
    "combine_health",
    "merge_collection_results",
    "normalize_revision_sequences",
    "observation_natural_key",
    "plan_incremental_realtime_window",
    "prepare_incremental_snapshot",
    "provenance_safe_result",
    "redact_text",
    "sanitize_mapping",
    "snapshot_mode_from_provenance",
    "weekly_asof_frame",
    "weekly_asof_join",
]
