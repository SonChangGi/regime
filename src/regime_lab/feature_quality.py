"""Deterministic quality diagnostics for the model feature matrix."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_QUALITY_ARTIFACT_PATH = "feature-quality.json"


def _feature_role(name: str) -> str:
    monitoring_markers = (
        "__missing",
        "__is_filled",
        "__coverage",
        "__coverage_1w",
        "__coverage_4w",
        "__revision_event",
        "__prior_event_count",
        "__age_days",
        "__release_lag_days",
    )
    return (
        "availability_or_event_monitor"
        if name.endswith(monitoring_markers)
        else "model_signal"
    )


def _finite_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric))


def _missing_streak(series: pd.Series) -> int:
    count = 0
    for missing in series.isna().iloc[::-1]:
        if not bool(missing):
            break
        count += 1
    return count


def _shift_score(series: pd.Series, *, recent_weeks: int) -> float | None:
    recent = series.iloc[-recent_weeks:].dropna()
    history = series.iloc[: -recent_weeks].tail(156).dropna()
    if len(recent) < 13 or len(history) < 52:
        return None
    q25, q75 = np.quantile(history.to_numpy(), [0.25, 0.75])
    scale = float(q75 - q25)
    if not math.isfinite(scale) or scale <= 1e-12:
        return (
            0.0
            if float(recent.median()) == float(history.median())
            else 999999.0
        )
    return round(abs(float(recent.median()) - float(history.median())) / scale, 6)


def feature_quality_document(
    features: pd.DataFrame,
    *,
    recent_weeks: int = 52,
) -> dict[str, Any]:
    """Summarize missingness, staleness, constants, and distribution shift."""

    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("feature quality requires a non-empty DataFrame")
    if not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise ValueError("feature quality index must be unique and increasing")
    if recent_weeks < 13:
        raise ValueError("recent_weeks must be at least 13")

    rows: list[dict[str, Any]] = []
    warning_count = 0
    unavailable_count = 0
    for name in map(str, features.columns):
        role = _feature_role(name)
        series = _finite_series(features[name])
        recent = series.iloc[-recent_weeks:]
        missing_total = int(series.isna().sum())
        missing_recent = int(recent.isna().sum())
        latest_streak = _missing_streak(series)
        valid = series.dropna()
        last_valid_at = None if valid.empty else pd.Timestamp(valid.index[-1]).isoformat()
        recent_unique = int(recent.dropna().nunique())
        shift = _shift_score(series, recent_weeks=recent_weeks)
        reasons: list[str] = []
        if valid.empty:
            status = "unavailable"
            unavailable_count += 1
            reasons.append("all_missing")
        else:
            if latest_streak > 4:
                reasons.append("latest_missing_streak")
            if len(recent) and missing_recent / len(recent) > 0.2:
                reasons.append("recent_missingness")
            if (
                role == "model_signal"
                and recent.notna().sum() >= 13
                and recent_unique <= 1
            ):
                reasons.append("recent_constant")
            if shift is not None and shift >= 2.0:
                reasons.append("distribution_shift")
            status = "warning" if reasons else "ok"
            warning_count += status == "warning"
        rows.append(
            {
                "feature": name,
                "role": role,
                "status": status,
                "reasons": reasons,
                "non_null_count": int(series.notna().sum()),
                "missing_rate": round(missing_total / len(series), 6),
                "recent_missing_rate": round(
                    missing_recent / len(recent) if len(recent) else 1.0,
                    6,
                ),
                "latest_missing_streak_weeks": latest_streak,
                "last_valid_at": last_valid_at,
                "recent_unique_values": recent_unique,
                "distribution_shift_iqr": shift,
            }
        )
    body: dict[str, Any] = {
        "schema_version": 1,
        "row_count": int(len(features)),
        "feature_count": int(features.shape[1]),
        "index_start": pd.Timestamp(features.index[0]).isoformat(),
        "index_end": pd.Timestamp(features.index[-1]).isoformat(),
        "recent_weeks": recent_weeks,
        "status": "warning" if warning_count or unavailable_count else "ok",
        "warning_feature_count": int(warning_count),
        "unavailable_feature_count": int(unavailable_count),
        "features": rows,
    }
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    body["sha256"] = hashlib.sha256(canonical).hexdigest()
    return body


def canonical_feature_quality_json_bytes(document: dict[str, Any]) -> bytes:
    """Serialize a self-authenticating feature-quality document deterministically."""

    body = dict(document)
    supplied = body.pop("sha256", None)
    canonical_body = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected = hashlib.sha256(canonical_body).hexdigest()
    if supplied != expected:
        raise ValueError("feature quality document sha256 is inconsistent")
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


def feature_quality_artifact_manifest(
    document: dict[str, Any],
) -> dict[str, Any]:
    payload = canonical_feature_quality_json_bytes(document)
    return {
        "path": FEATURE_QUALITY_ARTIFACT_PATH,
        "row_count": int(document["row_count"]),
        "feature_count": int(document["feature_count"]),
        "status": str(document["status"]),
        "warning_feature_count": int(document["warning_feature_count"]),
        "unavailable_feature_count": int(document["unavailable_feature_count"]),
        "content_sha256": str(document["sha256"]),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_feature_quality_artifact(
    manifest: dict[str, Any],
    directory: str | Path,
) -> dict[str, Any]:
    """Bind payload metadata to the exact staged diagnostic JSON bytes."""

    expected_fields = {
        "path",
        "row_count",
        "feature_count",
        "status",
        "warning_feature_count",
        "unavailable_feature_count",
        "content_sha256",
        "sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError("feature quality artifact manifest is invalid")
    if manifest["path"] != FEATURE_QUALITY_ARTIFACT_PATH:
        raise ValueError("feature quality artifact path is invalid")
    path = Path(directory) / FEATURE_QUALITY_ARTIFACT_PATH
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("feature quality artifact is missing/non-regular")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise RuntimeError("feature quality artifact hash mismatch")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("feature quality artifact is invalid JSON") from exc
    if not isinstance(document, dict):
        raise RuntimeError("feature quality artifact must be an object")
    if canonical_feature_quality_json_bytes(document) != payload:
        raise RuntimeError("feature quality artifact is not canonical")
    paired = {
        "row_count": document.get("row_count"),
        "feature_count": document.get("feature_count"),
        "status": document.get("status"),
        "warning_feature_count": document.get("warning_feature_count"),
        "unavailable_feature_count": document.get("unavailable_feature_count"),
        "content_sha256": document.get("sha256"),
    }
    if any(manifest[key] != value for key, value in paired.items()):
        raise RuntimeError("feature quality artifact metadata mismatch")
    return document


__all__ = [
    "FEATURE_QUALITY_ARTIFACT_PATH",
    "canonical_feature_quality_json_bytes",
    "feature_quality_artifact_manifest",
    "feature_quality_document",
    "verify_feature_quality_artifact",
]
