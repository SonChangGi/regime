"""Point-in-time USD-strength features for the isolated v5 research draft."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import math
from typing import Any

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from regime_lab.data.h10 import FIXED_BILATERAL_PANEL, SERIES_CATALOG
from regime_lab.data import Observation


UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")
CORE_DOLLAR_INDEXES: tuple[str, ...] = ("BRD", "AFE", "EME")
FX_MAX_OBSERVATION_AGE_DAYS = 10
FX_FIRST_SEEN_AVAILABILITY_BASIS = "collection_first_seen_at"
FX_ARCHIVE_AVAILABILITY_BASIS = "official_archive_release_schedule"
FX_ARCHIVE_REVISION_POLICY = (
    "later_official_release_preserved_as_new_vintage"
)
FX_ARCHIVE_CORRECTION_AVAILABILITY_BASIS = (
    "date_only_conservative_next_day"
)
FX_ARCHIVE_CORRECTION_QUARANTINE_WEEKS = 27


def _positive_unique_integers(
    values: Sequence[int],
    *,
    field_name: str,
) -> tuple[int, ...]:
    resolved: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} values must be positive integers")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} values must be positive integers"
            ) from exc
        if integer != value or integer < 1:
            raise ValueError(f"{field_name} values must be positive integers")
        resolved.append(integer)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return tuple(resolved)


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class FXFeatureConfig:
    horizons: tuple[int, ...] = (1, 4, 13)
    volatility_windows: tuple[int, ...] = (13, 26)
    core_indexes: tuple[str, ...] = CORE_DOLLAR_INDEXES
    bilateral_panel: tuple[str, ...] = FIXED_BILATERAL_PANEL
    minimum_bilateral_count: int = 6
    annualize_volatility: bool = True

    def __post_init__(self) -> None:
        horizons = _positive_unique_integers(
            self.horizons,
            field_name="horizons",
        )
        windows = _positive_unique_integers(
            self.volatility_windows,
            field_name="volatility_windows",
        )
        core = tuple(str(value).upper() for value in self.core_indexes)
        panel = tuple(str(value).upper() for value in self.bilateral_panel)
        if core != CORE_DOLLAR_INDEXES:
            raise ValueError("core_indexes must be exactly BRD, AFE, EME")
        if not panel or len(set(panel)) != len(panel):
            raise ValueError("bilateral_panel must be non-empty and unique")
        if set(core).intersection(panel):
            raise ValueError("core and bilateral series must not overlap")
        unknown = sorted(set(core + panel).difference(SERIES_CATALOG))
        if unknown:
            raise ValueError(f"unknown H.10 feature series: {unknown}")
        if "KRW" in panel:
            raise ValueError("KRW is intentionally excluded from the fixed panel")
        if not 1 <= self.minimum_bilateral_count <= len(panel):
            raise ValueError("minimum_bilateral_count is outside panel bounds")
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "volatility_windows", windows)
        object.__setattr__(self, "core_indexes", core)
        object.__setattr__(self, "bilateral_panel", panel)


@dataclass(frozen=True, slots=True)
class FXFeatureResult:
    """Numeric features plus non-model availability and quality surfaces."""

    features: pd.DataFrame
    weekly_usd_log_levels: pd.DataFrame
    weekly_availability: pd.DataFrame
    coverage: pd.DataFrame
    status: pd.DataFrame
    official_release_archive_ingest: bool = False
    availability_basis: str = FX_FIRST_SEEN_AVAILABILITY_BASIS
    archive_revision_policy: str = FX_ARCHIVE_REVISION_POLICY
    archive_correction_availability_basis: str = (
        FX_ARCHIVE_CORRECTION_AVAILABILITY_BASIS
    )


def unavailable_fx_context() -> dict[str, Any]:
    """Return the stable public shape when no PIT-safe H.10 row is usable."""

    return {
        "status": "unavailable",
        "method": "fed_h10_usd_strength",
        "bilateral_panel": list(FIXED_BILATERAL_PANEL),
        "coverage": {
            "available_pairs": 0,
            "required_pairs": len(FIXED_BILATERAL_PANEL),
            "available_indexes": 0,
            "required_indexes": len(CORE_DOLLAR_INDEXES),
        },
        "indexes": {},
        "bilateral": {},
        "observation_week": None,
        "feature_available_at": None,
        "direction": "positive_is_usd_appreciation",
    }


def _json_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 8) if math.isfinite(number) else None


def fx_context_at(
    result: FXFeatureResult,
    *,
    cutoff: datetime | pd.Timestamp,
) -> dict[str, Any]:
    """Select the latest derived row that was available by a model cutoff."""

    if not isinstance(result, FXFeatureResult):
        raise TypeError("result must be an FXFeatureResult")
    at = pd.Timestamp(cutoff)
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("cutoff must be timezone-aware")
    at = at.tz_convert("UTC")
    coverage = result.coverage.copy()
    available = pd.to_datetime(
        coverage["feature_available_at"], utc=True, errors="coerce"
    )
    week_index = pd.DatetimeIndex(coverage.index)
    week_index = (
        week_index.tz_localize("UTC")
        if week_index.tz is None
        else week_index.tz_convert("UTC")
    )
    eligible = coverage.index[available.le(at) & (week_index <= at)]
    if len(eligible) == 0:
        return unavailable_fx_context()

    week = pd.Timestamp(eligible[-1])
    feature_row = result.features.loc[week]
    coverage_row = coverage.loc[week]
    feature_status = str(coverage_row["feature_status"])
    public_status = {
        "ok": "ok",
        "partial": "partial",
        # Correction-equivalent quarantine protects retrospective scoring.
        # The latest official H.10 row remains complete current context.
        "correction_quarantine": "ok",
        "warming_up": "insufficient_history",
        "insufficient_coverage": "insufficient_history",
        "unavailable": "unavailable",
    }.get(feature_status, "unavailable")
    observation_age_days = (at.tz_convert(EASTERN).date() - week.date()).days
    if observation_age_days > FX_MAX_OBSERVATION_AGE_DAYS:
        public_status = "stale"

    indexes: dict[str, float | None] = {}
    for code, label in (("brd", "broad"), ("afe", "afe"), ("eme", "eme")):
        for horizon in (1, 4, 13):
            indexes[f"{label}_usd_log_return_{horizon}w"] = _json_number(
                feature_row.get(f"fx__{code}__usd_log_return_{horizon}w")
            )
        for window in (13, 26):
            indexes[f"{label}_realized_vol_{window}w"] = _json_number(
                feature_row.get(f"fx__{code}__realized_vol_{window}w")
            )
    for horizon in (1, 4, 13):
        indexes[f"eme_minus_afe_{horizon}w"] = _json_number(
            feature_row.get(f"fx__eme_minus_afe__usd_log_return_{horizon}w")
        )

    bilateral: dict[str, float | None] = {}
    for horizon in (1, 4, 13):
        bilateral[f"median_usd_log_return_{horizon}w"] = _json_number(
            feature_row.get(
                f"fx__bilateral__median_usd_log_return_{horizon}w"
            )
        )
        bilateral[f"usd_appreciating_share_{horizon}w"] = _json_number(
            feature_row.get(
                f"fx__bilateral__usd_appreciating_share_{horizon}w"
            )
        )
        bilateral[f"return_mad_{horizon}w"] = _json_number(
            feature_row.get(f"fx__bilateral__return_mad_{horizon}w")
        )

    feature_available = pd.Timestamp(coverage_row["feature_available_at"])
    return {
        "status": public_status,
        "method": "fed_h10_usd_strength",
        "bilateral_panel": list(FIXED_BILATERAL_PANEL),
        "coverage": {
            "available_pairs": int(coverage_row["bilateral_level_count"]),
            "required_pairs": len(FIXED_BILATERAL_PANEL),
            "available_indexes": int(coverage_row["core_level_count"]),
            "required_indexes": len(CORE_DOLLAR_INDEXES),
        },
        "indexes": indexes,
        "bilateral": bilateral,
        "observation_week": week.date().isoformat(),
        "observation_age_days": observation_age_days,
        "maximum_age_days": FX_MAX_OBSERVATION_AGE_DAYS,
        "feature_available_at": feature_available.isoformat(),
        "direction": "positive_is_usd_appreciation",
    }


def _record_rows(
    records: Iterable[Observation],
    *,
    selected_fx: tuple[str, ...],
    as_of: datetime | None,
) -> pd.DataFrame:
    cutoff = _aware_utc(as_of, field_name="as_of") if as_of is not None else None
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Observation):
            raise TypeError("records must contain Observation values")
        if record.source != "frb_h10":
            continue
        if cutoff is not None and record.available_at > cutoff:
            continue
        metadata = dict(record.metadata)
        fx_code = str(metadata.get("fx_code", "")).upper()
        if fx_code not in selected_fx:
            continue
        if str(metadata.get("frequency_code", "")) != "9":
            raise ValueError(f"non-business-day H.10 record for {fx_code}")
        spec = SERIES_CATALOG[fx_code]
        quote = str(metadata.get("quote_convention", ""))
        try:
            sign = int(metadata.get("usd_strength_sign"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"missing USD-strength sign for {fx_code}") from exc
        if quote != spec.quote_convention.value or sign != spec.usd_strength_sign:
            raise ValueError(f"quote metadata changed for {fx_code}")
        status = str(metadata.get("obs_status", ""))
        if not status:
            raise ValueError(f"observation status is missing for {fx_code}")
        if status == "A":
            if record.value is None or not math.isfinite(record.value) or record.value <= 0:
                raise ValueError(f"normal H.10 value is invalid for {fx_code}")
        elif record.value is not None:
            raise ValueError(f"non-normal H.10 value must be null for {fx_code}")
        rows.append(
            {
                "fx_code": fx_code,
                "observed_date": pd.Timestamp(record.observed_period_end),
                "value": record.value,
                "status": status,
                "sign": sign,
                "available_at": pd.Timestamp(record.available_at),
                "retrieved_at": pd.Timestamp(record.retrieved_at),
                "vintage_date": pd.Timestamp(record.vintage_date),
                "revision_seq": int(record.revision_seq),
                "raw_sha256": record.raw_sha256,
            }
        )
    if not rows:
        raise ValueError("no selected H.10 observations are available")

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        [
            "fx_code",
            "observed_date",
            "available_at",
            "vintage_date",
            "revision_seq",
            "retrieved_at",
            "raw_sha256",
        ],
        kind="mergesort",
    )
    # Full-history H.10 snapshots repeat unchanged observations.  Collapse
    # only consecutive semantic duplicates so their original first-seen time
    # survives, while an A -> B -> A revision sequence still retains the later
    # reversion as a distinct version.
    semantic = frame[["fx_code", "observed_date", "status", "value"]].copy()
    semantic["value"] = semantic["value"].fillna(float("-inf"))
    previous = semantic.groupby(
        ["fx_code", "observed_date"], sort=False
    )[["status", "value"]].shift(1)
    repeated = semantic[["status", "value"]].eq(previous).all(axis=1)
    frame = frame.loc[~repeated].copy()
    # A latest null version is a tombstone for that series/date; it must not
    # reveal an earlier normal value from the same observation date.
    frame = frame.drop_duplicates(
        ["fx_code", "observed_date"],
        keep="last",
    ).reset_index(drop=True)
    frame["week_end"] = (
        frame["observed_date"]
        .dt.to_period("W-FRI")
        .dt.end_time
        .dt.normalize()
    )
    return frame


def _row_datetime_max(frame: pd.DataFrame) -> pd.Series:
    values: list[pd.Timestamp | pd.NaT] = []
    for row in frame.itertuples(index=False, name=None):
        candidates = [pd.Timestamp(value) for value in row if pd.notna(value)]
        values.append(max(candidates) if candidates else pd.NaT)
    return pd.Series(
        pd.to_datetime(values, utc=True),
        index=frame.index,
        name="source_available_at",
    )


def _rolling_datetime_max(series: pd.Series, window: int) -> pd.Series:
    values = list(series)
    result: list[pd.Timestamp | pd.NaT] = []
    for position in range(len(values)):
        start = max(0, position - window + 1)
        candidates = [
            pd.Timestamp(value)
            for value in values[start : position + 1]
            if pd.notna(value)
        ]
        result.append(max(candidates) if candidates else pd.NaT)
    return pd.Series(
        pd.to_datetime(result, utc=True),
        index=series.index,
        name="feature_available_at",
    )


def _row_mad(frame: pd.DataFrame, *, minimum_count: int) -> pd.Series:
    count = frame.notna().sum(axis=1)
    median = frame.median(axis=1, skipna=True)
    deviation = frame.sub(median, axis=0).abs().median(axis=1, skipna=True)
    return deviation.where(count >= minimum_count)


def _build_weekly_levels(
    frame: pd.DataFrame,
    *,
    selected_fx: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    all_weeks = pd.DatetimeIndex(
        sorted(frame["week_end"].unique()),
        name="week_end",
    )
    valid = frame.loc[frame["value"].notna()].copy()
    valid["usd_log_level"] = valid["sign"].astype(float) * np.log(
        valid["value"].astype(float)
    )
    weekly = (
        valid.sort_values(["week_end", "fx_code", "observed_date"])
        .groupby(["week_end", "fx_code"], sort=True, as_index=False)
        .tail(1)
    )
    levels = weekly.pivot(
        index="week_end",
        columns="fx_code",
        values="usd_log_level",
    ).reindex(index=all_weeks, columns=selected_fx)
    levels.columns.name = None
    availability = weekly.pivot(
        index="week_end",
        columns="fx_code",
        values="available_at",
    ).reindex(index=all_weeks, columns=selected_fx)
    availability.columns.name = None
    non_normal = (
        frame.loc[frame["status"].ne("A")]
        .groupby("week_end")
        .size()
        .reindex(all_weeks, fill_value=0)
        .astype("int64")
    )
    non_normal.name = "non_normal_observation_count"
    return levels.astype(float), availability, non_normal


def build_fx_features(
    records: Iterable[Observation],
    config: FXFeatureConfig | None = None,
    *,
    as_of: datetime | None = None,
) -> FXFeatureResult:
    """Build causal weekly features without forward-filling absent H.10 weeks.

    The output index is the observation week ending Friday.  Availability is
    carried separately and remains the maximum first-seen timestamp required
    by the 26-week feature lookback; downstream joins must still enforce it.
    """

    cfg = config or FXFeatureConfig()
    selected_fx = cfg.core_indexes + cfg.bilateral_panel
    raw = _record_rows(records, selected_fx=selected_fx, as_of=as_of)
    levels, availability, non_normal = _build_weekly_levels(
        raw,
        selected_fx=selected_fx,
    )
    features = pd.DataFrame(index=levels.index)
    returns: dict[tuple[str, int], pd.Series] = {}
    one_week_returns: dict[str, pd.Series] = {}

    annualization = math.sqrt(52.0) if cfg.annualize_volatility else 1.0
    for fx_code in selected_fx:
        stem = fx_code.lower()
        series = levels[fx_code]
        features[f"fx__{stem}__usd_log_level"] = series
        for horizon in cfg.horizons:
            values = series.diff(horizon)
            returns[(fx_code, horizon)] = values
            features[
                f"fx__{stem}__usd_log_return_{horizon}w"
            ] = values
        one_week = series.diff(1)
        one_week_returns[fx_code] = one_week
        for window in cfg.volatility_windows:
            features[
                f"fx__{stem}__realized_vol_{window}w"
            ] = (
                one_week.rolling(window, min_periods=window).std(ddof=0)
                * annualization
            )

    coverage = pd.DataFrame(index=levels.index)
    core_levels = levels.loc[:, cfg.core_indexes]
    panel_levels = levels.loc[:, cfg.bilateral_panel]
    coverage["core_level_count"] = core_levels.notna().sum(axis=1).astype("int64")
    coverage["core_level_ratio"] = coverage["core_level_count"] / len(
        cfg.core_indexes
    )
    coverage["bilateral_level_count"] = (
        panel_levels.notna().sum(axis=1).astype("int64")
    )
    coverage["bilateral_level_ratio"] = coverage[
        "bilateral_level_count"
    ] / len(cfg.bilateral_panel)
    coverage["non_normal_observation_count"] = non_normal

    required_feature_columns: list[str] = []
    for horizon in cfg.horizons:
        core_returns = pd.DataFrame(
            {
                code: returns[(code, horizon)]
                for code in cfg.core_indexes
            },
            index=levels.index,
        )
        complete_core = core_returns.notna().sum(axis=1).eq(len(cfg.core_indexes))
        features[
            f"fx__eme_minus_afe__usd_log_return_{horizon}w"
        ] = (core_returns["EME"] - core_returns["AFE"]).where(complete_core)
        features[
            f"fx__broad_minus_afe__usd_log_return_{horizon}w"
        ] = (core_returns["BRD"] - core_returns["AFE"]).where(complete_core)
        features[
            f"fx__broad_minus_eme__usd_log_return_{horizon}w"
        ] = (core_returns["BRD"] - core_returns["EME"]).where(complete_core)
        features[
            f"fx__dollar_indexes__return_mad_{horizon}w"
        ] = _row_mad(core_returns, minimum_count=len(cfg.core_indexes))

        panel_returns = pd.DataFrame(
            {
                code: returns[(code, horizon)]
                for code in cfg.bilateral_panel
            },
            index=levels.index,
        )
        panel_count = panel_returns.notna().sum(axis=1).astype("int64")
        enough = panel_count.ge(cfg.minimum_bilateral_count)
        panel_median = panel_returns.median(axis=1, skipna=True).where(enough)
        appreciating = panel_returns.gt(0.0).where(panel_returns.notna())
        breadth = (
            appreciating.astype(float).sum(axis=1, skipna=True)
            / panel_count.replace(0, np.nan)
        ).where(enough)
        mad = _row_mad(
            panel_returns,
            minimum_count=cfg.minimum_bilateral_count,
        )
        median_name = f"fx__bilateral__median_usd_log_return_{horizon}w"
        breadth_name = f"fx__bilateral__usd_appreciating_share_{horizon}w"
        mad_name = f"fx__bilateral__return_mad_{horizon}w"
        features[median_name] = panel_median
        features[breadth_name] = breadth
        features[mad_name] = mad
        coverage[f"bilateral_return_{horizon}w_count"] = panel_count
        coverage[f"bilateral_return_{horizon}w_ratio"] = panel_count / len(
            cfg.bilateral_panel
        )
        required_feature_columns.extend((median_name, breadth_name, mad_name))
        required_feature_columns.extend(
            f"fx__{code.lower()}__usd_log_return_{horizon}w"
            for code in cfg.core_indexes
        )
        required_feature_columns.append(
            f"fx__eme_minus_afe__usd_log_return_{horizon}w"
        )

    for code in cfg.core_indexes:
        for window in cfg.volatility_windows:
            required_feature_columns.append(
                f"fx__{code.lower()}__realized_vol_{window}w"
            )

    source_available_at = _row_datetime_max(availability)
    longest_lookback = max(max(cfg.horizons), max(cfg.volatility_windows) + 1)
    feature_available_at = _rolling_datetime_max(
        source_available_at,
        longest_lookback,
    )
    coverage["source_available_at"] = source_available_at
    coverage["feature_available_at"] = feature_available_at

    core_count = coverage["core_level_count"]
    bilateral_count = coverage["bilateral_level_count"]
    source_status = pd.Series("ok", index=levels.index, dtype="object")
    nothing = core_count.eq(0) & bilateral_count.eq(0)
    insufficient = core_count.lt(len(cfg.core_indexes)) | bilateral_count.lt(
        cfg.minimum_bilateral_count
    )
    partial = bilateral_count.lt(len(cfg.bilateral_panel)) & ~insufficient
    source_status.loc[nothing] = "unavailable"
    source_status.loc[insufficient & ~nothing] = "insufficient_coverage"
    source_status.loc[partial] = "partial"
    coverage["source_status"] = source_status

    feature_ready = features.loc[:, required_feature_columns].notna().all(axis=1)
    feature_status = source_status.copy()
    usable_source = source_status.isin(("ok", "partial"))
    feature_status.loc[usable_source & ~feature_ready] = "warming_up"
    coverage["feature_status"] = feature_status
    coverage["archive_correction_quarantined"] = False
    coverage["archive_correction_available_at"] = pd.Series(
        pd.NaT,
        index=coverage.index,
        dtype="datetime64[ns, UTC]",
    )
    coverage["archive_correction_quarantine_until_week"] = pd.Series(
        pd.NaT,
        index=coverage.index,
        dtype="datetime64[ns]",
    )

    status = coverage.loc[:, ["source_status", "feature_status"]].copy()
    return FXFeatureResult(
        features=features,
        weekly_usd_log_levels=levels,
        weekly_availability=availability,
        coverage=coverage,
        status=status,
    )


def _model_cutoff_on_or_before(value: datetime) -> datetime:
    resolved = _aware_utc(value, field_name="as_of").astimezone(EASTERN)
    friday = resolved.date() - timedelta(days=(resolved.weekday() - 4) % 7)
    candidate = datetime.combine(friday, time(16, 0), tzinfo=EASTERN)
    if candidate > resolved:
        candidate -= timedelta(weeks=1)
    return candidate.astimezone(UTC)


def _model_cutoff_on_or_after(value: datetime) -> datetime:
    resolved = _aware_utc(
        value,
        field_name="correction_available_at",
    ).astimezone(EASTERN)
    friday = resolved.date() + timedelta(days=(4 - resolved.weekday()) % 7)
    candidate = datetime.combine(friday, time(16, 0), tzinfo=EASTERN)
    if candidate < resolved:
        candidate += timedelta(weeks=1)
    return candidate.astimezone(UTC)


def _add_model_cutoff_weeks(value: datetime, weeks: int) -> datetime:
    local = _aware_utc(value, field_name="model_cutoff").astimezone(EASTERN)
    target_date = local.date() + timedelta(weeks=weeks)
    return datetime.combine(
        target_date,
        time(16, 0),
        tzinfo=EASTERN,
    ).astimezone(UTC)


def _observation_week(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("W-FRI").end_time.normalize()


def build_official_archive_fx_features(
    records: Iterable[Observation],
    config: FXFeatureConfig | None = None,
    *,
    as_of: datetime,
    correction_available_at: Sequence[datetime] = (),
) -> FXFeatureResult:
    """Replay official archive vintages at each historical Friday cutoff."""

    cfg = config or FXFeatureConfig()
    selected_fx = cfg.core_indexes + cfg.bilateral_panel
    archive_records = tuple(records)
    if not archive_records:
        raise ValueError("official H.10 archive contains no observations")
    for record in archive_records:
        metadata = dict(record.metadata)
        if (
            record.source != "frb_h10"
            or metadata.get("official_release_archive_ingest") is not True
            or metadata.get("archive_chain_availability_basis")
            != FX_ARCHIVE_AVAILABILITY_BASIS
            or metadata.get("archive_revision_policy")
            != FX_ARCHIVE_REVISION_POLICY
            or metadata.get("availability_basis")
            not in {
                "archived_release_date_16_15_ET",
                FX_ARCHIVE_CORRECTION_AVAILABILITY_BASIS,
            }
        ):
            raise ValueError("official H.10 archive provenance is invalid")

    end_cutoff = _model_cutoff_on_or_before(as_of)
    corrections = tuple(
        sorted(
            {
                _aware_utc(value, field_name="correction_available_at")
                for value in correction_available_at
                if _aware_utc(
                    value,
                    field_name="correction_available_at",
                )
                <= end_cutoff
            }
        )
    )
    correction_windows = tuple(
        (
            correction,
            _model_cutoff_on_or_after(correction),
            _add_model_cutoff_weeks(
                _model_cutoff_on_or_after(correction),
                FX_ARCHIVE_CORRECTION_QUARANTINE_WEEKS - 1,
            ),
        )
        for correction in corrections
    )

    first_observation_week = min(
        _observation_week(record.observed_period_end)
        for record in archive_records
    )
    first_cutoff_week = first_observation_week + timedelta(weeks=1)
    end_cutoff_week = pd.Timestamp(
        end_cutoff.astimezone(EASTERN).date()
    )
    if first_cutoff_week > end_cutoff_week:
        raise ValueError("official H.10 archive has no completed model cutoff")
    cutoff_weeks = pd.date_range(
        start=first_cutoff_week,
        end=end_cutoff_week,
        freq="W-FRI",
    )
    longest_lookback = max(max(cfg.horizons), max(cfg.volatility_windows) + 1)
    indexed_records = tuple(
        (record, _observation_week(record.observed_period_end))
        for record in archive_records
        if str(record.metadata.get("fx_code", "")).upper() in selected_fx
    )

    feature_rows: list[pd.DataFrame] = []
    level_rows: list[pd.DataFrame] = []
    availability_rows: list[pd.DataFrame] = []
    coverage_rows: list[pd.DataFrame] = []
    for cutoff_week in cutoff_weeks:
        cutoff_local = datetime.combine(
            cutoff_week.date(),
            time(16, 0),
            tzinfo=EASTERN,
        )
        cutoff = cutoff_local.astimezone(UTC)
        target_week = cutoff_week - timedelta(weeks=1)
        window_start = target_week - timedelta(weeks=longest_lookback - 1)
        window_records = tuple(
            record
            for record, observed_week in indexed_records
            if window_start <= observed_week <= target_week
            and record.available_at <= cutoff
        )
        if not window_records:
            continue
        try:
            built = build_fx_features(window_records, cfg, as_of=cutoff)
        except ValueError:
            continue
        if target_week not in built.features.index:
            continue

        active_windows = tuple(
            window
            for window in correction_windows
            if window[1] <= cutoff <= window[2]
        )
        active = (
            max(active_windows, key=lambda item: (item[2], item[0]))
            if active_windows
            else None
        )
        coverage_row = built.coverage.loc[[target_week]].copy()
        is_quarantined = active is not None
        coverage_row.loc[
            target_week,
            "archive_correction_quarantined",
        ] = is_quarantined
        if active is not None:
            coverage_row.loc[
                target_week,
                "archive_correction_available_at",
            ] = pd.Timestamp(active[0])
            coverage_row.loc[
                target_week,
                "archive_correction_quarantine_until_week",
            ] = pd.Timestamp(active[2].astimezone(EASTERN).date())
            coverage_row.loc[target_week, "feature_status"] = (
                "correction_quarantine"
            )

        feature_rows.append(built.features.loc[[target_week]].copy())
        level_rows.append(
            built.weekly_usd_log_levels.loc[[target_week]].copy()
        )
        availability_rows.append(
            built.weekly_availability.loc[[target_week]].copy()
        )
        coverage_rows.append(coverage_row)

    if not feature_rows:
        raise ValueError("official H.10 archive produced no PIT feature rows")
    expected_weeks = pd.DatetimeIndex(
        cutoff_weeks - pd.offsets.Week(1),
        name=feature_rows[0].index.name,
    )
    features = pd.concat(feature_rows).sort_index().reindex(expected_weeks)
    levels = pd.concat(level_rows).sort_index().reindex(expected_weeks)
    availability = (
        pd.concat(availability_rows).sort_index().reindex(expected_weeks)
    )
    coverage = pd.concat(coverage_rows).sort_index().reindex(expected_weeks)
    for column in coverage.columns:
        if column.endswith("_count"):
            coverage[column] = coverage[column].fillna(0).astype("int64")
        elif column.endswith("_ratio"):
            coverage[column] = coverage[column].fillna(0.0).astype(float)
    coverage["source_status"] = coverage["source_status"].fillna("unavailable")
    coverage["feature_status"] = coverage["feature_status"].fillna("unavailable")
    coverage["archive_correction_quarantined"] = False
    coverage["archive_correction_available_at"] = pd.Series(
        pd.NaT,
        index=expected_weeks,
        dtype="datetime64[ns, UTC]",
    )
    coverage["archive_correction_quarantine_until_week"] = pd.Series(
        pd.NaT,
        index=expected_weeks,
        dtype="datetime64[ns]",
    )
    for observation_week in expected_weeks:
        cutoff = datetime.combine(
            (observation_week + timedelta(weeks=1)).date(),
            time(16, 0),
            tzinfo=EASTERN,
        ).astimezone(UTC)
        active_windows = tuple(
            window
            for window in correction_windows
            if window[1] <= cutoff <= window[2]
        )
        if not active_windows:
            continue
        active = max(active_windows, key=lambda item: (item[2], item[0]))
        coverage.loc[
            observation_week,
            "archive_correction_quarantined",
        ] = True
        coverage.loc[
            observation_week,
            "archive_correction_available_at",
        ] = pd.Timestamp(active[0])
        coverage.loc[
            observation_week,
            "archive_correction_quarantine_until_week",
        ] = pd.Timestamp(active[2].astimezone(EASTERN).date())
        coverage.loc[observation_week, "feature_status"] = (
            "correction_quarantine"
        )
    status = coverage.loc[:, ["source_status", "feature_status"]].copy()
    return FXFeatureResult(
        features=features,
        weekly_usd_log_levels=levels,
        weekly_availability=availability,
        coverage=coverage,
        status=status,
        official_release_archive_ingest=True,
        availability_basis=FX_ARCHIVE_AVAILABILITY_BASIS,
        archive_revision_policy=FX_ARCHIVE_REVISION_POLICY,
        archive_correction_availability_basis=(
            FX_ARCHIVE_CORRECTION_AVAILABILITY_BASIS
        ),
    )


__all__ = [
    "CORE_DOLLAR_INDEXES",
    "FX_MAX_OBSERVATION_AGE_DAYS",
    "FX_ARCHIVE_AVAILABILITY_BASIS",
    "FX_ARCHIVE_CORRECTION_AVAILABILITY_BASIS",
    "FX_ARCHIVE_CORRECTION_QUARANTINE_WEEKS",
    "FX_ARCHIVE_REVISION_POLICY",
    "FX_FIRST_SEEN_AVAILABILITY_BASIS",
    "FXFeatureConfig",
    "FXFeatureResult",
    "build_fx_features",
    "build_official_archive_fx_features",
    "fx_context_at",
    "unavailable_fx_context",
]
