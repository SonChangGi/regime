"""Point-in-time total-return reconstruction from dated corporate actions.

The operating label challengers must not treat a provider's latest adjusted
history as if that history had been available at every earlier forecast.  This
module instead consumes raw closes and event values carrying explicit release,
first-seen, retrieval, revision, and raw-content identities.

The input ``split_coefficient`` is new shares per old share for the interval and
``dividend_amount`` is cash per pre-split share.  Upstream event normalization
must aggregate any within-week ordering to that contract.  The weekly gross
return is therefore ``(close_t * split_t + dividend_t) / close_(t-1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.operating_contract import load_operating_contract


EvidenceTrack = Literal["operational_oos", "reconstructed_oos"]
CORPORATE_ACTION_CONTRACT = "split_then_dividend_per_pre_split_share_v1"
PIT_TOTAL_RETURN_PANEL_SCHEMA_VERSION = "pit-total-return-panel/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_COLUMNS = (
    "raw_close",
    "dividend_amount",
    "split_coefficient",
    "corporate_action_contract",
    "source_released_at",
    "provider_first_seen_at",
    "system_retrieved_at",
    "revision_seq",
    "raw_sha256",
)


@dataclass(frozen=True)
class PITTotalReturnResult:
    """A derived-only index plus its row-level availability audit."""

    total_return_index: pd.Series
    audit: pd.DataFrame
    input_snapshot_sha256: str
    evidence_track: EvidenceTrack


@dataclass(frozen=True)
class PITTotalReturnPanel:
    """A matched-symbol PIT panel whose lineage is verified, not inferred.

    Research labelers accept this object rather than a conveniently renamed
    DataFrame.  That prevents current-adjusted histories or mixed vintages from
    silently masquerading as point-in-time total-return inputs.
    """

    frame: pd.DataFrame
    results: Mapping[str, PITTotalReturnResult]
    decision_at: pd.Series
    evidence_track: EvidenceTrack
    corporate_action_contract: str
    input_snapshot_sha256: str
    schema_version: str = PIT_TOTAL_RETURN_PANEL_SCHEMA_VERSION

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index

    def __len__(self) -> int:
        return len(self.frame)

    def slice_rows(self, stop: int) -> "PITTotalReturnPanel":
        if isinstance(stop, bool) or int(stop) != stop:
            raise TypeError("PIT panel slice stop must be an integer")
        resolved = int(stop)
        if resolved < 1 or resolved > len(self.frame):
            raise ValueError("PIT panel slice stop must fall inside the panel")
        return build_pit_total_return_panel(
            {
                symbol: _slice_pit_result(result, resolved)
                for symbol, result in self.results.items()
            }
        )


def _aware_utc_series(values: pd.Series, *, field: str) -> pd.Series:
    parsed: list[pd.Timestamp] = []
    for raw in values:
        try:
            item = pd.Timestamp(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain valid timestamps") from exc
        if item.tzinfo is None or item.utcoffset() is None:
            raise ValueError(f"{field} must contain timezone-aware timestamps")
        parsed.append(item.tz_convert("UTC"))
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns, UTC]")


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("corporate-action frame must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("corporate-action frame must use a DatetimeIndex")
    if len(frame) < 2:
        raise ValueError("corporate-action frame requires at least two periods")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("observed periods must be unique and increasing")
    missing = sorted(set(_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise KeyError(f"missing corporate-action columns: {missing}")

    output = frame.loc[:, list(_REQUIRED_COLUMNS)].copy()
    raw_close = pd.to_numeric(output["raw_close"], errors="coerce").astype(float)
    dividends = pd.to_numeric(
        output["dividend_amount"], errors="coerce"
    ).astype(float)
    splits = pd.to_numeric(
        output["split_coefficient"], errors="coerce"
    ).astype(float)
    if (
        not np.isfinite(raw_close.to_numpy()).all()
        or not np.isfinite(dividends.to_numpy()).all()
        or not np.isfinite(splits.to_numpy()).all()
        or bool((raw_close <= 0.0).any())
        or bool((dividends < 0.0).any())
        or bool((splits <= 0.0).any())
    ):
        raise ValueError(
            "raw closes/splits must be positive and dividends non-negative"
        )
    if not output["corporate_action_contract"].eq(
        CORPORATE_ACTION_CONTRACT
    ).all():
        raise ValueError("corporate-action normalization contract is invalid")

    revision_numeric = pd.to_numeric(output["revision_seq"], errors="coerce")
    if (
        revision_numeric.isna().any()
        or bool((revision_numeric < 0).any())
        or not np.equal(revision_numeric, np.floor(revision_numeric)).all()
    ):
        raise ValueError("revision_seq must contain non-negative integers")
    hashes = output["raw_sha256"].astype(str)
    if not hashes.map(lambda value: _SHA256.fullmatch(value) is not None).all():
        raise ValueError("raw_sha256 must contain lowercase SHA-256 values")

    output["raw_close"] = raw_close
    output["dividend_amount"] = dividends
    output["split_coefficient"] = splits
    output["revision_seq"] = revision_numeric.astype(int)
    for field in (
        "source_released_at",
        "provider_first_seen_at",
        "system_retrieved_at",
    ):
        output[field] = _aware_utc_series(output[field], field=field)
    if bool(
        (
            output["provider_first_seen_at"]
            > output["system_retrieved_at"]
        ).any()
    ):
        raise ValueError("provider_first_seen_at cannot follow system retrieval")
    if bool(
        (output["source_released_at"] > output["system_retrieved_at"]).any()
    ):
        raise ValueError("source release cannot follow system retrieval")
    return output


def _validate_decisions(
    decisions: pd.Series,
    *,
    index: pd.DatetimeIndex,
) -> pd.Series:
    if not isinstance(decisions, pd.Series) or not decisions.index.equals(index):
        raise ValueError("decision_at must be a Series on the exact observed-period index")
    parsed = _aware_utc_series(decisions, field="decision_at")
    if not parsed.is_monotonic_increasing:
        raise ValueError("decision_at must be increasing")
    observed_dates = index.tz_localize(None).normalize()
    decision_dates = parsed.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    if bool((decision_dates < observed_dates).any()):
        raise ValueError("decision_at cannot precede its observed period")
    return parsed


def _allowed_evidence_tracks() -> tuple[str, ...]:
    """Read the evidence vocabulary from the active typed operating contract."""

    contract = load_operating_contract()
    return tuple(str(item) for item in contract.document["forecast"]["evidence_tracks"])


def _input_snapshot_hash_from_audit(
    audit: pd.DataFrame,
    *,
    evidence_track: EvidenceTrack,
) -> str:
    hash_rows: list[dict[str, object]] = []
    for observed_period, row in audit.iterrows():
        hash_rows.append(
            {
                "observed_period_end": pd.Timestamp(observed_period).isoformat(),
                "raw_close": float(row["raw_close"]),
                "dividend_amount": float(row["dividend_amount"]),
                "split_coefficient": float(row["split_coefficient"]),
                "corporate_action_contract": str(row["corporate_action_contract"]),
                "source_released_at": pd.Timestamp(
                    row["source_released_at"]
                ).isoformat(),
                "provider_first_seen_at": pd.Timestamp(
                    row["provider_first_seen_at"]
                ).isoformat(),
                "system_retrieved_at": pd.Timestamp(
                    row["system_retrieved_at"]
                ).isoformat(),
                "decision_at": pd.Timestamp(row["decision_at"]).isoformat(),
                "revision_seq": int(row["revision_seq"]),
                "raw_sha256": str(row["raw_sha256"]),
            }
        )
    return canonical_json_sha256_v1(
        {
            "schema_version": "pit-total-return-input/v1",
            "evidence_track": evidence_track,
            "records": hash_rows,
        }
    )


def _validate_pit_result(result: PITTotalReturnResult) -> None:
    if not isinstance(result, PITTotalReturnResult):
        raise TypeError("PIT panel values must be PITTotalReturnResult instances")
    if result.evidence_track not in _allowed_evidence_tracks():
        raise ValueError("PIT result uses an unsupported evidence track")
    if not isinstance(result.total_return_index.index, pd.DatetimeIndex):
        raise TypeError("PIT result must use a DatetimeIndex")
    if not result.audit.index.equals(result.total_return_index.index):
        raise ValueError("PIT result audit and derived index are not aligned")
    required_audit = set(_REQUIRED_COLUMNS).union(
        {
            "observed_period_end",
            "decision_at",
            "operating_available_at",
            "operational_eligible",
            "reconstructed_eligible",
            "period_return",
            "total_return_index",
            "evidence_track",
        }
    )
    if not required_audit.issubset(result.audit.columns):
        raise ValueError("PIT result audit is incomplete")
    if not result.audit["evidence_track"].eq(result.evidence_track).all():
        raise ValueError("PIT result evidence track is internally inconsistent")
    observed = pd.DatetimeIndex(result.audit["observed_period_end"])
    if not observed.equals(result.audit.index):
        raise ValueError("PIT observed-period identity is inconsistent")
    if not result.audit["corporate_action_contract"].eq(
        CORPORATE_ACTION_CONTRACT
    ).all():
        raise ValueError("PIT corporate-action contract is inconsistent")

    normalized = _validate_frame(result.audit)
    decisions = _validate_decisions(
        result.audit["decision_at"],
        index=result.audit.index,
    )
    expected_available = normalized[
        ["source_released_at", "provider_first_seen_at"]
    ].max(axis=1)
    expected_operational_eligible = (
        expected_available.le(decisions)
        & normalized["system_retrieved_at"].le(decisions)
    )
    expected_reconstructed_eligible = normalized["source_released_at"].le(
        decisions
    )
    observed_available = _aware_utc_series(
        result.audit["operating_available_at"],
        field="operating_available_at",
    )
    if not observed_available.equals(expected_available):
        raise ValueError("PIT operating availability does not match its clocks")
    if not result.audit["operational_eligible"].astype(bool).equals(
        expected_operational_eligible.astype(bool)
    ):
        raise ValueError("PIT operational eligibility does not match its clocks")
    if not result.audit["reconstructed_eligible"].astype(bool).equals(
        expected_reconstructed_eligible.astype(bool)
    ):
        raise ValueError("PIT reconstructed eligibility does not match its clocks")
    if result.evidence_track == "operational_oos" and not bool(
        expected_operational_eligible.all()
    ):
        raise ValueError("PIT operational result contains an ineligible observation")
    if result.evidence_track == "reconstructed_oos" and not bool(
        expected_reconstructed_eligible.all()
    ):
        raise ValueError("PIT reconstructed result contains an ineligible observation")

    previous_close = normalized["raw_close"].shift(1)
    expected_gross = (
        normalized["raw_close"] * normalized["split_coefficient"]
        + normalized["dividend_amount"]
    ) / previous_close
    expected_period_return = expected_gross - 1.0
    expected_period_return.iloc[0] = np.nan
    observed_period_return = pd.to_numeric(
        result.audit["period_return"], errors="coerce"
    ).astype(float)
    if not np.allclose(
        observed_period_return.to_numpy(),
        expected_period_return.to_numpy(),
        rtol=1e-13,
        atol=1e-13,
        equal_nan=True,
    ):
        raise ValueError("PIT period return does not match corporate-action inputs")

    observed_result_index = pd.to_numeric(
        result.total_return_index, errors="coerce"
    ).astype(float)
    base_value = float(observed_result_index.iloc[0])
    if not np.isfinite(base_value) or base_value <= 0.0:
        raise ValueError("PIT derived index base must be finite and positive")
    expected_index = pd.Series(
        base_value,
        index=result.audit.index,
        dtype=float,
    )
    expected_index.iloc[1:] = base_value * expected_gross.iloc[1:].cumprod()
    observed_audit_index = pd.to_numeric(
        result.audit["total_return_index"], errors="coerce"
    ).astype(float)
    if not np.array_equal(
        observed_audit_index.to_numpy(),
        observed_result_index.to_numpy(),
        equal_nan=True,
    ):
        raise ValueError("PIT derived index does not match its audit")
    if not np.allclose(
        observed_result_index.to_numpy(),
        expected_index.to_numpy(),
        rtol=1e-13,
        atol=1e-13,
        equal_nan=False,
    ):
        raise ValueError("PIT derived index does not match corporate-action inputs")
    expected_hash = _input_snapshot_hash_from_audit(
        result.audit,
        evidence_track=result.evidence_track,
    )
    if result.input_snapshot_sha256 != expected_hash:
        raise ValueError("PIT input snapshot hash does not match its audit")


def _slice_pit_result(
    result: PITTotalReturnResult,
    stop: int,
) -> PITTotalReturnResult:
    _validate_pit_result(result)
    audit = result.audit.iloc[:stop].copy()
    total_return = result.total_return_index.iloc[:stop].copy()
    return PITTotalReturnResult(
        total_return_index=total_return,
        audit=audit,
        input_snapshot_sha256=_input_snapshot_hash_from_audit(
            audit,
            evidence_track=result.evidence_track,
        ),
        evidence_track=result.evidence_track,
    )


def build_pit_total_return_panel(
    results: Mapping[str, PITTotalReturnResult],
) -> PITTotalReturnPanel:
    """Bind symbol-level PIT results into one exact-index research panel."""

    if not isinstance(results, Mapping) or not results:
        raise ValueError("PIT panel requires at least one symbol result")
    normalized: dict[str, PITTotalReturnResult] = {}
    for raw_symbol, result in results.items():
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in normalized:
            raise ValueError("PIT panel symbols must be unique and non-empty")
        _validate_pit_result(result)
        normalized[symbol] = PITTotalReturnResult(
            total_return_index=result.total_return_index.copy(deep=True),
            audit=result.audit.copy(deep=True),
            input_snapshot_sha256=str(result.input_snapshot_sha256),
            evidence_track=result.evidence_track,
        )

    first = next(iter(normalized.values()))
    index = first.total_return_index.index
    decisions = first.audit["decision_at"].copy()
    evidence_track = first.evidence_track
    for symbol, result in normalized.items():
        if not result.total_return_index.index.equals(index):
            raise ValueError(f"PIT panel symbol {symbol} does not share the exact index")
        if not result.audit["decision_at"].equals(decisions):
            raise ValueError(f"PIT panel symbol {symbol} has a different decision clock")
        if result.evidence_track != evidence_track:
            raise ValueError("PIT panel cannot mix evidence tracks")

    frame = pd.DataFrame(
        {
            f"{symbol.lower()}_pit_total_return": result.total_return_index
            for symbol, result in normalized.items()
        },
        index=index,
        dtype=float,
    )
    bindings = [
        {
            "symbol": symbol,
            "column": f"{symbol.lower()}_pit_total_return",
            "input_snapshot_sha256": result.input_snapshot_sha256,
        }
        for symbol, result in sorted(normalized.items())
    ]
    panel_hash = canonical_json_sha256_v1(
        {
            "schema_version": PIT_TOTAL_RETURN_PANEL_SCHEMA_VERSION,
            "evidence_track": evidence_track,
            "corporate_action_contract": CORPORATE_ACTION_CONTRACT,
            "bindings": bindings,
        }
    )
    return PITTotalReturnPanel(
        frame=frame,
        results=MappingProxyType(dict(normalized)),
        decision_at=decisions,
        evidence_track=evidence_track,
        corporate_action_contract=CORPORATE_ACTION_CONTRACT,
        input_snapshot_sha256=panel_hash,
    )


def validate_pit_total_return_panel(panel: PITTotalReturnPanel) -> None:
    """Rebuild a panel from bound audits and reject any post-build mutation."""

    if not isinstance(panel, PITTotalReturnPanel):
        raise TypeError("expected a PITTotalReturnPanel")
    rebuilt = build_pit_total_return_panel(panel.results)
    if (
        panel.schema_version != PIT_TOTAL_RETURN_PANEL_SCHEMA_VERSION
        or panel.evidence_track != rebuilt.evidence_track
        or panel.corporate_action_contract != rebuilt.corporate_action_contract
        or panel.input_snapshot_sha256 != rebuilt.input_snapshot_sha256
        or not panel.decision_at.equals(rebuilt.decision_at)
        or not panel.frame.equals(rebuilt.frame)
    ):
        raise ValueError("PIT total-return panel lineage or contents were mutated")


def reconstruct_pit_total_return(
    frame: pd.DataFrame,
    *,
    decision_at: pd.Series,
    evidence_track: EvidenceTrack,
    base_value: float = 100.0,
) -> PITTotalReturnResult:
    """Reconstruct a revision-bound total-return index without future inputs.

    ``operational_oos`` fails if any row was first seen or retrieved after its
    decision.  ``reconstructed_oos`` relaxes first-seen/retrieval but still
    requires the source release at or before the decision, so retrospective
    research cannot use information that did not yet economically exist.
    """

    if evidence_track not in _allowed_evidence_tracks():
        raise ValueError("unsupported PIT evidence track")
    if not np.isfinite(float(base_value)) or float(base_value) <= 0.0:
        raise ValueError("base_value must be finite and positive")
    normalized = _validate_frame(frame)
    decisions = _validate_decisions(decision_at, index=frame.index)

    operating_available = normalized[
        ["source_released_at", "provider_first_seen_at"]
    ].max(axis=1)
    operational_eligible = (
        operating_available.le(decisions)
        & normalized["system_retrieved_at"].le(decisions)
    )
    reconstructed_eligible = normalized["source_released_at"].le(decisions)
    if evidence_track == "reconstructed_oos" and not bool(
        reconstructed_eligible.all()
    ):
        invalid = [
            item.isoformat()
            for item in frame.index[~reconstructed_eligible.to_numpy()]
        ]
        raise ValueError(
            "reconstructed PIT source was not released by decision_at: "
            + ", ".join(invalid)
        )
    if evidence_track == "operational_oos" and not bool(operational_eligible.all()):
        invalid = [
            item.isoformat()
            for item in frame.index[~operational_eligible.to_numpy()]
        ]
        raise ValueError(
            "operational PIT input was not available by decision_at: "
            + ", ".join(invalid)
        )

    previous_close = normalized["raw_close"].shift(1)
    gross_return = (
        normalized["raw_close"] * normalized["split_coefficient"]
        + normalized["dividend_amount"]
    ) / previous_close
    period_return = gross_return - 1.0
    period_return.iloc[0] = np.nan
    if bool((gross_return.iloc[1:] <= 0.0).any()) or not np.isfinite(
        gross_return.iloc[1:].to_numpy()
    ).all():
        raise ValueError("corporate actions imply an invalid total return")

    total_return = pd.Series(
        float(base_value),
        index=frame.index,
        name="pit_total_return",
        dtype=float,
    )
    total_return.iloc[1:] = float(base_value) * gross_return.iloc[1:].cumprod()

    audit = normalized.copy()
    audit.insert(0, "observed_period_end", frame.index)
    audit["decision_at"] = decisions
    audit["operating_available_at"] = operating_available
    audit["operational_eligible"] = operational_eligible.astype(bool)
    audit["reconstructed_eligible"] = reconstructed_eligible.astype(bool)
    audit["period_return"] = period_return
    audit["total_return_index"] = total_return
    audit["evidence_track"] = evidence_track

    input_hash = _input_snapshot_hash_from_audit(
        audit,
        evidence_track=evidence_track,
    )
    return PITTotalReturnResult(
        total_return_index=total_return,
        audit=audit,
        input_snapshot_sha256=input_hash,
        evidence_track=evidence_track,
    )


__all__ = [
    "CORPORATE_ACTION_CONTRACT",
    "EvidenceTrack",
    "PIT_TOTAL_RETURN_PANEL_SCHEMA_VERSION",
    "PITTotalReturnPanel",
    "PITTotalReturnResult",
    "build_pit_total_return_panel",
    "reconstruct_pit_total_return",
    "validate_pit_total_return_panel",
]
