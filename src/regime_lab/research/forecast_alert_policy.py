"""One-week alert packets with causal whole-policy selection and budget accounting.

The old weekly and unlimited-episode policies remain reproducible comparators.
New packets expire at their own next-week target; later events cannot redeem them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from regime_lab.schema import STATE_ORDER
from regime_lab.research.forecast_enhancement_diagnostics import alert_policies as legacy_alert_policies
from regime_lab.research.forecast_enhancement_models import EnhancementProtocol


POLICY = "bounded_episode_budget"
YEAR_WEEKS = 52.1775
OFF_THRESHOLD = float(np.nextafter(1., 2.))


@dataclass(frozen=True)
class AlertPolicy:
    """Derived decision policy; deliberately outside the numerical model cache."""

    schema_version: str = "forecast-alert-policy/2"
    validity_weeks: int = 1
    calibration_window_weeks: int = 156
    minimum_calibration_weeks: int = 52
    annual_false_week_budget: float = 4.
    cooldown_weeks: int = 2
    exit_threshold_fraction: float = .65
    thresholds: tuple[float, ...] = tuple(i / 20 for i in range(1, 20))

    def __post_init__(self):
        if self.validity_weeks != 1:
            raise ValueError("one-week worsening requires a one-week packet")
        for name in ("calibration_window_weeks", "minimum_calibration_weeks", "cooldown_weeks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "cooldown_weeks" else 1):
                raise ValueError(f"invalid {name}")
        if self.minimum_calibration_weeks > self.calibration_window_weeks:
            raise ValueError("minimum history exceeds calibration window")
        if not np.isfinite(self.annual_false_week_budget) or self.annual_false_week_budget < 0:
            raise ValueError("invalid false-week budget")
        if not 0 < self.exit_threshold_fraction < 1:
            raise ValueError("exit threshold fraction must be between zero and one")
        if not self.thresholds or any(not np.isfinite(t) or not 0 < t <= 1 for t in self.thresholds):
            raise ValueError("invalid threshold grid")

    def record(self):
        return {**asdict(self), "selected_policy": POLICY,
                "target": "one_week_worsening", "budget": self.annual_false_week_budget,
                "budget_unit": "false_alert_weeks_per_year", "annualization_weeks": YEAR_WEEKS,
                "budget_denominator": "all_strictly_matured_forecast_weeks_in_rolling_window",
                "matching": "one_packet_to_one_worsening_at_its_own_next_week_target",
                "episode_definition": "one_issued_packet_expiring_at_its_target",
                "selection": "full_packet_expiry_cooldown_reset_replay_on_strictly_matured_history",
                "budget_enforcement": "feasible_historical_policy_then_realized_false_weeks_plus_pending_packet_reservations",
                "lead_definition": "matched_packet_origin_to_its_own_target",
                "off_policy": "available_when_history_or_budget_is_insufficient"}


@dataclass
class _State:
    cooldown: int = 0
    release_threshold: float | None = None

    def advance(self, probability: float, threshold: float, config: AlertPolicy, permitted: bool = True):
        """Truth never enters packet expiry, reset or cooldown transitions."""
        if self.release_threshold is not None and probability < self.release_threshold:
            self.release_threshold = None
        if self.cooldown:
            self.cooldown -= 1
            return False, "cooldown"
        if self.release_threshold is not None:
            return False, "awaiting_reset"
        if not permitted:
            return False, "budget_limited"
        if threshold > 1:
            return False, "no_supported_policy"
        if probability < threshold:
            return False, "watching"
        self.cooldown = config.cooldown_weeks
        # Freeze the reset level at issuance; changing the selected threshold
        # cannot rearm a continuously high signal.
        self.release_threshold = threshold * config.exit_threshold_fraction
        return True, "issued"


def replay_candidate(probabilities, actual, threshold: float, config: AlertPolicy):
    """Evaluate the whole fixed policy, including expiry and rearming."""
    state = _State()
    alerts = np.array([state.advance(float(p), threshold, config)[0] for p in probabilities], bool)
    truth = np.asarray(actual, bool)
    false = int((alerts & ~truth).sum())
    return {"threshold": float(threshold), "hit_count": int((alerts & truth).sum()),
            "false_alert_weeks": false, "alert_packets": int(alerts.sum()),
            "false_alerts_per_year": false / len(truth) * YEAR_WEEKS if len(truth) else 0.}


def select_policy(probabilities, actual, config: AlertPolicy):
    """Predeclared candidates maximize hits subject to the false-week budget."""
    candidates = [replay_candidate(probabilities, actual, t, config)
                  for t in (*config.thresholds, OFF_THRESHOLD)]
    passing = [r for r in candidates if r["false_alerts_per_year"] <= config.annual_false_week_budget + 1e-12]
    return min(passing, key=lambda r: (-r["hit_count"], r["false_alert_weeks"], r["alert_packets"], -r["threshold"]))


def _validated(frame):
    required = {"model", "origin_date", "target_date", "current_state", "actual", "worsening_probability", "evaluation_split"}
    if not required <= set(frame.columns):
        raise ValueError("incomplete one-week alert inputs")
    frame = frame.copy()
    for name in ("origin_date", "target_date"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
        if frame[name].isna().any():
            raise ValueError("missing alert date")
    if frame.duplicated(["model", "origin_date"]).any() or frame.duplicated(["model", "target_date"]).any():
        raise ValueError("one-to-one alert matching requires unique origins and targets")
    duration = (frame.target_date - frame.origin_date).dt.total_seconds() / 604800
    # Calendar-week dates can span a daylight-saving clock change.
    if not duration.between(1 - 1 / 168, 1 + 1 / 168).all():
        raise ValueError("alert target must be exactly the next calendar week")
    p = frame.worsening_probability.to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid worsening probability")
    if not frame.current_state.isin(STATE_ORDER).all() or not frame.loc[frame.actual.notna(), "actual"].isin(STATE_ORDER).all():
        raise ValueError("invalid alert state")
    return frame


def bounded_alert_history(frame, config: AlertPolicy = AlertPolicy()):
    frame = _validated(frame)
    rows = []
    for model, group in frame.groupby("model"):
        group = group.sort_values("origin_date")
        state = _State()
        issued = []
        for record in group.itertuples(index=False):
            matured = group.loc[(group.target_date < record.origin_date) & group.actual.notna()]
            history = matured.tail(config.calibration_window_weeks)
            ready = len(history) >= config.minimum_calibration_weeks
            truth = np.array([STATE_ORDER.index(a) > STATE_ORDER.index(c) for a, c in zip(history.actual, history.current_state)], bool)
            selected = (select_policy(history.worsening_probability.to_numpy(float), truth, config)
                        if ready else {"threshold": OFF_THRESHOLD, "hit_count": 0, "false_alert_weeks": 0, "false_alerts_per_year": 0., "alert_packets": 0})
            history_truth = dict(zip(history.target_date, truth))
            # Settled false packets consume budget; every unresolved packet
            # reserves one false week until its own outcome becomes available.
            settled_false = sum(packet["target_date"] in history_truth and not history_truth[packet["target_date"]] for packet in issued)
            settled_targets = set(matured.target_date)
            pending = sum(packet["target_date"] not in settled_targets for packet in issued)
            allowance = int(np.floor(config.annual_false_week_budget * len(history) / YEAR_WEEKS + 1e-12))
            capacity = max(0, allowance - settled_false - pending)
            alert, status = state.advance(float(record.worsening_probability), selected["threshold"], config, permitted=ready and capacity > 0)
            if not ready:
                status = "insufficient_history"
            elif selected["threshold"] > 1 and status in ("watching", "budget_limited", "no_supported_policy"):
                status = "no_supported_policy"
            if alert:
                issued.append({"origin_date": record.origin_date, "target_date": record.target_date})
            actual = None if pd.isna(record.actual) else STATE_ORDER.index(record.actual) > STATE_ORDER.index(record.current_state)
            event_id = f"{model}:one_week_worsening:{record.target_date.isoformat()}" if actual else None
            packet_id = f"{model}:{POLICY}:{record.origin_date.isoformat()}" if alert else None
            rows.append({"model": model, "policy": POLICY, "origin_date": record.origin_date,
                         "target_date": record.target_date, "valid_until": record.target_date,
                         "horizon_weeks": 1, "target": "one_week_worsening", "current_state": record.current_state,
                         "probability": float(record.worsening_probability), "threshold": selected["threshold"],
                         "alert": alert, "episode_start": alert, "packet_id": packet_id,
                         "actual": actual, "event_id": event_id, "matched_event_id": event_id if alert else None,
                         "matched_packet_id": packet_id if actual else None,
                         "evaluation_split": record.evaluation_split, "policy_status": status, "reason": status,
                         "calibration_rows": len(history), "last_policy_target": history.target_date.max() if len(history) else None,
                         "policy_validation_hit_count": selected["hit_count"],
                         "policy_validation_false_weeks": selected["false_alert_weeks"],
                         "policy_validation_false_alerts_per_year": selected["false_alerts_per_year"],
                         "budget_allowed_false_weeks": allowance, "budget_settled_false_weeks": int(settled_false),
                         "budget_pending_reservations": int(pending), "budget_capacity_remaining": capacity - int(alert),
                         "cooldown_remaining": state.cooldown, "release_threshold": state.release_threshold,
                         "lead_weeks": 1. if alert and actual else None})
    return pd.DataFrame(rows)


def bounded_metrics(frame, config: AlertPolicy = AlertPolicy()):
    rows = []
    mature = frame.loc[frame.actual.notna() & frame.evaluation_split.isin(["selection", "retrospective_diagnostic"])]
    for (model, period), group in mature.groupby(["model", "evaluation_split"]):
        group = group.sort_values("origin_date")
        truth, alert = group.actual.to_numpy(bool), group.alert.to_numpy(bool)
        hits, false = int((truth & alert).sum()), int((~truth & alert).sum())
        matches = group.loc[truth & alert, "matched_event_id"]
        if matches.isna().any() or matches.duplicated().any():
            raise ValueError("alert matching must be one-to-one")
        rate = false / len(group) * YEAR_WEEKS
        rows.append({"model": model, "policy": POLICY, "evaluation_split": period, "horizon_weeks": 1,
                     "evaluation_start": group.origin_date.min(), "evaluation_end": group.origin_date.max(),
                     "evaluation_target_end": group.target_date.max(), "n_predictions": len(group),
                     "event_count": int(truth.sum()), "hit_count": hits, "matched_event_count": hits,
                     "recall": hits / int(truth.sum()) if truth.any() else None,
                     "false_alert_weeks": false, "false_alerts_per_year": rate,
                     "false_alert_episodes": false, "false_alert_episodes_per_year": rate,
                     "alert_episodes": int(alert.sum()), "expired_alert_count": int(alert.sum()),
                     "expired_without_event_count": false, "right_censored_alert_episodes": 0,
                     "annual_budget": config.annual_false_week_budget, "budget_unit": "false_alert_weeks_per_year",
                     "false_week_budget_exceeded": rate > config.annual_false_week_budget + 1e-12,
                     "false_episode_budget_exceeded": None,
                     "mean_lead_weeks_detected_events": 1. if hits else None,
                     "median_lead_weeks_detected_events": 1. if hits else None,
                     "lead_definition": "matched_one_week_packet_only",
                     "policy_ready_weeks": int(group.calibration_rows.ge(config.minimum_calibration_weeks).sum()),
                     "budget_limited_weeks": int(group.policy_status.eq("budget_limited").sum()),
                     "comparison_role": "bounded_candidate"})
    return rows


def alert_policies(frame, protocol: EnhancementProtocol = EnhancementProtocol(), *, config: AlertPolicy | None = None):
    """Preserve legacy numbers and add the bounded, fully budgeted candidate."""
    config = config or AlertPolicy(annual_false_week_budget=protocol.alert_annual_budget,
                                  cooldown_weeks=protocol.alert_cooldown_weeks,
                                  exit_threshold_fraction=protocol.alert_exit_fraction)
    frame = _validated(frame)
    legacy = legacy_alert_policies(frame, protocol)
    legacy_history = pd.DataFrame(legacy["history"])
    for row in legacy["metrics"]:
        group = legacy_history.loc[legacy_history.model.eq(row["model"]) & legacy_history.policy.eq(row["policy"]) & legacy_history.evaluation_split.eq(row["evaluation_split"]) & legacy_history.actual.notna()]
        row.update({"evaluation_start": group.origin_date.min(), "evaluation_end": group.origin_date.max(),
                    "evaluation_target_end": group.target_date.max(), "comparison_role": "preserved_legacy_baseline"})
    bounded = bounded_alert_history(frame, config)
    return {"metrics": legacy["metrics"] + bounded_metrics(bounded, config),
            "history": legacy["history"] + bounded.to_dict("records"),
            "latest": legacy["latest"] + bounded.loc[bounded.origin_date.eq(bounded.origin_date.max())].to_dict("records"),
            "policy": config.record(), "legacy_policy": legacy["policy"]}
