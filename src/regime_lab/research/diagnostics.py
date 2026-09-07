"""Matched paired uncertainty and event-conditioned value of existing features."""

from __future__ import annotations
import numpy as np
import pandas as pd
from .downside import paired_block_interval


def ablation_diagnostics(predictions: pd.DataFrame) -> dict:
    required = {
        "variant",
        "origin_date",
        "target_date",
        "evaluation_split",
        "actual",
        "current_state",
        "p_risk_on",
        "p_transition",
        "p_risk_off",
    }
    if not required.issubset(predictions):
        raise ValueError("ablation OOS fields missing")
    p = predictions.copy()
    state_idx = p.actual.map({"risk_on": 0, "transition": 1, "risk_off": 2}).to_numpy()
    probabilities = p[["p_risk_on", "p_transition", "p_risk_off"]].to_numpy(float)
    p["loss"] = -np.log(np.clip(probabilities[np.arange(len(p)), state_idx], 1e-12, 1))
    rows = []
    for split, frame in p.groupby("evaluation_split"):
        baseline = frame[frame.variant.eq("all_structural")].set_index(
            ["origin_date", "target_date"]
        )
        if baseline.index.has_duplicates:
            raise ValueError("ablation baseline duplicate origins")
        for variant, group in frame.groupby("variant"):
            group = group.set_index(["origin_date", "target_date"]).sort_index()
            common = group.index.intersection(baseline.index)
            a = group.loc[common]
            b = baseline.loc[common]
            if not a.actual.equals(b.actual):
                raise ValueError("matched ablation targets differ")
            delta = a.loss.to_numpy() - b.loss.to_numpy()
            lo, hi = paired_block_interval(delta)
            event = a.actual.ne(a.current_state).to_numpy()
            riskoff = a.actual.eq("risk_off").to_numpy()
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "weeks": len(common),
                    "delta_log_loss": float(delta.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "event_weeks": int(event.sum()),
                    "event_delta_log_loss": float(delta[event].mean())
                    if event.any()
                    else None,
                    "risk_off_weeks": int(riskoff.sum()),
                    "risk_off_delta_log_loss": float(delta[riskoff].mean())
                    if riskoff.any()
                    else None,
                    "practical_equivalence_margin": 0.01,
                    "equivalence_supported": lo is not None
                    and lo > -0.01
                    and hi < 0.01,
                }
            )
    return {
        "ablation": rows,
        "baseline": "all_structural",
        "delta_definition": "variant_minus_all_structural; lower_is_better",
        "uncertainty": "paired circular 13-week moving blocks; 999 resamples",
    }
