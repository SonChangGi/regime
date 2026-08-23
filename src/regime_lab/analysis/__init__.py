"""Causal regime analysis, labeling, and non-DL model evaluation."""

from .ablation import (
    FEATURE_ABLATION_CONTRACT,
    FeatureAblationResult,
    feature_ablation_manifest_document,
    feature_columns_sha256,
    run_feature_ablation,
)
from .features import FeatureConfig, build_weekly_features
from .labels import STATE_ORDER, CausalRegimeLabeler, RegimeLabelConfig
from .models import MODEL_NAMES, MODEL_REGISTRY, BenchmarkProfile
from .models import GaussianHMMChallenger, ModelSpec, SmoothedMarkovClassifier
from .models import align_probabilities, augment_with_current_state, hmm_available
from .models import model_manifest, model_manifest_sha256
from .nowcast import ShadowNowcastConfig, ShadowNowcastResult, filter_shadow_nowcast
from .transitions import DurationAwareTVTPHurdleClassifier
from .transitions import MarkovDiscriminativeBlendClassifier
from .transitions import causal_state_durations, derive_causal_transition_features
from .validation import TRANSITION_MODELS, BenchmarkResult, TransitionBenchmarkResult
from .validation import evaluate_predictions, evaluate_transition_predictions
from .validation import forecast_next_regime, run_benchmark, run_transition_benchmark
from .validation import select_champion
from .validation import select_champion_with_diagnostics
from .structural_models import (
    MULTISCALE_ENSEMBLE_AGGREGATION,
    MULTISCALE_ENSEMBLE_HALF_LIVES_WEEKS,
    MULTISCALE_ENSEMBLE_MODEL_NAME,
    MULTISCALE_INNER_POOL_METHOD,
    MULTISCALE_SCALE_FORECAST_COLUMNS,
    DynamicEnsembleResult,
    MultiscaleEnsembleResult,
    StructuralForecastResult,
)
from .structural_models import augment_benchmark_with_structural_models
from .structural_models import build_causal_dynamic_ensemble_oos
from .structural_models import build_xgb_hazard_destination_oos
from .structural_models import causal_dynamic_ensemble
from .structural_models import causal_multiscale_ensemble
from .structural_models import forecast_structural_probabilities
from .structural_models import project_joint_survival_hazard
from .structural_models import xgb_hazard_destination_probability
from .directional import (
    DirectionalBenchmarkResult,
    first_departure_targets,
    reconcile_directional_risk,
    run_directional_transition_benchmark,
)
from .duration import duration_context
from .fx import FXFeatureConfig, FXFeatureResult, build_fx_features, fx_context_at
from .fx_ablation import (
    FX_VARIANT_ORDER,
    align_fx_features_to_cutoffs,
    fx_ablation_readiness,
    fx_ablation_variants,
)
from .outcomes import ConditionalOutcomeResult, build_conditional_asset_statistics

__all__ = [
    "STATE_ORDER",
    "MODEL_NAMES",
    "MODEL_REGISTRY",
    "TRANSITION_MODELS",
    "BenchmarkProfile",
    "BenchmarkResult",
    "TransitionBenchmarkResult",
    "CausalRegimeLabeler",
    "FeatureConfig",
    "GaussianHMMChallenger",
    "ModelSpec",
    "RegimeLabelConfig",
    "SmoothedMarkovClassifier",
    "ShadowNowcastConfig",
    "ShadowNowcastResult",
    "DurationAwareTVTPHurdleClassifier",
    "DynamicEnsembleResult",
    "MultiscaleEnsembleResult",
    "FeatureAblationResult",
    "FEATURE_ABLATION_CONTRACT",
    "MarkovDiscriminativeBlendClassifier",
    "StructuralForecastResult",
    "MULTISCALE_ENSEMBLE_AGGREGATION",
    "MULTISCALE_ENSEMBLE_HALF_LIVES_WEEKS",
    "MULTISCALE_ENSEMBLE_MODEL_NAME",
    "MULTISCALE_INNER_POOL_METHOD",
    "MULTISCALE_SCALE_FORECAST_COLUMNS",
    "align_probabilities",
    "augment_with_current_state",
    "augment_benchmark_with_structural_models",
    "build_causal_dynamic_ensemble_oos",
    "build_xgb_hazard_destination_oos",
    "build_weekly_features",
    "causal_state_durations",
    "causal_dynamic_ensemble",
    "causal_multiscale_ensemble",
    "derive_causal_transition_features",
    "evaluate_predictions",
    "evaluate_transition_predictions",
    "forecast_next_regime",
    "forecast_structural_probabilities",
    "filter_shadow_nowcast",
    "feature_ablation_manifest_document",
    "feature_columns_sha256",
    "hmm_available",
    "model_manifest",
    "model_manifest_sha256",
    "project_joint_survival_hazard",
    "run_benchmark",
    "run_feature_ablation",
    "run_transition_benchmark",
    "select_champion",
    "select_champion_with_diagnostics",
    "xgb_hazard_destination_probability",
    "ConditionalOutcomeResult",
    "DirectionalBenchmarkResult",
    "FXFeatureConfig",
    "FXFeatureResult",
    "FX_VARIANT_ORDER",
    "align_fx_features_to_cutoffs",
    "build_conditional_asset_statistics",
    "build_fx_features",
    "duration_context",
    "first_departure_targets",
    "fx_context_at",
    "fx_ablation_readiness",
    "fx_ablation_variants",
    "reconcile_directional_risk",
    "run_directional_transition_benchmark",
]
