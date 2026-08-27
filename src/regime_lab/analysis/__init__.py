"""Causal regime analysis, labeling, and non-DL model evaluation."""

from .ablation import (
    FEATURE_ABLATION_CONTRACT,
    FeatureAblationResult,
    feature_ablation_manifest_document,
    feature_columns_sha256,
    run_feature_ablation,
)
from .features import FeatureConfig, build_weekly_features
from .dynamic_factor_tvtp import (
    DynamicFactorTVTPConfig,
    DynamicFactorTVTPShadowResult,
    run_dynamic_factor_tvtp_shadow,
)
from .labels import STATE_ORDER, CausalRegimeLabeler, RegimeLabelConfig
from .label_evaluation import (
    LABEL_EVALUATION_HORIZONS,
    LabelEvaluationResult,
    build_external_origin_outcomes,
    compare_label_sensitivity,
    crash_recovery_lags,
    evaluate_label_definition,
    label_duration_summary,
    label_flip_summary,
    label_occupancy,
    prefix_stability_report,
    summarize_external_outcomes,
)
from .label_research import (
    IMPLEMENTABLE_LABEL_SPECS,
    MEMBERSHIP_SEMANTICS,
    ResearchRegimeLabeler,
    make_research_labeler,
    raw_research_label_components,
)
from .label_spec import (
    BROAD_EQUITY_BREADTH_SYMBOLS,
    DEFAULT_LABEL_SPEC_ID,
    FIXED_NINE_SECTORS,
    LABEL_SPEC_VERSION_LOCK,
    LabelSpecError,
    LabelSpecRegistry,
    LabelSpecification,
    label_spec_manifest_document,
    load_label_spec,
    load_label_spec_registry,
)
from .mechanism_ablation import (
    FEATURE_ROLES,
    MECHANISM_ABLATION_SCHEMA_VERSION,
    MECHANISM_TRACKS,
    MechanismAblationResult,
    MechanismAblationSpec,
    load_mechanism_ablation_spec,
    mechanism_ablation_manifest_document,
    run_mechanism_ablation,
)
from .models import MODEL_NAMES, MODEL_REGISTRY, BenchmarkProfile
from .models import GaussianHMMChallenger, ModelSpec, SmoothedMarkovClassifier
from .models import align_probabilities, augment_with_current_state, hmm_available
from .models import model_manifest, model_manifest_sha256
from .model_confidence_set import (
    MCS_DEFAULT_ALPHA,
    MCS_DEFAULT_BLOCK_WEEKS,
    MCS_DEFAULT_RESAMPLES,
    MCS_DEFAULT_SEED,
    MCS_METHOD,
    ModelConfidenceSetResult,
    ModelConfidenceSetStep,
    model_confidence_set,
    validate_matched_loss_matrix,
)
from .pit_total_return import (
    CORPORATE_ACTION_CONTRACT,
    PIT_TOTAL_RETURN_PANEL_SCHEMA_VERSION,
    PITTotalReturnPanel,
    PITTotalReturnResult,
    build_pit_total_return_panel,
    reconstruct_pit_total_return,
    validate_pit_total_return_panel,
)
from .pagan_sossounov import (
    PaganSossounovConfig,
    PaganSossounovResult,
    pagan_sossounov_chronology,
)
from .selection_evaluation import (
    SELECTION_EVALUATION_ROLE,
    SELECTION_EVALUATION_SCHEMA_VERSION,
    SELECTION_EVIDENCE_STATUSES,
    build_selection_evaluation,
    validate_selection_evaluation,
)
from .nowcast import ShadowNowcastConfig, ShadowNowcastResult, filter_shadow_nowcast
from .shadow_regimes import (
    BOCPDConfig,
    BOCPDShadowResult,
    DirectJumpHSMMConfig,
    FilteredHSMMShadowResult,
    bayesian_online_changepoint_shadow,
    filter_direct_jump_hsmm_shadow,
    shadow_model_registry_document,
)
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
from .decision_shadow import build_decision_shadow, load_decision_shadow_spec

__all__ = [
    "STATE_ORDER",
    "MODEL_NAMES",
    "MODEL_REGISTRY",
    "TRANSITION_MODELS",
    "BenchmarkProfile",
    "BenchmarkResult",
    "TransitionBenchmarkResult",
    "CausalRegimeLabeler",
    "BROAD_EQUITY_BREADTH_SYMBOLS",
    "DEFAULT_LABEL_SPEC_ID",
    "FeatureConfig",
    "FIXED_NINE_SECTORS",
    "GaussianHMMChallenger",
    "ModelSpec",
    "ModelConfidenceSetResult",
    "ModelConfidenceSetStep",
    "MCS_DEFAULT_ALPHA",
    "MCS_DEFAULT_BLOCK_WEEKS",
    "MCS_DEFAULT_RESAMPLES",
    "MCS_DEFAULT_SEED",
    "MCS_METHOD",
    "CORPORATE_ACTION_CONTRACT",
    "PIT_TOTAL_RETURN_PANEL_SCHEMA_VERSION",
    "PITTotalReturnPanel",
    "PITTotalReturnResult",
    "PaganSossounovConfig",
    "PaganSossounovResult",
    "SELECTION_EVALUATION_ROLE",
    "SELECTION_EVALUATION_SCHEMA_VERSION",
    "SELECTION_EVIDENCE_STATUSES",
    "RegimeLabelConfig",
    "ResearchRegimeLabeler",
    "IMPLEMENTABLE_LABEL_SPECS",
    "LABEL_EVALUATION_HORIZONS",
    "LABEL_SPEC_VERSION_LOCK",
    "MEMBERSHIP_SEMANTICS",
    "LabelEvaluationResult",
    "LabelSpecError",
    "LabelSpecRegistry",
    "LabelSpecification",
    "MechanismAblationResult",
    "MechanismAblationSpec",
    "SmoothedMarkovClassifier",
    "ShadowNowcastConfig",
    "ShadowNowcastResult",
    "DurationAwareTVTPHurdleClassifier",
    "DynamicFactorTVTPConfig",
    "DynamicFactorTVTPShadowResult",
    "DynamicEnsembleResult",
    "MultiscaleEnsembleResult",
    "FeatureAblationResult",
    "FEATURE_ABLATION_CONTRACT",
    "FEATURE_ROLES",
    "MECHANISM_ABLATION_SCHEMA_VERSION",
    "MECHANISM_TRACKS",
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
    "build_pit_total_return_panel",
    "build_external_origin_outcomes",
    "causal_state_durations",
    "causal_dynamic_ensemble",
    "causal_multiscale_ensemble",
    "derive_causal_transition_features",
    "evaluate_predictions",
    "evaluate_label_definition",
    "evaluate_transition_predictions",
    "forecast_next_regime",
    "forecast_structural_probabilities",
    "filter_shadow_nowcast",
    "filter_direct_jump_hsmm_shadow",
    "feature_ablation_manifest_document",
    "feature_columns_sha256",
    "compare_label_sensitivity",
    "crash_recovery_lags",
    "hmm_available",
    "model_manifest",
    "model_manifest_sha256",
    "model_confidence_set",
    "build_selection_evaluation",
    "label_duration_summary",
    "label_flip_summary",
    "label_occupancy",
    "label_spec_manifest_document",
    "load_label_spec",
    "load_label_spec_registry",
    "load_mechanism_ablation_spec",
    "make_research_labeler",
    "prefix_stability_report",
    "pagan_sossounov_chronology",
    "project_joint_survival_hazard",
    "reconstruct_pit_total_return",
    "run_benchmark",
    "run_feature_ablation",
    "run_mechanism_ablation",
    "run_transition_benchmark",
    "raw_research_label_components",
    "select_champion",
    "select_champion_with_diagnostics",
    "summarize_external_outcomes",
    "validate_matched_loss_matrix",
    "validate_pit_total_return_panel",
    "validate_selection_evaluation",
    "mechanism_ablation_manifest_document",
    "BOCPDConfig",
    "BOCPDShadowResult",
    "DirectJumpHSMMConfig",
    "FilteredHSMMShadowResult",
    "bayesian_online_changepoint_shadow",
    "shadow_model_registry_document",
    "xgb_hazard_destination_probability",
    "ConditionalOutcomeResult",
    "DirectionalBenchmarkResult",
    "FXFeatureConfig",
    "FXFeatureResult",
    "FX_VARIANT_ORDER",
    "align_fx_features_to_cutoffs",
    "build_conditional_asset_statistics",
    "build_decision_shadow",
    "load_decision_shadow_spec",
    "build_fx_features",
    "duration_context",
    "first_departure_targets",
    "fx_context_at",
    "fx_ablation_readiness",
    "fx_ablation_variants",
    "reconcile_directional_risk",
    "run_directional_transition_benchmark",
    "run_dynamic_factor_tvtp_shadow",
]
