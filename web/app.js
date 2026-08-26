(function () {
  "use strict";

  const DATA_URL = "./data/regime-results.json";
  const V5_COMPARISON_URL = "./data/v5-vs-v4-comparison.json";
  const V5_SELECTION_FAMILY_AUDIT_URL = "./data/selection-family-audit.json";
  const SVG_NS = "http://www.w3.org/2000/svg";
  const THEME_STORAGE_KEY = "quant-research-theme";
  const LEGACY_THEME_STORAGE_KEYS = Object.freeze(["regime-dashboard-theme"]);
  const STATE_ORDER = Object.freeze(["risk_on", "transition", "risk_off"]);
  const V3_RESULT_VERSION = "weekly-regime-result-v3";
  const V3_MODEL_VERSION = "weekly-nondl-structural-v3";
  const V3_LABEL_VERSION = "market-causal-3state-v1";
  const V3_FEATURE_SET_VERSION = "weekly-pit-market-internals-v3";
  const V4_RESULT_VERSION = "weekly-regime-result-v4";
  const V4_MODEL_VERSION = "weekly-nondl-structural-v4";
  const V4_LABEL_VERSION = "market-causal-3state-v1";
  const V4_FEATURE_SET_VERSION = "weekly-pit-structural-v4";
  const V5_RESULT_VERSION = "weekly-regime-result-v5";
  const V5_SCHEMA_VERSION = "2.1.0";
  const V5_MODEL_VERSION = "weekly-nondl-structural-v5";
  const V5_LABEL_VERSION = "market-causal-3state-v1";
  const V5_FEATURE_SET_VERSION = "weekly-pit-structural-v5";
  const V5_PUBLICATION_STATUS = "reviewed_publication";
  const V5_PUBLICATION_REVIEW_SCHEMA = "regime-v5-publication-review/1";
  const OUTCOME_ASSETS = Object.freeze(["SPY", "QQQ", "IWM", "TLT", "HYG", "UUP"]);
  const OUTCOME_ASSET_LABELS = Object.freeze({
    SPY: "미국 대형주",
    QQQ: "나스닥 100",
    IWM: "미국 소형주",
    TLT: "미 장기국채",
    HYG: "미 하이일드",
    UUP: "달러",
  });
  const FX_BILATERAL_PANEL = Object.freeze(["EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "CNY", "MXN", "BRL"]);
  const FX_ABLATION_VARIANTS = Object.freeze([
    "v4_control",
    "v4_plus_broad_index",
    "v4_plus_bilateral_panel",
    "v4_plus_all_fx",
  ]);
  const V5_SOURCE_LICENSES = Object.freeze({
    demo: Object.freeze({
      synthetic_market: "synthetic_fixture",
      synthetic_macro: "synthetic_fixture",
    }),
    live: Object.freeze({
      alpha_vantage: "private_noncommercial",
      alfred: "user_confirmed_ml_storage_derived",
      frb_h10: "federal_reserve_board_public_domain_citation_requested",
    }),
  });
  const V5_SOURCE_STATUSES = Object.freeze([
    "ok",
    "stale",
    "degraded",
    "quota_exhausted",
    "schema_changed",
    "revision_gap",
    "rights_unconfirmed",
    "license_blocked",
    "unavailable",
  ]);
  const FX_STATUS_LABELS = Object.freeze({
    ok: "정상",
    partial: "부분 가용",
    degraded: "최근 정상값",
    stale: "지연",
    insufficient_history: "표본 축적 중",
    unavailable: "사용 불가",
    evaluated: "완료",
  });
  const MODEL_HEALTH_REASON_LABELS = Object.freeze({
    weak_generalization: "일반화 약화",
    calibration_drift: "보정 드리프트",
  });
  const FX_DISPLAY_METRICS = Object.freeze([
    Object.freeze({ block: "indexes", key: "broad_usd_log_return_1w", label: "광의 달러 · 1주", primary: true }),
    Object.freeze({ block: "indexes", key: "broad_usd_log_return_4w", label: "광의 달러 · 4주", primary: true }),
    Object.freeze({ block: "indexes", key: "broad_usd_log_return_13w", label: "광의 달러 · 13주", primary: true }),
    Object.freeze({ block: "bilateral", key: "usd_appreciating_share_1w", label: "달러 강세 통화 비중 · 1주", primary: true }),
    Object.freeze({ block: "indexes", key: "broad_realized_vol_13w", label: "광의 달러 변동성 · 13주", primary: false }),
    Object.freeze({ block: "bilateral", key: "median_usd_log_return_1w", label: "9개 통화 중앙값 · 1주", primary: false }),
    Object.freeze({ block: "bilateral", key: "median_usd_log_return_13w", label: "9개 통화 중앙값 · 13주", primary: false }),
    Object.freeze({ block: "bilateral", key: "usd_appreciating_share_13w", label: "달러 강세 통화 비중 · 13주", primary: false }),
  ]);
  const V5_RESEARCH_ARTIFACT_PATHS = Object.freeze({
    directional_oos_predictions: "directional-oos-predictions.csv",
    directional_model_leaderboard: "directional-model-leaderboard.csv",
    directional_walk_forward_splits: "directional-walk-forward-splits.csv",
    directional_selection_diagnostics: "directional-selection-diagnostics.csv",
    directional_forecasts: "directional-forecasts.csv",
    conditional_asset_outcomes: "conditional-asset-outcomes.csv",
    conditional_asset_statistics: "conditional-asset-statistics.csv",
    model_conditioned_asset_outcomes: "model-conditioned-asset-outcomes.csv",
    model_conditioned_asset_statistics: "model-conditioned-asset-statistics.csv",
    fx_features: "fx-features.csv",
    fx_coverage: "fx-coverage.csv",
    fx_ablation_oos: "fx-ablation-oos.csv",
  });
  const V5_REQUIRED_RESEARCH_ARTIFACTS = Object.freeze([
    "directional_oos_predictions",
    "directional_model_leaderboard",
    "directional_walk_forward_splits",
    "directional_selection_diagnostics",
    "directional_forecasts",
    "conditional_asset_outcomes",
    "conditional_asset_statistics",
  ]);
  const V5_CORE_ARTIFACT_PATHS = Object.freeze({
    oos_predictions: "oos-predictions.csv",
    model_leaderboard: "model-leaderboard.csv",
    walk_forward_splits: "walk-forward-splits.csv",
    selection_diagnostics: "selection-diagnostics.csv",
    stacking_weights: "stacking-weights.csv",
    multiscale_ensemble_scales: "multiscale-ensemble-scales.csv",
  });
  const V5_FX_RESEARCH_ARTIFACTS = Object.freeze([
    "fx_features", "fx_coverage", "fx_ablation_oos",
  ]);
  const V5_MODEL_CONDITIONED_RESEARCH_ARTIFACTS = Object.freeze([
    "model_conditioned_asset_outcomes", "model_conditioned_asset_statistics",
  ]);
  const FROZEN_V4_BASELINE_V3 = Object.freeze({
    result_version: V3_RESULT_VERSION,
    label_version: V3_LABEL_VERSION,
    model_version: V3_MODEL_VERSION,
    champion: "markov",
    payload_sha256: "de93c585117b2784750f586a4f84ad99964c63081b252ad7affd7a75bd797095",
    artifacts_inventory_sha256: "8ef3778cc8c36faff0c80e2bf094f1f11bd6966ab3b7b2d6edb84ba292aff6b9",
    captured_at: "2026-08-13",
  });
  const FROZEN_V4_STRUCTURAL_PREREGISTRATION = Object.freeze({
    path: "config/structural_v4.json",
    sha256: "2f53ada564efca770261f16ce6eb16ec9c9782bde014de7a7d85b7b24dbe407b",
  });
  const TRANSITION_HORIZONS = Object.freeze([1, 4, 13]);
  const DEFAULT_SNAP_NOTE = "비관측일은 직전 관측 주로 이동합니다.";
  const CHART_DIMENSIONS = Object.freeze({
    width: 1200,
    height: 560,
    margin: Object.freeze({ top: 42, right: 18, bottom: 76, left: 60 }),
    panelGap: 64,
    outcomeOffset: 26,
  });
  // Frozen v3/v4 rendering fallback only. Active v5 output must supply the
  // canonical labels, colours, and symbols in payload.states.
  const FROZEN_LEGACY_STATE_META = Object.freeze({
    risk_on: Object.freeze({ label: "Risk-on", ko: "위험 선호", symbol: "↗", short: "↗" }),
    transition: Object.freeze({ label: "Transition", ko: "전환", symbol: "◆", short: "◆" }),
    risk_off: Object.freeze({ label: "Risk-off", ko: "위험 회피", symbol: "↘", short: "↘" }),
  });

  const STATUS_META = Object.freeze({
    ok: Object.freeze({ label: "정상", symbol: "✓", className: "status-ok", severity: 0 }),
    low: Object.freeze({ label: "낮음", symbol: "✓", className: "status-low", severity: 0 }),
    unknown: Object.freeze({ label: "확인 필요", symbol: "?", className: "status-unknown", severity: 1 }),
    degraded: Object.freeze({ label: "저하", symbol: "!", className: "status-degraded", severity: 2 }),
    stale: Object.freeze({ label: "지연", symbol: "!", className: "status-stale", severity: 2 }),
    medium: Object.freeze({ label: "주의", symbol: "!", className: "status-medium", severity: 2 }),
    error: Object.freeze({ label: "오류", symbol: "×", className: "status-error", severity: 3 }),
    blocked: Object.freeze({ label: "차단", symbol: "×", className: "status-blocked", severity: 3 }),
    high: Object.freeze({ label: "높음", symbol: "×", className: "status-high", severity: 3 }),
  });

  const HEALTH_LABELS = Object.freeze({
    ok: "정상",
    ready: "준비됨",
    stale: "지연",
    degraded: "저하",
    partial: "부분 가용",
    quota_exhausted: "할당량 소진",
    schema_changed: "스키마 변경",
    revision_gap: "빈티지 누락",
    rights_unconfirmed: "권리 미확인",
    license_blocked: "라이선스 차단",
    unavailable: "사용 불가",
    error: "오류",
  });

  const FACTOR_META = Object.freeze({
    trend: "추세",
    stress: "시장 스트레스",
    macro: "거시경제",
    financial_conditions: "금융 여건",
  });

  const MARKET_LABELS = Object.freeze({
    spy_trend_26w: "SPY 26주 추세",
    spy_realized_vol_13w: "SPY 13주 변동성",
    spy_drawdown_52w: "SPY 52주 고점 대비",
    gics_sector_breadth_4w: "섹터 상승 비중 · 4주",
    hyg_lqd_relative_13w: "HYG − LQD · 13주",
    anfci_change_4w: "ANFCI 변화 · 4주",
    close: "시장 종가",
    weekly_return: "주간 수익률",
    spy_weekly_return: "SPY 주간 수익률",
    return_1w: "1주 수익률",
    realized_vol_13w: "13주 실현 변동성",
    downside_vol_13w: "13주 하방 변동성",
    drawdown_52w: "52주 고점 대비 낙폭",
    drawdown: "고점 대비 낙폭",
    volume: "거래량",
    breadth: "시장 폭",
    rsp_spy: "RSP / SPY",
    iwm_spy: "IWM / SPY",
  });

  const state = {
    raw: null,
    comparisonSummary: null,
    selectionFamilyAudit: null,
    weekly: [],
    selectedIndex: -1,
    historyWindow: 52,
    preferredHistoryWindow: 52,
    validationWarnings: [],
    chartHistory: [],
    chartPinnedDate: null,
    chartPreviewDate: null,
    forecastExpiryTimer: null,
    transitionHorizon: 1,
    outcomeAsset: "SPY",
    outcomeHorizon: 13,
    outcomeBasis: "observed",
    comparisonModel: null,
  };

  const dom = {};

  class DataContractError extends Error {
    constructor(messages) {
      super(messages.join(" "));
      this.name = "DataContractError";
      this.messages = messages;
    }
  }

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function hasExactKeys(value, expectedKeys) {
    if (!isObject(value)) return false;
    const actualKeys = Object.keys(value);
    return actualKeys.length === expectedKeys.length
      && actualKeys.every((key) => expectedKeys.includes(key));
  }

  function isLowerSha256(value) {
    return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
  }

  function isIsoTimestamp(value) {
    return typeof value === "string" && value.length > 0 && Number.isFinite(Date.parse(value));
  }

  function finiteNumber(value) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
    return null;
  }

  function strictFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function probability(value) {
    const number = finiteNumber(value);
    return number !== null && number >= 0 && number <= 1 ? number : null;
  }

  function strictProbability(value) {
    const number = strictFiniteNumber(value);
    return number !== null && number >= 0 && number <= 1 ? number : null;
  }

  function isIsoDate(value) {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    const parsed = new Date(`${value}T00:00:00Z`);
    return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
  }

  function isoDateOffset(value, days) {
    if (!isIsoDate(value) || !Number.isInteger(days)) return null;
    const date = new Date(`${value}T00:00:00Z`);
    date.setUTCDate(date.getUTCDate() + days);
    return date.toISOString().slice(0, 10);
  }

  function firstValue(object, keys) {
    if (!isObject(object)) return null;
    for (const key of keys) {
      const value = object[key];
      if (value !== undefined && value !== null && value !== "") return value;
    }
    return null;
  }

  function textValue(value, fallback = "—") {
    if (value === null || value === undefined || value === "") return fallback;
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }
    return fallback;
  }

  function resultIdentity(payload) {
    const meta = isObject(payload && payload.meta) ? payload.meta : {};
    const model = isObject(payload && payload.model) ? payload.model : {};
    const execution = isObject(model.execution_parameters) ? model.execution_parameters : {};
    const mode = typeof meta.mode === "string" ? meta.mode.trim().toLowerCase() : "";
    const rawProfile = firstValue(execution, ["profile"]) || firstValue(model, ["profile"]);
    const normalizedProfile = typeof rawProfile === "string" ? rawProfile.trim().toLowerCase() : "";
    const profile = ["quick", "standard", "full"].includes(normalizedProfile)
      ? normalizedProfile.toUpperCase()
      : null;

    if (mode === "demo") {
      return {
        mode,
        profile,
        label: ["모의자료", profile, "파이프라인 검증"].filter(Boolean).join(" · "),
      };
    }
    if (mode === "live") {
      const labels = ["실데이터", profile].filter(Boolean);
      const lifecycle = isObject(model.lifecycle) ? model.lifecycle : {};
      const deployment = isObject(lifecycle.deployment) ? lifecycle.deployment.status : null;
      if (meta.publication_status === V5_PUBLICATION_STATUS) {
        labels.push("공개 운영");
      } else if (deployment === "reviewed") {
        labels.push("검토 완료 · 미배포");
      } else if (model.selection_status === "selected_by_gate") {
        labels.push("연구 후보 · 미배포");
      }
      return {
        mode,
        profile,
        label: labels.join(" · "),
      };
    }
    return { mode: "unknown", profile, label: "결과 유형 확인 필요" };
  }

  function fxStatusLabel(status) {
    return FX_STATUS_LABELS[status] || "확인 필요";
  }

  function modelHealthReasonLabels(reasons) {
    return Array.isArray(reasons)
      ? reasons.map((reason) => MODEL_HEALTH_REASON_LABELS[reason]).filter(Boolean)
      : [];
  }

  function splitHasExactMarkovParity(split) {
    if (!isObject(split) || !isObject(split.probability_parity)) return false;
    const numeric = split.probability_parity.probability_numeric;
    const tokens = split.probability_parity.probability_token_bytes;
    const deltas = split.delta_left_minus_right;
    return isObject(numeric)
      && isObject(tokens)
      && isObject(deltas)
      && numeric.exact_float_parity === true
      && numeric.maximum_absolute_difference === 0
      && numeric.mismatch_rows === 0
      && numeric.mismatch_values === 0
      && tokens.exact_parity === true
      && tokens.mismatch_rows === 0
      && tokens.mismatch_values === 0
      && ["log_loss", "brier", "balanced_accuracy", "fallback_rate"]
        .every((metric) => deltas[metric] === 0);
  }

  function validateV5ComparisonSummary(report, payload, payloadSha256) {
    if (!isObject(report) || !isObject(payload) || !isLowerSha256(payloadSha256)) return null;
    if (
      report.schema_version !== "regime-v5-v4-matched-comparison/1"
      || report.report_role !== "derived_only_diagnostic_comparison"
      || report.promotion_interpretation !== "prohibited"
    ) return null;
    const inputs = report.inputs;
    const model = payload.model;
    const baseline = isObject(model) ? model.baseline_v4 : null;
    const selectionDiagnostics = isObject(model) ? model.selection_diagnostics : null;
    const gate = isObject(report.v5_causal_multiscale_ensemble_vs_v5_markov)
      ? report.v5_causal_multiscale_ensemble_vs_v5_markov.selection_gate_crosscheck
      : null;
    const thresholds = Array.isArray(selectionDiagnostics)
      ? [...new Set(selectionDiagnostics.map((row) => (
        isObject(row) ? strictFiniteNumber(row.minimum_log_loss_improvement) : null
      )))]
      : [];
    const selectionThreshold = thresholds.length === 1 ? thresholds[0] : null;
    const gateMatchesThreshold = isObject(gate) && (
      (
        selectionThreshold === 0.05
        && gate.artifact_role === "selection_only_existing_champion_gate"
        && typeof gate.pairwise_gate_against_markov === "boolean"
        && !("multiscale_gate_against_selection_reference" in gate)
      )
      || (
        selectionThreshold === 0.01
        && gate.artifact_role === "selection_family_independently_recomputed"
        && typeof gate.multiscale_gate_against_selection_reference === "boolean"
        && !("pairwise_gate_against_markov" in gate)
      )
    );
    if (
      !isObject(inputs)
      || !isObject(inputs.v5)
      || !isObject(inputs.v5.regime_results)
      || inputs.v5.regime_results.sha256 !== payloadSha256
      || !isObject(inputs.frozen_v4)
      || !isObject(inputs.frozen_v4.sha256sums)
      || !isObject(baseline)
      || inputs.frozen_v4.sha256sums.sha256 !== baseline.artifacts_inventory_sha256
      || !gateMatchesThreshold
    ) return null;
    const comparison = report.v5_markov_vs_frozen_v4_markov;
    if (!isObject(comparison) || !isObject(comparison.common_keys)) return null;
    const commonKeys = strictFiniteNumber(comparison.common_keys.count);
    const selection = comparison.primary_selection;
    const holdout = comparison.post_selection_holdout;
    const selectionKeys = isObject(selection) && isObject(selection.common_keys)
      ? strictFiniteNumber(selection.common_keys.count)
      : null;
    const holdoutKeys = isObject(holdout) && isObject(holdout.common_keys)
      ? strictFiniteNumber(holdout.common_keys.count)
      : null;
    if (
      !Number.isInteger(commonKeys)
      || commonKeys <= 0
      || !Number.isInteger(selectionKeys)
      || !Number.isInteger(holdoutKeys)
      || selectionKeys + holdoutKeys !== commonKeys
      || !splitHasExactMarkovParity(selection)
      || !splitHasExactMarkovParity(holdout)
    ) return null;
    return Object.freeze({ commonKeys, selectionKeys, holdoutKeys, exactParity: true });
  }

  async function sha256Text(value) {
    if (
      typeof TextEncoder === "undefined"
      || typeof globalThis === "undefined"
      || !globalThis.crypto
      || !globalThis.crypto.subtle
    ) return null;
    const digest = await globalThis.crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(value),
    );
    return [...new Uint8Array(digest)]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }

  async function loadV5ComparisonSummary(payload, payloadText) {
    if (!isObject(payload.meta) || payload.meta.result_version !== V5_RESULT_VERSION) return null;
    const payloadSha256 = await sha256Text(payloadText);
    if (!payloadSha256) return null;
    try {
      const response = await fetch(V5_COMPARISON_URL, {
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) return null;
      return validateV5ComparisonSummary(await response.json(), payload, payloadSha256);
    } catch (_error) {
      return null;
    }
  }

  function canonicalJsonText(value) {
    if (value === null || typeof value === "boolean" || typeof value === "string") {
      return JSON.stringify(value);
    }
    if (typeof value === "number") {
      if (!Number.isFinite(value)) throw new TypeError("canonical JSON cannot contain non-finite numbers");
      return JSON.stringify(value);
    }
    if (Array.isArray(value)) return `[${value.map(canonicalJsonText).join(",")}]`;
    if (isObject(value)) {
      return `{${Object.keys(value).sort().map((key) => (
        `${JSON.stringify(key)}:${canonicalJsonText(value[key])}`
      )).join(",")}}`;
    }
    throw new TypeError("canonical JSON contains an unsupported value");
  }

  function canonicalJsonFromSource(source, omittedTopLevelKey = null) {
    if (typeof source !== "string") throw new TypeError("JSON source must be a string");
    let index = 0;
    const skipWhitespace = () => {
      while (/\s/.test(source[index] || "")) index += 1;
    };
    const parseStringToken = () => {
      const start = index;
      index += 1;
      while (index < source.length) {
        if (source[index] === "\\") {
          index += 2;
          continue;
        }
        if (source[index] === '"') {
          index += 1;
          const decoded = JSON.parse(source.slice(start, index));
          return { decoded, text: JSON.stringify(decoded) };
        }
        index += 1;
      }
      throw new SyntaxError("unterminated JSON string");
    };
    const parseValue = (depth) => {
      skipWhitespace();
      const token = source[index];
      if (token === '"') return parseStringToken().text;
      if (token === "[") {
        index += 1;
        const items = [];
        skipWhitespace();
        if (source[index] === "]") {
          index += 1;
          return "[]";
        }
        while (index < source.length) {
          items.push(parseValue(depth + 1));
          skipWhitespace();
          if (source[index] === "]") {
            index += 1;
            return `[${items.join(",")}]`;
          }
          if (source[index] !== ",") throw new SyntaxError("invalid JSON array");
          index += 1;
        }
        throw new SyntaxError("unterminated JSON array");
      }
      if (token === "{") {
        index += 1;
        const entries = [];
        const keys = new Set();
        skipWhitespace();
        if (source[index] === "}") {
          index += 1;
          return "{}";
        }
        while (index < source.length) {
          skipWhitespace();
          if (source[index] !== '"') throw new SyntaxError("invalid JSON object key");
          const key = parseStringToken();
          if (keys.has(key.decoded)) throw new SyntaxError("duplicate JSON object key");
          keys.add(key.decoded);
          skipWhitespace();
          if (source[index] !== ":") throw new SyntaxError("invalid JSON object separator");
          index += 1;
          const value = parseValue(depth + 1);
          if (!(depth === 0 && key.decoded === omittedTopLevelKey)) {
            entries.push({ key: key.decoded, text: `${key.text}:${value}` });
          }
          skipWhitespace();
          if (source[index] === "}") {
            index += 1;
            entries.sort((left, right) => left.key < right.key ? -1 : left.key > right.key ? 1 : 0);
            return `{${entries.map((entry) => entry.text).join(",")}}`;
          }
          if (source[index] !== ",") throw new SyntaxError("invalid JSON object");
          index += 1;
        }
        throw new SyntaxError("unterminated JSON object");
      }
      const scalar = source.slice(index).match(/^(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)/);
      if (!scalar) throw new SyntaxError("invalid JSON scalar");
      index += scalar[0].length;
      return scalar[0];
    };
    const result = parseValue(0);
    skipWhitespace();
    if (index !== source.length) throw new SyntaxError("trailing JSON content");
    return result;
  }

  function safeRelativeArtifactPath(value) {
    return typeof value === "string"
      && value.length > 0
      && !value.startsWith("/")
      && !value.split("/").includes("..")
      && !value.includes("\\");
  }

  function validateSelectionFamilyAuditSemantics(report, payload) {
    if (!isObject(report) || !isObject(payload)) return null;
    const meta = isObject(payload.meta) ? payload.meta : {};
    const selection = isObject(payload.selection) ? payload.selection : {};
    const model = isObject(payload.model) ? payload.model : {};
    const topLevelFields = [
      "schema_version", "status", "generation_id", "evidence_track", "evidence_status",
      "candidate_manifest_sha256", "selection_period", "source_artifacts",
      "candidate_count", "candidate_set", "champion", "runner_up",
      "selection_reason", "policy_sha256", "complexity_registry",
      "complexity_registry_sha256", "fallback", "common_origin_contract",
      "candidates", "supplemental_evaluation", "sha256",
    ];
    if (
      !hasExactKeys(report, topLevelFields)
      || meta.result_version !== V5_RESULT_VERSION
      || report.schema_version !== "selection-family-audit/v2"
      || report.status !== "completed"
      || report.generation_id !== meta.generation_id
      || !isLowerSha256(report.sha256)
      || report.candidate_manifest_sha256 !== model.candidate_manifest_sha256
      || report.policy_sha256 !== selection.policy_sha256
      || report.complexity_registry_sha256 !== selection.complexity_registry_sha256
      || report.evidence_track !== payload.forecast?.evidence_track
      || (
        report.evidence_track === "operational_oos"
          ? report.evidence_status !== "operational_oos"
          : !["historical_reconstructed_oos", "synthetic_fixture"].includes(report.evidence_status)
      )
      || report.champion !== modelName(model.champion)
      || report.runner_up !== selection.runner_up
      || report.selection_reason !== selection.selection_reason
    ) return null;

    const selectionPeriod = report.selection_period;
    const selectionTimestamps = isObject(selectionPeriod)
      ? [
        selectionPeriod.selection_end_at,
        selectionPeriod.first_origin_at,
        selectionPeriod.last_origin_at,
        selectionPeriod.first_target_at,
        selectionPeriod.last_target_at,
      ]
      : [];
    if (
      !hasExactKeys(selectionPeriod, [
        "role", "declared", "selection_end_at", "first_origin_at",
        "last_origin_at", "first_target_at", "last_target_at",
      ])
      || selectionPeriod.role !== "predeployment_selection_only"
      || typeof selectionPeriod.declared !== "string"
      || !selectionPeriod.declared
      || (typeof model.selection_period === "string" && selectionPeriod.declared !== model.selection_period)
      || selectionTimestamps.some((value) => !isZonedIsoTimestamp(value))
      || Date.parse(selectionPeriod.first_origin_at) > Date.parse(selectionPeriod.last_origin_at)
      || Date.parse(selectionPeriod.first_target_at) > Date.parse(selectionPeriod.last_target_at)
      || Date.parse(selectionPeriod.last_target_at) >= Date.parse(selectionPeriod.selection_end_at)
    ) return null;

    const sourceArtifacts = report.source_artifacts;
    const payloadArtifacts = isObject(model.core_artifacts) ? model.core_artifacts : {};
    if (!hasExactKeys(sourceArtifacts, ["selection_diagnostics", "oos_predictions"])) return null;
    for (const key of ["selection_diagnostics", "oos_predictions"]) {
      const record = sourceArtifacts[key];
      if (
        !hasExactKeys(record, ["path", "sha256", "row_count"])
        || !safeRelativeArtifactPath(record.path)
        || !isLowerSha256(record.sha256)
        || !Number.isInteger(record.row_count)
        || record.row_count <= 0
        || !isObject(payloadArtifacts[key])
        || record.path !== payloadArtifacts[key].path
        || record.sha256 !== payloadArtifacts[key].sha256
      ) return null;
    }
    const candidates = Array.isArray(report.candidate_set) ? report.candidate_set : [];
    const payloadCandidates = Array.isArray(selection.candidate_set) ? selection.candidate_set : [];
    const rows = Array.isArray(report.candidates) ? report.candidates : [];
    const registry = isObject(report.complexity_registry) ? report.complexity_registry : {};
    if (
      candidates.length === 0
      || report.candidate_count !== candidates.length
      || JSON.stringify(candidates) !== JSON.stringify(payloadCandidates)
      || JSON.stringify(rows.map((row) => isObject(row) ? row.model : null)) !== JSON.stringify(candidates)
      || Object.keys(registry).length !== candidates.length
      || candidates.some((name) => !Object.prototype.hasOwnProperty.call(registry, name))
    ) return null;

    const selectedRows = rows.filter((row) => isObject(row) && row.selected === true);
    const runnerUpRows = rows.filter((row) => isObject(row) && row.runner_up === true);
    const validRows = rows.every((row, index) => (
      isObject(row)
      && row.candidate_order === index + 1
      && row.model === candidates[index]
      && typeof row.selected === "boolean"
      && typeof row.runner_up === "boolean"
      && typeof row.is_reference === "boolean"
      && Number.isInteger(row.complexity_rank)
      && row.complexity_rank >= 0
      && registry[row.model] === row.complexity_rank
      && isObject(row.gate)
      && typeof row.gate.passed_all === "boolean"
      && typeof row.gate.reason === "string"
      && Array.isArray(row.gate.failed_checks)
      && Number.isInteger(row.gate.fallback_count)
      && row.gate.fallback_count >= 0
      && isObject(row.metrics)
      && strictFiniteNumber(row.metrics.log_loss) !== null
      && row.metrics.log_loss >= 0
      && strictFiniteNumber(row.metrics.brier) !== null
      && row.metrics.brier >= 0
    ));
    if (
      !validRows
      || rows.filter((row) => row.is_reference === true).length !== 1
      || selectedRows.length !== 1
      || selectedRows[0].model !== report.champion
      || selectedRows[0].gate.passed_all !== true
      || (report.runner_up === null && runnerUpRows.length !== 0)
      || (report.runner_up !== null && (
        runnerUpRows.length !== 1
        || runnerUpRows[0].model !== report.runner_up
        || runnerUpRows[0].gate.passed_all !== true
      ))
    ) return null;

    const fallback = report.fallback;
    if (
      fallback !== null
      && (
        !hasExactKeys(fallback, ["model", "trigger", "reason"])
        || !candidates.includes(fallback.model)
        || typeof fallback.trigger !== "string"
        || !fallback.trigger
        || typeof fallback.reason !== "string"
        || !fallback.reason
      )
    ) return null;

    const origins = report.common_origin_contract;
    const originColumns = [
      "origin_date", "target_date", "evaluation_split", "current_state",
      "actual", "train_size", "gap",
    ];
    if (
      !isObject(origins)
      || origins.status !== "matched"
      || JSON.stringify(origins.columns) !== JSON.stringify(originColumns)
      || !Number.isInteger(origins.origin_count)
      || origins.origin_count <= 0
      || !isZonedIsoTimestamp(origins.first_origin_at)
      || !isZonedIsoTimestamp(origins.last_origin_at)
      || Date.parse(origins.first_origin_at) > Date.parse(origins.last_origin_at)
      || origins.first_origin_at !== selectionPeriod.first_origin_at
      || origins.last_origin_at !== selectionPeriod.last_origin_at
      || !isLowerSha256(origins.origins_sha256)
      || sourceArtifacts.selection_diagnostics.row_count !== candidates.length
      || sourceArtifacts.oos_predictions.row_count < origins.origin_count * candidates.length
    ) return null;

    const supplemental = report.supplemental_evaluation;
    if (
      !isObject(supplemental)
      || supplemental.schema_version !== "regime-selection-evaluation/1"
      || supplemental.status !== "completed"
      || supplemental.evidence_status !== report.evidence_status
      || supplemental.role !== "supplemental_not_selection_gate"
      || supplemental.evaluation_split !== "selection"
      || supplemental.holdout_rows_used !== 0
      || supplemental.selection_effect !== "none"
      || supplemental.selected_champion_unchanged !== report.champion
      || JSON.stringify(supplemental.candidate_set) !== JSON.stringify(candidates)
      || !isObject(supplemental.common_origin_contract)
      || supplemental.common_origin_contract.status !== "matched"
      || supplemental.common_origin_contract.origin_count !== origins.origin_count
      || supplemental.common_origin_contract.first_origin_at !== origins.first_origin_at
      || supplemental.common_origin_contract.last_origin_at !== origins.last_origin_at
      || !isObject(supplemental.primary_metric_crosscheck)
      || supplemental.primary_metric_crosscheck.status !== "matched"
      || supplemental.primary_metric_crosscheck.changes_holm_gate !== false
      || supplemental.primary_metric_crosscheck.changes_champion !== false
      || !isLowerSha256(supplemental.sha256)
    ) return null;

    return Object.freeze({
      source: "selection-family-audit/v2",
      candidateCount: candidates.length,
      candidates: Object.freeze([...candidates]),
      champion: report.champion,
      runnerUp: report.runner_up,
      selectionReason: report.selection_reason,
      policySha256: report.policy_sha256,
      evidenceTrack: report.evidence_track,
      evidenceStatus: report.evidence_status,
      originCount: origins.origin_count,
      fallback: isObject(report.fallback) ? Object.freeze({ ...report.fallback }) : null,
    });
  }

  async function validateSelectionFamilyAudit(report, payload, sourceText = null) {
    const validated = validateSelectionFamilyAuditSemantics(report, payload);
    if (!validated) return null;
    const body = { ...report };
    delete body.sha256;
    const supplementalBody = { ...report.supplemental_evaluation };
    delete supplementalBody.sha256;
    let hashes = null;
    try {
      hashes = await Promise.all([
        sha256Text(
          typeof sourceText === "string"
            ? canonicalJsonFromSource(sourceText, "sha256")
            : canonicalJsonText(body),
        ),
        sha256Text(canonicalJsonText(report.complexity_registry)),
        typeof sourceText === "string"
          ? Promise.resolve(report.supplemental_evaluation.sha256)
          : sha256Text(canonicalJsonText(supplementalBody)),
      ]);
    } catch (_error) {
      return null;
    }
    return (
      hashes[0] === report.sha256
      && hashes[1] === report.complexity_registry_sha256
      && hashes[2] === report.supplemental_evaluation.sha256
    ) ? validated : null;
  }

  async function loadSelectionFamilyAudit(payload) {
    if (!isObject(payload.meta) || payload.meta.result_version !== V5_RESULT_VERSION) return null;
    try {
      const response = await fetch(V5_SELECTION_FAMILY_AUDIT_URL, {
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) return null;
      const sourceText = await response.text();
      return await validateSelectionFamilyAudit(JSON.parse(sourceText), payload, sourceText);
    } catch (_error) {
      return null;
    }
  }

  function selectionEvidenceForDisplay(payload, audit = null) {
    if (isObject(audit) && audit.source === "selection-family-audit/v2") return audit;
    const selection = isObject(payload) && isObject(payload.selection) ? payload.selection : {};
    const candidates = Array.isArray(selection.candidate_set)
      ? selection.candidate_set.filter((name) => typeof name === "string" && name)
      : forecastComparisonModels(payload);
    return Object.freeze({
      source: "payload",
      candidateCount: candidates.length,
      candidates: Object.freeze([...candidates]),
      champion: modelName(isObject(payload?.model) ? payload.model.champion : null),
      runnerUp: selection.runner_up ?? null,
      selectionReason: selection.selection_reason ?? null,
      policySha256: selection.policy_sha256 ?? null,
      evidenceTrack: isObject(payload?.forecast) ? payload.forecast.evidence_track : null,
      evidenceStatus: null,
      originCount: null,
      fallback: null,
    });
  }

  function createElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function createSvg(tag, attributes = {}) {
    const element = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attributes)) {
      element.setAttribute(key, String(value));
    }
    return element;
  }

  function setText(element, value, fallback = "—") {
    element.textContent = textValue(value, fallback);
  }

  function formatPercent(value, digits = 1) {
    const number = probability(value);
    if (number === null) return "—";
    return formatSignedPercent(number, digits);
  }

  function formatSignedPercent(value, digits = 1) {
    const number = finiteNumber(value);
    if (number === null) return "—";
    const factor = 10 ** (digits + 2);
    const rounded = Math.round(number * factor) / factor;
    const normalized = Object.is(rounded, -0) ? 0 : rounded;
    return new Intl.NumberFormat("ko-KR", {
      style: "percent",
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).format(normalized);
  }

  function formatNumber(value, digits = 2) {
    const number = finiteNumber(value);
    if (number === null) return "—";
    const factor = 10 ** digits;
    const rounded = Math.round(number * factor) / factor;
    const normalized = Object.is(rounded, -0) ? 0 : rounded;
    return new Intl.NumberFormat("ko-KR", {
      maximumFractionDigits: digits,
      minimumFractionDigits: 0,
    }).format(normalized);
  }

  function formatCompactNumber(value, digits = 2) {
    const number = finiteNumber(value);
    if (number === null) return "—";
    return new Intl.NumberFormat("ko-KR", {
      notation: Math.abs(number) >= 1000 ? "compact" : "standard",
      maximumFractionDigits: digits,
    }).format(number);
  }

  function parseDate(value) {
    return isIsoDate(value) ? new Date(`${value}T00:00:00Z`) : null;
  }

  function formatDate(value, includeYear = true) {
    const date = parseDate(value);
    if (!date) return textValue(value);
    return new Intl.DateTimeFormat("ko-KR", {
      timeZone: "UTC",
      year: includeYear ? "numeric" : undefined,
      month: "short",
      day: "numeric",
    }).format(date);
  }

  function formatDateTime(value) {
    if (!value) return "—";
    if (isIsoDate(value)) return formatDate(value);
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return textValue(value);
    const configuredTimeZone = isObject(state.raw && state.raw.meta)
      ? firstValue(state.raw.meta, ["timezone"])
      : null;
    const timeZone = typeof configuredTimeZone === "string" && configuredTimeZone
      ? configuredTimeZone
      : "America/New_York";
    return new Intl.DateTimeFormat("ko-KR", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZone,
      timeZoneName: "short",
    }).format(date);
  }

  function displayFreshness(dataAsOf, maximumAgeDays = 10, nowMilliseconds = Date.now()) {
    const cutoff = Date.parse(dataAsOf);
    const now = finiteNumber(nowMilliseconds);
    const maximumAge = Number.isInteger(maximumAgeDays) && maximumAgeDays >= 0
      ? maximumAgeDays
      : 10;
    if (!Number.isFinite(cutoff) || now === null) return null;
    const ageDays = Math.max(0, Math.floor((now - cutoff) / 86400000));
    return {
      age_days: ageDays,
      maximum_age_days: maximumAge,
      status: ageDays <= maximumAge ? "current" : "stale",
    };
  }

  function publicationSnapshotLabel() {
    const meta = isObject(state.raw && state.raw.meta) ? state.raw.meta : {};
    const cutoff = firstValue(meta, ["data_as_of", "dataAsOf", "cutoff_at"]);
    const latest = state.weekly.length ? state.weekly[state.weekly.length - 1] : null;
    const cutoffDate = typeof cutoff === "string" && isIsoDate(cutoff.slice(0, 10))
      ? cutoff.slice(0, 10)
      : cutoff;
    const date = (latest && latest.date) || cutoffDate;
    return date ? `공개 스냅샷 ${formatDate(date)}` : "공개 스냅샷";
  }

  function humanizeKey(key) {
    return String(key)
      .replace(/_/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function stateMeta(code, payload = state.raw) {
    const rows = isObject(payload) && Array.isArray(payload.states) ? payload.states : [];
    const supplied = rows.find((row) => isObject(row) && row.id === code);
    if (supplied) {
      const symbol = textValue(supplied.symbol, "?");
      return {
        label: textValue(supplied.label, textValue(code, "Unknown")),
        ko: textValue(firstValue(supplied, ["label_ko", "ko"]), "알 수 없음"),
        description: textValue(supplied.description, ""),
        color: textValue(supplied.color, ""),
        symbol,
        short: symbol,
      };
    }
    return FROZEN_LEGACY_STATE_META[code]
      || { label: textValue(code, "Unknown"), ko: "알 수 없음", symbol: "?", short: "?", color: "" };
  }

  function applyPayloadStateTheme(payload = state.raw) {
    if (typeof document === "undefined" || !document.documentElement) return;
    for (const code of STATE_ORDER) {
      const color = stateMeta(code, payload).color;
      if (/^#[0-9a-f]{6}$/i.test(color)) {
        document.documentElement.style.setProperty(`--${code.replaceAll("_", "-")}`, color);
      }
    }
    for (const element of document.querySelectorAll("[data-state-label]")) {
      element.textContent = stateMeta(element.dataset.stateLabel, payload).label;
    }
    for (const element of document.querySelectorAll("[data-state-symbol]")) {
      element.textContent = stateMeta(element.dataset.stateSymbol, payload).symbol;
    }
  }

  function formatDurationSeconds(value) {
    const seconds = strictFiniteNumber(value);
    if (seconds === null || seconds < 0) return "—";
    if (seconds === 0) return "0시간";
    const totalHours = Math.floor(seconds / 3600);
    const days = Math.floor(totalHours / 24);
    const hours = totalHours % 24;
    return [days ? `${days}일` : null, hours || !days ? `${hours}시간` : null]
      .filter(Boolean)
      .join(" ");
  }

  function selectedForecastIsHistorical() {
    return state.selectedIndex >= 0
      && state.weekly.length > 0
      && state.selectedIndex < state.weekly.length - 1;
  }

  function forecastSurfacePolicy(
    payload,
    selectedIndex,
    weeklyLength,
    nowMilliseconds = Date.now(),
  ) {
    const resultVersion = isObject(payload) && isObject(payload.meta)
      ? payload.meta.result_version
      : null;
    const latestSelection = Number.isInteger(selectedIndex)
      && Number.isInteger(weeklyLength)
      && weeklyLength > 0
      && selectedIndex === weeklyLength - 1;
    const expiredLatest = resultVersion === V5_RESULT_VERSION
      && latestSelection
      && !forecastAvailability(payload, nowMilliseconds).current;
    return Object.freeze({
      expiredLatest,
      showCurrentForecast: !expiredLatest,
      preserveHistory: true,
    });
  }

  function applyExpiredForecastDomState(elements, policy) {
    if (!isObject(elements) || !isObject(policy)) return;
    if (policy.expiredLatest) {
      for (const id of [
        "next-regime-card",
        "transition-card",
        "model-forecast-field",
        "model-forecast-explorer",
      ]) {
        if (elements[id]) elements[id].hidden = true;
      }
    }
    if (policy.preserveHistory && elements.history) elements.history.hidden = false;
  }

  function suppressCurrentForecastSurface() {
    return forecastSurfacePolicy(
      state.raw,
      state.selectedIndex,
      state.weekly.length,
    ).expiredLatest;
  }

  function scheduleForecastExpiryRefresh(availability) {
    if (typeof window === "undefined" || typeof window.setTimeout !== "function") return;
    if (state.forecastExpiryTimer !== null) {
      window.clearTimeout(state.forecastExpiryTimer);
      state.forecastExpiryTimer = null;
    }
    if (!availability.current || availability.remaining_seconds <= 0) return;
    const delay = Math.min(2147483647, availability.remaining_seconds * 1000 + 250);
    state.forecastExpiryTimer = window.setTimeout(() => {
      state.forecastExpiryTimer = null;
      if (selectedWeek()) renderSelectedWeek();
    }, delay);
  }

  function renderContractOverview() {
    const isV5 = isV5Payload();
    dom["contract-overview-grid"].hidden = !isV5;
    if (!isV5) return;

    const label = isObject(state.raw.label) ? state.raw.label : {};
    const shortHash = typeof label.spec_sha256 === "string" ? label.spec_sha256.slice(0, 8) : "hash 없음";
    setText(dom["label-spec-identity"], `${textValue(label.spec_version, "정의 미기재")} · ${shortHash}`);
    const membershipText = label.membership_semantics === "distance_to_anchor_not_posterior"
      ? "관측 소속도는 임계 anchor와의 거리이며 posterior가 아닙니다. 다음 주 값은 t까지의 정보로 만든 t+1 예측확률입니다."
      : "관측 소속도와 다음 주 예측확률은 서로 다른 값입니다.";
    setText(dom["membership-definition"], membershipText);

    const forecast = isObject(state.raw.forecast) ? state.raw.forecast : {};
    const availability = forecastAvailability(state.raw);
    const evidenceLabel = forecast.evidence_track === "operational_oos"
      ? "실제 운영 OOS"
      : forecast.evidence_track === "reconstructed_oos"
        ? "재구성 OOS"
        : "근거 트랙 미기재";
    const currentLabel = availability.current ? "예측 유효" : "예측기간 종료";
    setText(dom["forecast-contract-status"], `${currentLabel} · ${evidenceLabel}`);
    dom["forecast-contract-status"].classList.toggle("is-expired", !availability.current);
    setText(dom["forecast-origin-at"], formatDateTime(forecast.origin_at));
    setText(dom["forecast-decision-at"], forecast.decision_at ? formatDateTime(forecast.decision_at) : "발행시각 없음");
    setText(dom["forecast-target-at"], formatDateTime(forecast.target_at));
    setText(dom["forecast-remaining-horizon"], formatDurationSeconds(availability.remaining_seconds));
    dom["forecast-expired-notice"].hidden = availability.current;
    dom["hero-results"].classList.toggle(
      "has-expired-current-forecast",
      !selectedForecastIsHistorical() && !availability.current,
    );
    scheduleForecastExpiryRefresh(availability);
  }

  function normalizeStatus(value) {
    let candidate = value;
    if (isObject(value)) {
      candidate = firstValue(value, ["status", "overall", "state", "quality_status", "level"]);
    }
    const normalized = String(candidate || "unknown").trim().toLowerCase().replace(/[\s-]+/g, "_");

    if (["ok", "ready", "healthy", "current", "available", "pass", "passed"].includes(normalized)) return "ok";
    if (["stale", "delayed", "lagged"].includes(normalized)) return "stale";
    if ([
      "degraded",
      "partial",
      "warning",
      "quota_exhausted",
      "schema_changed",
      "revision_gap",
      "missing_optional",
    ].includes(normalized)) return "degraded";
    if ([
      "blocked",
      "license_blocked",
      "rights_unconfirmed",
      "unavailable",
      "failed",
      "failure",
    ].includes(normalized)) return "blocked";
    if (["error", "invalid", "corrupt"].includes(normalized)) return "error";
    return "unknown";
  }

  function healthLabel(value) {
    let candidate = value;
    if (isObject(value)) {
      candidate = firstValue(value, ["status", "overall", "state", "quality_status", "level"]);
    }
    const raw = String(candidate || "unknown").trim().toLowerCase().replace(/[\s-]+/g, "_");
    return HEALTH_LABELS[raw] || STATUS_META[normalizeStatus(raw)].label;
  }

  function worstStatus(statuses) {
    if (!statuses.length) return "unknown";
    return statuses.reduce((worst, current) => {
      const normalized = normalizeStatus(current);
      return STATUS_META[normalized].severity > STATUS_META[worst].severity ? normalized : worst;
    }, "ok");
  }

  function setStatusBadge(element, status, customLabel) {
    const normalized = STATUS_META[status] ? status : normalizeStatus(status);
    const meta = STATUS_META[normalized] || STATUS_META.unknown;
    element.className = `status-badge ${meta.className}`;
    const symbol = createElement("span", "status-symbol", meta.symbol);
    symbol.setAttribute("aria-hidden", "true");
    const label = createElement("span", null, customLabel || meta.label);
    element.replaceChildren(symbol, label);
  }

  function validateStateVector(value, path, errors, noun) {
    if (!isObject(value)) {
      errors.push(`${path} ${noun} 객체가 없습니다.`);
      return null;
    }
    if (!hasExactKeys(value, STATE_ORDER)) {
      errors.push(`${path} ${noun} 키는 표준 세 상태와 정확히 일치해야 합니다.`);
      return null;
    }
    let sum = 0;
    let valid = true;
    for (const code of STATE_ORDER) {
      const resolved = strictProbability(value[code]);
      if (resolved === null) {
        errors.push(`${path}.${code} ${noun}가 0–1 범위의 숫자가 아닙니다.`);
        valid = false;
      } else {
        sum += resolved;
      }
    }
    if (valid && Math.abs(sum - 1) > 0.000001) errors.push(`${path} ${noun} 합계가 1이 아닙니다.`);
    return valid ? value : null;
  }

  function validateOptionalFinite(value, path, errors, minimum = null, maximum = null) {
    if (value === null || value === undefined) return;
    const number = strictFiniteNumber(value);
    if (
      number === null
      || (minimum !== null && number < minimum)
      || (maximum !== null && number > maximum)
    ) {
      errors.push(`${path} 값이 허용 범위의 유한한 숫자 또는 null이 아닙니다.`);
    }
  }

  function validateInteger(value, path, errors, minimum = 0) {
    if (!Number.isInteger(value) || value < minimum) errors.push(`${path}는 ${minimum} 이상의 정수여야 합니다.`);
  }

  function championSelectionEvidence(value) {
    const model = isObject(value) ? value : {};
    const champion = typeof model.champion === "string" && model.champion
      ? model.champion
      : null;
    const leaderboard = Array.isArray(model.leaderboard) ? model.leaderboard : [];
    const diagnostics = Array.isArray(model.selection_diagnostics)
      ? model.selection_diagnostics
      : [];
    const leaderboardSelections = leaderboard
      .filter((row) => isObject(row) && row.selected === true)
      .map((row) => row.name);
    const leaderboardChampions = leaderboard
      .filter((row) => isObject(row) && row.is_champion === true)
      .map((row) => row.name);
    const diagnosticSelections = diagnostics
      .filter((row) => isObject(row) && row.selected === true)
      .map((row) => row.model);
    const diagnostic = diagnostics.find(
      (row) => isObject(row) && row.model === champion,
    );
    return Object.freeze({
      champion,
      leaderboardSelections: Object.freeze(leaderboardSelections),
      leaderboardChampions: Object.freeze(leaderboardChampions),
      diagnosticSelections: Object.freeze(diagnosticSelections),
      valid: champion !== null
        && leaderboardSelections.length === 1
        && leaderboardSelections[0] === champion
        && leaderboardChampions.length === 1
        && leaderboardChampions[0] === champion
        && diagnosticSelections.length === 1
        && diagnosticSelections[0] === champion
        && isObject(diagnostic)
        && diagnostic.gate_passed === true,
    });
  }

  function isZonedIsoTimestamp(value) {
    return typeof value === "string"
      && /(?:Z|[+-]\d{2}:\d{2})$/.test(value)
      && Number.isFinite(Date.parse(value));
  }

  function validateV5StateDefinitions(rows, errors) {
    const fields = ["id", "label", "label_ko", "description", "color", "symbol"];
    if (!Array.isArray(rows) || rows.length !== STATE_ORDER.length) return;
    rows.forEach((row, index) => {
      const path = `states[${index}]`;
      if (!hasExactKeys(row, fields)) {
        errors.push(`${path} 메타데이터 필드가 v5 계약과 일치하지 않습니다.`);
        return;
      }
      if (row.id !== STATE_ORDER[index]) errors.push(`${path}.id 순서가 올바르지 않습니다.`);
      for (const field of ["label", "label_ko", "description", "symbol"]) {
        if (typeof row[field] !== "string" || !row[field].trim()) {
          errors.push(`${path}.${field}는 비어 있지 않은 문자열이어야 합니다.`);
        }
      }
      if (typeof row.color !== "string" || !/^#[0-9a-f]{6}$/i.test(row.color)) {
        errors.push(`${path}.color는 6자리 hex 색상이어야 합니다.`);
      }
    });
  }

  function validateV5LabelEnvelope(label, errors) {
    const fields = [
      "spec_id", "spec_version", "spec_sha256", "fit_period",
      "input_scope", "membership_semantics",
    ];
    if (!hasExactKeys(label, fields)) {
      errors.push("v5 label 필드가 계약과 일치하지 않습니다.");
      return;
    }
    if (label.spec_id !== "v1_spy_hysteresis" || label.spec_version !== V5_LABEL_VERSION) {
      errors.push("v5 label canonical spec이 올바르지 않습니다.");
    }
    if (!isLowerSha256(label.spec_sha256)) errors.push("v5 label.spec_sha256이 올바르지 않습니다.");
    if (label.input_scope !== "SPY adjusted close only") errors.push("v5 label.input_scope가 올바르지 않습니다.");
    if (label.membership_semantics !== "distance_to_anchor_not_posterior") {
      errors.push("v5 관측 소속도 의미가 올바르지 않습니다.");
    }
    const fit = label.fit_period;
    if (
      !hasExactKeys(fit, ["start", "end", "weeks"])
      || !isIsoDate(fit.start)
      || !isIsoDate(fit.end)
      || fit.end < fit.start
      || fit.weeks !== 520
    ) {
      errors.push("v5 label.fit_period가 올바르지 않습니다.");
    }
  }

  function forecastAvailability(payload, nowMilliseconds = Date.now()) {
    const forecast = isObject(payload) ? payload.forecast : null;
    const targetMilliseconds = isObject(forecast) ? Date.parse(forecast.target_at) : NaN;
    const now = strictFiniteNumber(nowMilliseconds);
    const contractActive = isObject(forecast) && forecast.status === "active";
    const current = contractActive
      && now !== null
      && Number.isFinite(targetMilliseconds)
      && now < targetMilliseconds;
    return Object.freeze({
      status: current ? "active" : contractActive ? "elapsed" : "expired",
      current,
      remaining_seconds: current
        ? Math.max(0, Math.floor((targetMilliseconds - now) / 1000))
        : 0,
    });
  }

  function validateV5ForecastEnvelope(forecast, mode, errors) {
    const fields = [
      "status", "origin_at", "decision_at", "target_at",
      "remaining_horizon", "evidence_track",
    ];
    if (!hasExactKeys(forecast, fields)) {
      errors.push("v5 forecast 필드가 계약과 일치하지 않습니다.");
      return;
    }
    if (!["active", "expired"].includes(forecast.status)) errors.push("v5 forecast.status가 올바르지 않습니다.");
    if (!["operational_oos", "reconstructed_oos"].includes(forecast.evidence_track)) {
      errors.push("v5 forecast.evidence_track이 올바르지 않습니다.");
    }
    const origin = isZonedIsoTimestamp(forecast.origin_at) ? Date.parse(forecast.origin_at) : NaN;
    const target = isZonedIsoTimestamp(forecast.target_at) ? Date.parse(forecast.target_at) : NaN;
    if (!Number.isFinite(origin) || !Number.isFinite(target) || target - origin !== 7 * 86400000) {
      errors.push("v5 forecast origin/target 1주 구간이 올바르지 않습니다.");
      return;
    }
    if (!Number.isInteger(forecast.remaining_horizon) || forecast.remaining_horizon < 0) {
      errors.push("v5 forecast.remaining_horizon이 올바르지 않습니다.");
      return;
    }
    if (forecast.status === "active") {
      const decision = isZonedIsoTimestamp(forecast.decision_at) ? Date.parse(forecast.decision_at) : NaN;
      if (!Number.isFinite(decision) || decision < origin || decision >= target) {
        errors.push("v5 forecast는 origin_at <= decision_at < target_at이어야 합니다.");
      } else if (forecast.remaining_horizon !== Math.floor((target - decision) / 1000)) {
        errors.push("v5 forecast.remaining_horizon이 decision_at과 일치하지 않습니다.");
      }
    } else if (forecast.decision_at !== null || forecast.remaining_horizon !== 0 || mode === "live") {
      errors.push("v5 expired forecast 상태가 운영 계약과 일치하지 않습니다.");
    }
  }

  function validateV5SelectionEnvelope(selection, model, errors) {
    const fields = [
      "schema_version", "status", "policy_sha256", "complexity_registry_sha256",
      "candidate_set", "runner_up", "selection_reason", "simplicity_tolerance",
      "tie_break_order", "operating_champion",
    ];
    if (!hasExactKeys(selection, fields)) {
      errors.push("v5 selection 필드가 계약과 일치하지 않습니다.");
      return;
    }
    if (selection.schema_version !== "regime-selection-evidence/1" || selection.status !== "selected_by_gate") {
      errors.push("v5 selection schema/status가 올바르지 않습니다.");
    }
    if (selection.status !== model.selection_status) errors.push("v5 selection status alias가 일치하지 않습니다.");
    if (!isLowerSha256(selection.policy_sha256) || !isLowerSha256(selection.complexity_registry_sha256)) {
      errors.push("v5 selection policy hash가 올바르지 않습니다.");
    }
    const candidates = Array.isArray(selection.candidate_set) ? selection.candidate_set : [];
    const diagnosticNames = Array.isArray(model.selection_diagnostics)
      ? model.selection_diagnostics.map((row) => isObject(row) ? row.model : null)
      : [];
    if (
      !candidates.length
      || candidates.some((name) => typeof name !== "string" || !name)
      || new Set(candidates).size !== candidates.length
      || JSON.stringify(candidates) !== JSON.stringify(diagnosticNames)
    ) {
      errors.push("v5 selection.candidate_set이 진단 후보와 일치하지 않습니다.");
    }
    if (
      selection.runner_up !== null
      && (!candidates.includes(selection.runner_up) || selection.runner_up === model.champion)
    ) {
      errors.push("v5 selection.runner_up이 올바르지 않습니다.");
    }
    if (![
      "best_gate_passing_log_loss",
      "simplicity_tiebreak_within_tolerance",
      "reference_fallback_no_challenger_passed",
    ].includes(selection.selection_reason)) {
      errors.push("v5 selection.selection_reason이 올바르지 않습니다.");
    }
    if (selection.simplicity_tolerance !== 0.01) errors.push("v5 simplicity_tolerance은 0.01이어야 합니다.");
    if (JSON.stringify(selection.tie_break_order) !== JSON.stringify(["complexity_rank", "calibration_error", "log_loss", "model"])) {
      errors.push("v5 selection.tie_break_order가 올바르지 않습니다.");
    }
    if (
      typeof selection.operating_champion !== "string"
      || !selection.operating_champion
      || !candidates.includes(selection.operating_champion)
    ) {
      errors.push("v5 selection.operating_champion이 없습니다.");
    }
  }

  function validateV5Lifecycle(payload, errors) {
    const { meta, model } = payload;
    const lifecycle = model.lifecycle;
    if (!hasExactKeys(lifecycle, ["selection", "deployment", "publication"])) {
      errors.push("v5 model.lifecycle 필드가 계약과 일치하지 않습니다.");
      return;
    }
    const selection = lifecycle.selection;
    const deployment = lifecycle.deployment;
    const publication = lifecycle.publication;
    if (
      !hasExactKeys(selection, ["status"])
      || !hasExactKeys(deployment, ["status"])
      || !hasExactKeys(publication, ["status"])
    ) {
      errors.push("v5 lifecycle 하위 필드가 계약과 일치하지 않습니다.");
      return;
    }
    if (selection.status !== "selected_by_gate" || model.selection_status !== selection.status) {
      errors.push("v5 lifecycle selection status가 올바르지 않습니다.");
    }
    if (!["candidate", "reviewed", "operating"].includes(deployment.status)) {
      errors.push("v5 lifecycle deployment status가 올바르지 않습니다.");
    }
    if (!["unpublished", V5_PUBLICATION_STATUS].includes(publication.status) || meta.publication_status !== publication.status) {
      errors.push("v5 lifecycle publication status가 올바르지 않습니다.");
    }
    const allowed = (
      publication.status === "unpublished" && ["candidate", "reviewed"].includes(deployment.status)
    ) || (
      publication.status === V5_PUBLICATION_STATUS && deployment.status === "operating"
    );
    if (!allowed) errors.push("v5 lifecycle 조합이 올바르지 않습니다.");
  }

  function validateV5Envelope(payload, errors) {
    validateV5StateDefinitions(payload.states, errors);
    validateV5LabelEnvelope(payload.label, errors);
    validateV5ForecastEnvelope(payload.forecast, payload.meta.mode, errors);
    validateV5SelectionEnvelope(payload.selection, payload.model, errors);
    validateV5Lifecycle(payload, errors);
  }

  function validateV5ModelContract(payload, errors) {
    const { meta, model } = payload;
    validateV5Envelope(payload, errors);
    if (typeof meta.generation_id !== "string" || !meta.generation_id) errors.push("v5 meta.generation_id가 없습니다.");
    if (!["demo", "live"].includes(meta.mode)) errors.push("v5 meta.mode는 demo 또는 live여야 합니다.");
    if (!Array.isArray(meta.warnings) || meta.warnings.some((warning) => typeof warning !== "string" || !warning.trim())) {
      errors.push("v5 meta.warnings는 비어 있지 않은 문자열만 포함하는 배열이어야 합니다.");
    }
    if (!isZonedIsoTimestamp(meta.generated_at) || !isZonedIsoTimestamp(meta.data_as_of)) {
      errors.push("v5 meta.generated_at과 data_as_of는 timezone을 포함한 ISO-8601이어야 합니다.");
    }
    const publicationStatus = meta.publication_status;
    if (publicationStatus === V5_PUBLICATION_STATUS) {
      const review = meta.publication_review;
      if (meta.mode !== "live") errors.push("v5 reviewed publication은 live 결과여야 합니다.");
      if (!isLowerSha256(meta.generation_manifest_sha256)) {
        errors.push("v5 reviewed publication에는 generation manifest hash가 필요합니다.");
      }
      const reviewFields = [
        "schema_version", "decision", "reviewed_at", "reviewed_candidate_sha256",
        "champion", "multiscale_promoted", "fx_promoted",
      ];
      if (!hasExactKeys(review, reviewFields)) {
        errors.push("v5 publication_review 필드가 계약과 일치하지 않습니다.");
      } else {
        if (review.schema_version !== V5_PUBLICATION_REVIEW_SCHEMA) errors.push("v5 publication_review schema가 올바르지 않습니다.");
        if (review.decision !== "publish_v5_research_snapshot") errors.push("v5 publication_review decision이 올바르지 않습니다.");
        if (!isZonedIsoTimestamp(review.reviewed_at)) errors.push("v5 publication_review reviewed_at이 올바르지 않습니다.");
        if (!isLowerSha256(review.reviewed_candidate_sha256)) errors.push("v5 publication_review candidate hash가 올바르지 않습니다.");
        if (review.champion !== model.champion) errors.push("v5 publication_review champion이 일치하지 않습니다.");
        if (
          typeof review.multiscale_promoted !== "boolean"
          || review.multiscale_promoted
            !== (model.champion === "causal_multiscale_ensemble")
        ) {
          errors.push("v5 publication_review 멀티스케일 승격 상태가 공식 모델과 일치하지 않습니다.");
        }
        if (review.fx_promoted !== false) {
          errors.push("v5 publication_review는 FX 비승격을 보존해야 합니다.");
        }
      }
    } else if (publicationStatus === "unpublished") {
      if (meta.publication_review !== undefined) {
        errors.push("v5 unpublished 결과에는 publication_review가 없어야 합니다.");
      }
    } else {
      errors.push("v5 publication_status가 올바르지 않습니다.");
    }
    if (meta.generation_manifest_sha256 !== undefined && !isLowerSha256(meta.generation_manifest_sha256)) {
      errors.push("v5 generation_manifest_sha256이 올바르지 않습니다.");
    }
    if (!isObject(meta.freshness)) {
      errors.push("v5 meta.freshness 객체가 없습니다.");
    } else {
      if (!hasExactKeys(meta.freshness, ["cadence", "maximum_age_days", "age_days", "status", "data_as_of"])) {
        errors.push("v5 freshness 필드가 계약과 일치하지 않습니다.");
      }
      if (meta.freshness.cadence !== "weekly") errors.push("v5 freshness.cadence는 weekly여야 합니다.");
      validateInteger(meta.freshness.maximum_age_days, "v5 freshness.maximum_age_days", errors, 1);
      validateInteger(meta.freshness.age_days, "v5 freshness.age_days", errors);
      if (meta.freshness.maximum_age_days !== 10) errors.push("v5 freshness.maximum_age_days는 10이어야 합니다.");
      const expectedAge = Math.max(0, Math.floor((Date.parse(meta.generated_at) - Date.parse(meta.data_as_of)) / 86400000));
      if (meta.freshness.age_days !== expectedAge) errors.push("v5 freshness.age_days가 기준 시점과 일치하지 않습니다.");
      const expectedStatus = expectedAge <= meta.freshness.maximum_age_days ? "current" : "stale";
      if (meta.freshness.status !== expectedStatus) errors.push("v5 freshness.status가 age_days와 일치하지 않습니다.");
      if (!isZonedIsoTimestamp(meta.freshness.data_as_of) || Date.parse(meta.freshness.data_as_of) !== Date.parse(meta.data_as_of)) {
        errors.push("v5 freshness.data_as_of가 meta.data_as_of와 일치하지 않습니다.");
      }
    }

    if (model.version !== V5_MODEL_VERSION) errors.push(`v5 model.version은 ${V5_MODEL_VERSION}이어야 합니다.`);
    if (model.label_version !== V5_LABEL_VERSION) errors.push(`v5 model.label_version은 ${V5_LABEL_VERSION}이어야 합니다.`);
    if (model.feature_set_version !== V5_FEATURE_SET_VERSION) errors.push(`v5 model.feature_set_version은 ${V5_FEATURE_SET_VERSION}이어야 합니다.`);
    if (!Array.isArray(model.leaderboard) || !model.leaderboard.length) errors.push("v5 model.leaderboard는 비어 있지 않아야 합니다.");
    if (!championSelectionEvidence(model).valid) {
      errors.push("v5 공식 모델은 leaderboard와 selection diagnostics의 단일 gate 통과 모델이어야 합니다.");
    }
    const forecastComparison = model.forecast_comparison;
    if (!isObject(forecastComparison)) {
      errors.push("v5 model.forecast_comparison 객체가 없습니다.");
    } else {
      const comparisonFields = ["role", "horizon_weeks", "models"];
      if (!hasExactKeys(forecastComparison, comparisonFields)) {
        errors.push("v5 model.forecast_comparison 필드가 계약과 일치하지 않습니다.");
      } else {
        if (forecastComparison.role !== "research_comparison" || forecastComparison.horizon_weeks !== 1) {
          errors.push("v5 model.forecast_comparison 역할·horizon이 올바르지 않습니다.");
        }
        const comparisonModels = forecastComparison.models;
        if (
          !Array.isArray(comparisonModels)
          || !comparisonModels.length
          || comparisonModels.some((name) => typeof name !== "string" || !name)
          || new Set(comparisonModels).size !== comparisonModels.length
          || !comparisonModels.includes(model.champion)
          || !comparisonModels.includes(payload.selection?.operating_champion)
        ) {
          errors.push("v5 model.forecast_comparison 모델 목록이 올바르지 않습니다.");
        }
        const leaderboardNames = new Set(
          Array.isArray(model.leaderboard)
            ? model.leaderboard.map((row) => isObject(row) ? row.name : null).filter(Boolean)
            : [],
        );
        if (
          Array.isArray(comparisonModels)
          && comparisonModels.some((name) => !leaderboardNames.has(name))
        ) {
          errors.push("v5 model.forecast_comparison 모델이 leaderboard에 없습니다.");
        }
      }
    }

    const baseline = model.baseline_v4;
    const baselineFields = [
      "result_version", "label_version", "model_version", "feature_set_version",
      "champion", "payload_sha256", "artifacts_inventory_sha256", "captured_at",
    ];
    if (!isObject(baseline) || baselineFields.some((field) => !(field in baseline))) {
      errors.push("v5 model.baseline_v4 필드가 계약과 일치하지 않습니다.");
    } else {
      if (baseline.result_version !== V4_RESULT_VERSION) errors.push("v5 baseline_v4.result_version이 올바르지 않습니다.");
      if (baseline.label_version !== V4_LABEL_VERSION) errors.push("v5 baseline_v4.label_version이 올바르지 않습니다.");
      if (baseline.model_version !== V4_MODEL_VERSION) errors.push("v5 baseline_v4.model_version이 올바르지 않습니다.");
      if (baseline.feature_set_version !== V4_FEATURE_SET_VERSION) errors.push("v5 baseline_v4.feature_set_version이 올바르지 않습니다.");
      if (!isLowerSha256(baseline.payload_sha256) || !isLowerSha256(baseline.artifacts_inventory_sha256)) {
        errors.push("v5 baseline_v4 hash는 소문자 SHA-256이어야 합니다.");
      }
      if (!isIsoDate(baseline.captured_at)) errors.push("v5 baseline_v4.captured_at은 ISO date여야 합니다.");
    }

    const preregistration = model.structural_preregistration;
    if (
      !isObject(preregistration)
      || preregistration.path !== "config/structural_v5.json"
      || !isLowerSha256(preregistration.sha256)
    ) {
      errors.push("v5 structural_preregistration 계약이 올바르지 않습니다.");
    }

    const execution = model.execution_parameters;
    const executionFields = [
      "profile", "directional_minimum_selection_predictions",
      "directional_minimum_diagnostic_predictions", "directional_maximum_selection_origins",
      "directional_maximum_diagnostic_origins", "duration_bootstrap_resamples",
      "conditional_outcome_bootstrap_resamples", "preregistered_bootstrap_resamples",
      "preregistration_overrides", "sha256",
    ];
    if (!hasExactKeys(execution, executionFields)) {
      errors.push("v5 execution_parameters 필드가 계약과 일치하지 않습니다.");
    } else {
      if (!["quick", "standard", "full"].includes(execution.profile)) errors.push("v5 execution profile이 올바르지 않습니다.");
      const minimum = execution.profile === "quick" ? 3 : 12;
      const maximum = execution.profile === "quick" ? 3 : execution.profile === "standard" ? 60 : null;
      if (execution.directional_minimum_selection_predictions !== minimum || execution.directional_minimum_diagnostic_predictions !== minimum) {
        errors.push("v5 directional 실행 최소 표본이 profile과 일치하지 않습니다.");
      }
      if (execution.directional_maximum_selection_origins !== maximum || execution.directional_maximum_diagnostic_origins !== maximum) {
        errors.push("v5 directional 실행 최대 표본이 profile과 일치하지 않습니다.");
      }
      validateInteger(execution.duration_bootstrap_resamples, "v5 duration bootstrap", errors, 1);
      validateInteger(execution.conditional_outcome_bootstrap_resamples, "v5 outcome bootstrap", errors, 1);
      if (execution.preregistered_bootstrap_resamples !== 1999 || !Array.isArray(execution.preregistration_overrides) || !isLowerSha256(execution.sha256)) {
        errors.push("v5 execution_parameters 사전등록 연결이 올바르지 않습니다.");
      }
    }
    const executionProfile = isObject(execution) ? execution.profile : null;
    if (model.profile !== executionProfile || !["quick", "standard", "full"].includes(model.profile)) {
      errors.push("v5 model.profile은 execution_parameters.profile과 일치해야 합니다.");
    }
    if (meta.mode === "live" && executionProfile === "quick") {
      errors.push("v5 live 결과는 quick profile을 사용할 수 없습니다.");
    }

    const expectedSourceLicenses = V5_SOURCE_LICENSES[meta.mode];
    if (expectedSourceLicenses) {
      const sourcesById = new Map();
      let invalidSource = false;
      for (const source of payload.sources) {
        if (!isObject(source) || typeof source.id !== "string" || sourcesById.has(source.id)) {
          invalidSource = true;
          continue;
        }
        sourcesById.set(source.id, source);
      }
      const expectedSourceIds = Object.keys(expectedSourceLicenses);
      if (
        invalidSource
        || sourcesById.size !== expectedSourceIds.length
        || expectedSourceIds.some((sourceId) => !sourcesById.has(sourceId))
      ) {
        errors.push(`v5 sources identity가 meta.mode=${meta.mode}와 일치하지 않습니다.`);
      } else {
        for (const [sourceId, expectedLicense] of Object.entries(expectedSourceLicenses)) {
          const source = sourcesById.get(sourceId);
          if (source.license_class !== expectedLicense) {
            errors.push(`v5 sources.${sourceId}.license_class가 mode 계약과 일치하지 않습니다.`);
          }
          if (!V5_SOURCE_STATUSES.includes(source.status)) {
            errors.push(`v5 sources.${sourceId}.status가 지원하는 상태가 아닙니다.`);
          }
        }
        if (meta.mode === "live") {
          const h10 = sourcesById.get("frb_h10");
          const fxProvenanceFields = [
            "official_release_archive_ingest", "availability_basis",
            "archive_revision_policy", "archive_correction_availability_basis",
          ];
          for (const field of fxProvenanceFields) {
            if (!(field in h10) || h10[field] !== model.fx_ablation?.[field]) {
              errors.push(`v5 sources.frb_h10.${field}가 model FX provenance와 일치하지 않습니다.`);
            }
          }
          const releaseCount = h10.archive_release_count;
          const correctionCount = h10.archive_correction_count;
          const correctionValues = h10.archive_correction_available_at;
          const correctionsValid = Array.isArray(correctionValues)
            && correctionValues.every((value, index) => (
              isZonedIsoTimestamp(value)
              && /(?:Z|\+00:00)$/.test(value)
              && (index === 0 || Date.parse(correctionValues[index - 1]) < Date.parse(value))
            ));
          if (
            !Number.isInteger(releaseCount) || releaseCount < 0
            || !Number.isInteger(correctionCount) || correctionCount < 0
            || releaseCount < correctionCount
            || !correctionsValid
            || correctionCount !== (Array.isArray(correctionValues) ? correctionValues.length : -1)
          ) {
            errors.push("v5 sources.frb_h10 archive release·correction inventory가 올바르지 않습니다.");
          }
          if (
            h10.archive_correction_quarantine_weeks !== 27
            || h10.archive_evaluation_start !== "2022-01-01"
            || h10.archive_evaluation_start_rationale !== "post_2019_06_24_jan06_index_rebase_common_scale"
          ) {
            errors.push("v5 sources.frb_h10 archive 평가·정정 격리 계약이 올바르지 않습니다.");
          }
          const archiveIngest = model.fx_ablation?.official_release_archive_ingest;
          if (
            (archiveIngest === true && releaseCount < 1)
            || (archiveIngest === false && (
              releaseCount !== 0 || correctionCount !== 0
              || (Array.isArray(correctionValues) && correctionValues.length !== 0)
            ))
          ) {
            errors.push("v5 sources.frb_h10 archive inventory가 ingest mode와 일치하지 않습니다.");
          }
        }
      }
    }

    const directional = model.directional_transition;
    if (!isObject(directional)) {
      errors.push("v5 model.directional_transition 객체가 없습니다.");
    } else {
      if (
        directional.target !== "first_departure_state_within_h_or_no_departure"
        || directional.deployed_direction_role !== "first_destination_given_departure"
        || directional.selection_metric !== "conditional_destination_log_loss"
        || directional.minimum_selection_departure_events !== 8
        || directional.minimum_selection_destination_classes !== 2
        || directional.minimum_selection_event_blocks !== 3
      ) {
        errors.push("v5 directional_transition 배포 역할·선정 gate가 올바르지 않습니다.");
      }
      const champions = directional.champions;
      if (
        !hasExactKeys(champions, ["1w", "4w", "13w"])
        || Object.values(champions).some((value) => typeof value !== "string" || !value)
      ) {
        errors.push("v5 directional_transition.champions 계약이 올바르지 않습니다.");
      }
      if (!Array.isArray(directional.leaderboard) || !directional.leaderboard.length) {
        errors.push("v5 directional_transition.leaderboard는 비어 있지 않아야 합니다.");
      }
      if (!Array.isArray(directional.selection_diagnostics) || !directional.selection_diagnostics.length) {
        errors.push("v5 directional_transition.selection_diagnostics는 비어 있지 않아야 합니다.");
      }
      if (!isIsoDate(directional.selection_end)) errors.push("v5 directional_transition.selection_end가 실제 날짜가 아닙니다.");
    }

    const health = model.model_health;
    if (!isObject(health) || !["ok", "review_due"].includes(health.status) || !Array.isArray(health.reasons)) {
      errors.push("v5 model.model_health 계약이 올바르지 않습니다.");
    }

    if (model.champion_core_feature_set_version !== V4_FEATURE_SET_VERSION) {
      errors.push("v5 champion은 v4 core feature 계약을 유지해야 합니다.");
    }
    if (model.fx_role !== "context_and_preregistered_shadow_ablation") {
      errors.push("v5 model.fx_role이 올바르지 않습니다.");
    }
    const fxAblation = model.fx_ablation;
    const fxAblationFields = [
      "role", "variants", "minimum_common_weeks",
      "historical_availability_backfill", "official_release_archive_ingest",
      "availability_basis", "archive_revision_policy",
      "archive_correction_availability_basis", "status", "eligible_common_weeks",
      "first_eligible_cutoff", "last_eligible_cutoff", "manifest",
      "status_reason", "common_origin_required_pairs", "minimum_train_weeks",
      "target_horizon_weeks", "purge_weeks", "target_availability_rule",
      "model", "common_evaluation_origins", "variant_metrics", "gate",
      "promotion_allowed", "promotion_candidate", "core_champion_promoted",
    ];
    if (!hasExactKeys(fxAblation, fxAblationFields)) {
      errors.push("v5 model.fx_ablation 필드가 계약과 일치하지 않습니다.");
    } else {
      const variantsMatch = Array.isArray(fxAblation.variants)
        && fxAblation.variants.length === FX_ABLATION_VARIANTS.length
        && fxAblation.variants.every((value, index) => value === FX_ABLATION_VARIANTS[index]);
      if (!variantsMatch) errors.push("v5 FX ablation variant 순서가 올바르지 않습니다.");
      if (
        fxAblation.role !== "prospective_shadow"
        || fxAblation.minimum_common_weeks !== 156
        || fxAblation.historical_availability_backfill !== false
      ) {
        errors.push("v5 FX ablation은 비소급 prospective shadow여야 합니다.");
      }
      if (
        typeof fxAblation.official_release_archive_ingest !== "boolean"
        || !["official_archive_release_schedule", "collection_first_seen_at"].includes(fxAblation.availability_basis)
        || fxAblation.official_release_archive_ingest !== (fxAblation.availability_basis === "official_archive_release_schedule")
        || fxAblation.archive_revision_policy !== "later_official_release_preserved_as_new_vintage"
        || fxAblation.archive_correction_availability_basis !== "date_only_conservative_next_day"
      ) {
        errors.push("v5 FX ablation archive availability·revision provenance가 올바르지 않습니다.");
      }
      const fxStatuses = ["unavailable", "insufficient_history", "evaluated"];
      if (!fxStatuses.includes(fxAblation.status)) errors.push("v5 FX ablation status가 올바르지 않습니다.");
      const statusReasons = {
        unavailable: [
          "fx_feature_result_unavailable", "fx_feature_contract_unavailable",
          "fixed_nine_pair_contract_unavailable", "fx_model_features_non_numeric",
        ],
        insufficient_history: [
          "eligible_common_weeks_below_156",
          "no_origin_has_104_strictly_available_training_targets",
        ],
        evaluated: [null],
      };
      if (!statusReasons[fxAblation.status]?.includes(fxAblation.status_reason)) {
        errors.push("v5 FX ablation status_reason이 올바르지 않습니다.");
      }
      validateInteger(fxAblation.eligible_common_weeks, "v5 FX eligible_common_weeks", errors);
      const eligible = fxAblation.eligible_common_weeks;
      const hasBounds = isIsoDate(fxAblation.first_eligible_cutoff)
        && isIsoDate(fxAblation.last_eligible_cutoff)
        && fxAblation.first_eligible_cutoff <= fxAblation.last_eligible_cutoff;
      if ((eligible === 0 && (fxAblation.first_eligible_cutoff !== null || fxAblation.last_eligible_cutoff !== null))
        || (eligible > 0 && !hasBounds)) {
        errors.push("v5 FX ablation 공통 시점 범위가 올바르지 않습니다.");
      }
      const manifest = fxAblation.manifest;
      if (!Array.isArray(manifest) || (manifest.length !== 0 && manifest.length !== FX_ABLATION_VARIANTS.length)) {
        errors.push("v5 FX ablation manifest가 불완전합니다.");
      } else if (manifest.length) {
        manifest.forEach((row, index) => {
          const expectedVariant = FX_ABLATION_VARIANTS[index];
          const validCount = Number.isInteger(row && row.feature_count)
            && row.feature_count >= 0
            && ((expectedVariant === "v4_control") === (row.feature_count === 0));
          if (
            !hasExactKeys(row, ["variant", "feature_count", "feature_columns_sha256"])
            || row.variant !== expectedVariant
            || !validCount
            || !isLowerSha256(row.feature_columns_sha256)
          ) {
            errors.push(`v5 FX ablation manifest[${index}]가 올바르지 않습니다.`);
          }
        });
      }
      if ((eligible > 0 && manifest.length === 0)
        || (fxAblation.status === "evaluated" && eligible < 156)
        || (fxAblation.status_reason === "eligible_common_weeks_below_156" && eligible >= 156)
        || (fxAblation.status === "unavailable"
          && fxAblation.status_reason !== "fx_model_features_non_numeric"
          && (eligible !== 0 || manifest.length !== 0))) {
        errors.push("v5 FX ablation readiness가 공통 표본 수와 일치하지 않습니다.");
      }

      if (
        fxAblation.common_origin_required_pairs !== 9
        || fxAblation.minimum_train_weeks !== 104
        || fxAblation.target_horizon_weeks !== 1
        || fxAblation.purge_weeks !== 1
        || fxAblation.target_availability_rule !== "last_train_target_strictly_before_evaluation_origin"
      ) {
        errors.push("v5 FX ablation 공통 origin·purge 계약이 올바르지 않습니다.");
      }
      const fxModel = fxAblation.model;
      const fxModelFields = [
        "name", "horizon_weeks", "multiclass", "regularization", "regularization_c",
        "class_weight", "solver", "max_iter", "tolerance", "random_state",
        "imputation", "scaling", "fit_window", "state_order",
      ];
      const fxModelStatesMatch = Array.isArray(fxModel?.state_order)
        && fxModel.state_order.length === STATE_ORDER.length
        && fxModel.state_order.every((value, index) => value === STATE_ORDER[index]);
      if (
        !hasExactKeys(fxModel, fxModelFields)
        || fxModel.name !== "fixed_l2_multinomial_logistic"
        || fxModel.horizon_weeks !== 1
        || fxModel.multiclass !== "multinomial"
        || fxModel.regularization !== "l2"
        || fxModel.regularization_c !== 0.1
        || fxModel.class_weight !== null
        || fxModel.solver !== "lbfgs"
        || fxModel.max_iter !== 2000
        || fxModel.tolerance !== 1e-6
        || fxModel.random_state !== 17
        || fxModel.imputation !== "expanding_train_median"
        || fxModel.scaling !== "expanding_train_standard"
        || fxModel.fit_window !== "expanding"
        || !fxModelStatesMatch
      ) {
        errors.push("v5 FX ablation 고정 모델 계약이 올바르지 않습니다.");
      }

      const origins = fxAblation.common_evaluation_origins;
      const originFields = ["count", "first_origin", "last_origin", "sha256", "rows"];
      const originRows = Array.isArray(origins?.rows) ? origins.rows : [];
      if (
        !hasExactKeys(origins, originFields)
        || !Number.isInteger(origins.count)
        || origins.count < 0
        || origins.count !== originRows.length
      ) {
        errors.push("v5 FX ablation 공통 평가 origin 계약이 올바르지 않습니다.");
      } else if (origins.count === 0) {
        if (origins.first_origin !== null || origins.last_origin !== null || origins.sha256 !== null) {
          errors.push("v5 FX ablation 빈 평가 origin 요약이 일관되지 않습니다.");
        }
      } else {
        if (
          !isIsoDate(origins.first_origin)
          || !isIsoDate(origins.last_origin)
          || !isLowerSha256(origins.sha256)
          || origins.first_origin !== originRows[0]?.origin_date
          || origins.last_origin !== originRows[originRows.length - 1]?.origin_date
        ) {
          errors.push("v5 FX ablation 평가 origin 요약이 일관되지 않습니다.");
        }
        originRows.forEach((row, index) => {
          const validRow = hasExactKeys(row, [
            "origin_date", "target_date", "train_size", "train_start_origin",
            "last_train_origin", "last_train_target", "purged_origin_count",
          ])
            && isIsoDate(row.origin_date)
            && isIsoDate(row.target_date)
            && isIsoDate(row.train_start_origin)
            && isIsoDate(row.last_train_origin)
            && isIsoDate(row.last_train_target)
            && Number.isInteger(row.train_size)
            && row.train_size >= 104
            && row.purged_origin_count === 1
            && row.last_train_target < row.origin_date;
          if (!validRow) errors.push(`v5 FX ablation 평가 origin[${index}] purge 계약이 올바르지 않습니다.`);
        });
      }

      const variantMetrics = fxAblation.variant_metrics;
      if (!Array.isArray(variantMetrics)) {
        errors.push("v5 FX ablation variant_metrics가 배열이 아닙니다.");
      } else if (fxAblation.status === "evaluated") {
        if (variantMetrics.length !== FX_ABLATION_VARIANTS.length || origins.count < 1) {
          errors.push("v5 FX ablation 평가 metric이 불완전합니다.");
        }
        variantMetrics.forEach((row, index) => {
          const expectedVariant = FX_ABLATION_VARIANTS[index];
          const fields = [
            "variant", "feature_count", "fx_feature_count", "feature_columns_sha256",
            "log_loss", "brier", "accuracy", "balanced_accuracy", "n", "n_predictions", "fallback", "fallback_count",
            "fallback_reasons", "first_origin", "last_origin", "origin_sha256",
          ];
          const valid = hasExactKeys(row, fields)
            && row.variant === expectedVariant
            && Number.isInteger(row.feature_count) && row.feature_count >= 1
            && Number.isInteger(row.fx_feature_count) && row.fx_feature_count >= 0
            && ((expectedVariant === "v4_control") === (row.fx_feature_count === 0))
            && isLowerSha256(row.feature_columns_sha256)
            && Number.isFinite(row.log_loss) && row.log_loss >= 0
            && Number.isFinite(row.brier) && row.brier >= 0 && row.brier <= 2
            && Number.isFinite(row.accuracy) && row.accuracy >= 0 && row.accuracy <= 1
            && Number.isFinite(row.balanced_accuracy) && row.balanced_accuracy >= 0 && row.balanced_accuracy <= 1
            && row.n === origins.count && row.n_predictions === origins.count
            && typeof row.fallback === "boolean"
            && Number.isInteger(row.fallback_count) && row.fallback_count >= 0
            && row.fallback === (row.fallback_count > 0)
            && isObject(row.fallback_reasons)
            && row.first_origin === origins.first_origin
            && row.last_origin === origins.last_origin
            && row.origin_sha256 === origins.sha256;
          if (!valid) errors.push(`v5 FX ablation variant_metrics[${index}]가 올바르지 않습니다.`);
        });
      } else if (variantMetrics.length !== 0 || origins.count !== 0) {
        errors.push("v5 FX ablation 미평가 출력은 비어 있어야 합니다.");
      }

      const fxGate = fxAblation.gate;
      const fxGateFields = [
        "reference_variant", "method", "bootstrap_block_weeks",
        "bootstrap_effective_block_weeks", "bootstrap_resamples", "bootstrap_seed",
        "alpha", "minimum_log_loss_improvement", "brier_tolerance",
        "comparisons", "passed_variants",
      ];
      const gateBaseValid = hasExactKeys(fxGate, fxGateFields)
        && fxGate.reference_variant === "v4_control"
        && fxGate.method === "paired_circular_moving_block_bootstrap_holm"
        && fxGate.bootstrap_block_weeks === 13
        && fxGate.bootstrap_resamples === 1999
        && fxGate.bootstrap_seed === 17
        && fxGate.alpha === 0.05
        && fxGate.minimum_log_loss_improvement === 0.05
        && fxGate.brier_tolerance === 0.01
        && Array.isArray(fxGate.comparisons)
        && Array.isArray(fxGate.passed_variants);
      if (!gateBaseValid) {
        errors.push("v5 FX ablation gate 계약이 올바르지 않습니다.");
      } else if (fxAblation.status === "evaluated") {
        const expectedBlock = Math.min(13, Math.max(1, Math.floor(origins.count / 2)));
        if (
          fxGate.bootstrap_effective_block_weeks !== expectedBlock
          || fxGate.comparisons.length !== FX_ABLATION_VARIANTS.length - 1
          || fxGate.comparisons.some((row, index) => (
            !hasExactKeys(row, [
              "variant", "reference_variant", "mean_log_loss_improvement", "brier_difference",
              "control_fallback_count", "fallback_count", "raw_p_value",
              "holm_adjusted_p_value", "gate_passed", "gate_reasons",
            ])
            || row.variant !== FX_ABLATION_VARIANTS[index + 1]
            || row.reference_variant !== "v4_control"
            || !Number.isFinite(row.mean_log_loss_improvement)
            || !Number.isFinite(row.brier_difference)
            || !Number.isFinite(row.raw_p_value) || row.raw_p_value < 0 || row.raw_p_value > 1
            || !Number.isFinite(row.holm_adjusted_p_value) || row.holm_adjusted_p_value < 0 || row.holm_adjusted_p_value > 1
            || typeof row.gate_passed !== "boolean"
            || !Array.isArray(row.gate_reasons) || row.gate_reasons.length === 0
          ))) {
          errors.push("v5 FX ablation paired gate 결과가 올바르지 않습니다.");
        }
      } else if (
        fxGate.bootstrap_effective_block_weeks !== null
        || fxGate.comparisons.length !== 0
        || fxGate.passed_variants.length !== 0
      ) {
        errors.push("v5 FX ablation 미평가 gate는 비어 있어야 합니다.");
      }
      if (
        fxAblation.promotion_allowed !== false
        || fxAblation.promotion_candidate !== null
        || fxAblation.core_champion_promoted !== false
      ) {
        errors.push("v5 FX ablation은 core champion을 자동 승격할 수 없습니다.");
      }
    }

    const coreArtifacts = model.core_artifacts;
    const coreArtifactKeys = Object.keys(V5_CORE_ARTIFACT_PATHS);
    if (!hasExactKeys(coreArtifacts, coreArtifactKeys)) {
      errors.push("v5 model.core_artifacts manifest가 불완전합니다.");
    } else {
      for (const [key, expectedPath] of Object.entries(V5_CORE_ARTIFACT_PATHS)) {
        const artifact = coreArtifacts[key];
        if (
          !hasExactKeys(artifact, ["path", "row_count", "sha256"])
          || artifact.path !== expectedPath
          || !Number.isInteger(artifact.row_count)
          || artifact.row_count < 1
          || !isLowerSha256(artifact.sha256)
        ) {
          errors.push(`v5 core artifact ${key} 계약이 올바르지 않습니다.`);
        }
      }
    }

    const structuralModels = model.structural_models;
    const structuralModelKeys = [
      "xgb_hazard_destination", "causal_dynamic_ensemble",
      "joint_survival_hazard", "causal_multiscale_ensemble",
    ];
    if (!hasExactKeys(structuralModels, structuralModelKeys)) {
      errors.push("v5 model.structural_models 키가 정확한 계약과 일치하지 않습니다.");
    } else {
      const expectedExperts = ["markov", "xgboost", "xgb_hazard_destination"];
      const hazard = structuralModels.xgb_hazard_destination;
      if (
        !hasExactKeys(hazard, ["hazard_model", "destination_model", "direct_jump_floor"])
        || hazard.hazard_model !== "binary_xgboost"
        || hazard.destination_model !== "xgboost"
        || hazard.direct_jump_floor !== 0.000001
      ) {
        errors.push("v5 xgb_hazard_destination 계약이 올바르지 않습니다.");
      }
      const ensemble = structuralModels.causal_dynamic_ensemble;
      if (
        !hasExactKeys(ensemble, ["experts", "half_life_weeks", "minimum_history_rows", "eligible_loss_rule"])
        || !Array.isArray(ensemble.experts)
        || ensemble.experts.length !== expectedExperts.length
        || ensemble.experts.some((expert, index) => expert !== expectedExperts[index])
        || ensemble.half_life_weeks !== 52
        || ensemble.minimum_history_rows !== 26
        || ensemble.eligible_loss_rule !== "target_date_strictly_before_origin"
      ) {
        errors.push("v5 causal_dynamic_ensemble 계약이 올바르지 않습니다.");
      }
      const survival = structuralModels.joint_survival_hazard;
      if (
        !hasExactKeys(survival, ["base_target_weeks", "horizons_weeks", "future_covariates", "identity"])
        || survival.base_target_weeks !== 1
        || !Array.isArray(survival.horizons_weeks)
        || survival.horizons_weeks.length !== TRANSITION_HORIZONS.length
        || survival.horizons_weeks.some((value, index) => value !== TRANSITION_HORIZONS[index])
        || survival.future_covariates !== "origin_values_frozen"
        || survival.identity !== "one_minus_product_one_minus_weekly_hazard"
      ) {
        errors.push("v5 joint_survival_hazard 계약이 올바르지 않습니다.");
      }
      const multiscale = structuralModels.causal_multiscale_ensemble;
      const multiscaleFields = [
        "role", "experts", "scale_half_lives_weeks", "outer_scale_weights",
        "aggregation", "inner_pool_method", "minimum_history_rows",
        "eligible_loss_rule", "selection_gate", "automatic_promotion_bypass",
        "sidecar",
      ];
      const scales = [26, 52, 104];
      const sidecarMatches = isObject(coreArtifacts)
        && isObject(multiscale?.sidecar)
        && JSON.stringify(multiscale.sidecar) === JSON.stringify(coreArtifacts.multiscale_ensemble_scales);
      if (
        !hasExactKeys(multiscale, multiscaleFields)
        || multiscale.role !== "v5_opt_in_candidate"
        || !Array.isArray(multiscale.experts)
        || multiscale.experts.length !== expectedExperts.length
        || multiscale.experts.some((expert, index) => expert !== expectedExperts[index])
        || !Array.isArray(multiscale.scale_half_lives_weeks)
        || multiscale.scale_half_lives_weeks.length !== scales.length
        || multiscale.scale_half_lives_weeks.some((value, index) => value !== scales[index])
        || !Array.isArray(multiscale.outer_scale_weights)
        || multiscale.outer_scale_weights.length !== scales.length
        || multiscale.outer_scale_weights.some((value) => Math.abs(value - (1 / 3)) > 1e-15)
        || multiscale.aggregation !== "fixed_equal_probability_average"
        || multiscale.inner_pool_method !== "causal_discounted_completed_oos_log_score"
        || multiscale.minimum_history_rows !== 26
        || multiscale.eligible_loss_rule !== "target_date_strictly_before_origin"
        || multiscale.selection_gate !== "existing_multiclass_holm_log_loss_brier_zero_fallback"
        || multiscale.automatic_promotion_bypass !== false
        || !sidecarMatches
      ) {
        errors.push("v5 causal_multiscale_ensemble과 sidecar 계약이 올바르지 않습니다.");
      }
    }

    const candidateManifest = model.candidate_manifest;
    const candidateNames = Array.isArray(candidateManifest?.models)
      ? candidateManifest.models.map((row) => row?.name)
      : [];
    const expectedCandidateNames = isObject(payload.selection) && Array.isArray(payload.selection.candidate_set)
      ? payload.selection.candidate_set
      : [];
    if (
      !isObject(candidateManifest)
      || candidateManifest.profile !== model.profile
      || candidateManifest.random_state !== 17
      || !isLowerSha256(model.candidate_manifest_sha256)
      || candidateNames.length !== expectedCandidateNames.length
      || new Set(candidateNames).size !== candidateNames.length
      || expectedCandidateNames.some((name) => !candidateNames.includes(name))
    ) {
      errors.push("v5 candidate_manifest model set·profile 계약이 올바르지 않습니다.");
    }

    const researchArtifacts = model.research_artifacts;
    const researchKeys = isObject(researchArtifacts) ? Object.keys(researchArtifacts) : [];
    const hasRequiredResearch = V5_REQUIRED_RESEARCH_ARTIFACTS.every((key) => researchKeys.includes(key));
    const fxResearchCount = V5_FX_RESEARCH_ARTIFACTS.filter((key) => researchKeys.includes(key)).length;
    const modelConditionedResearchCount = V5_MODEL_CONDITIONED_RESEARCH_ARTIFACTS
      .filter((key) => researchKeys.includes(key)).length;
    const allowedResearchKeys = new Set([
      ...V5_REQUIRED_RESEARCH_ARTIFACTS,
      ...V5_FX_RESEARCH_ARTIFACTS,
      ...V5_MODEL_CONDITIONED_RESEARCH_ARTIFACTS,
    ]);
    if (
      !isObject(researchArtifacts)
      || !hasRequiredResearch
      || researchKeys.some((key) => !allowedResearchKeys.has(key))
      || (fxResearchCount !== 0 && fxResearchCount !== V5_FX_RESEARCH_ARTIFACTS.length)
      || modelConditionedResearchCount !== V5_MODEL_CONDITIONED_RESEARCH_ARTIFACTS.length
      || (fxAblation?.status === "evaluated" && fxResearchCount !== V5_FX_RESEARCH_ARTIFACTS.length)
    ) {
      errors.push("v5 model.research_artifacts manifest가 불완전합니다.");
    } else {
      for (const key of researchKeys) {
        const artifact = researchArtifacts[key];
        if (
          !hasExactKeys(artifact, ["path", "row_count", "sha256"])
          || artifact.path !== V5_RESEARCH_ARTIFACT_PATHS[key]
          || !Number.isInteger(artifact.row_count)
          || artifact.row_count < (key === "fx_ablation_oos" ? 0 : 1)
          || !isLowerSha256(artifact.sha256)
        ) {
          errors.push(`v5 research artifact ${key} 계약이 올바르지 않습니다.`);
        }
      }
      const expectedConditionalRows = OUTCOME_ASSETS.length * STATE_ORDER.length * TRANSITION_HORIZONS.length;
      const conditionalArtifact = researchArtifacts.conditional_asset_statistics;
      if (!isObject(conditionalArtifact) || conditionalArtifact.row_count !== expectedConditionalRows) {
        errors.push("v5 conditional asset statistics artifact 행 수가 올바르지 않습니다.");
      }
      const comparisonModelNames = isObject(forecastComparison) && Array.isArray(forecastComparison.models)
        ? forecastComparison.models
        : [];
      const modelConditionedArtifact = researchArtifacts.model_conditioned_asset_statistics;
      const expectedModelConditionedRows = comparisonModelNames.length * expectedConditionalRows;
      if (
        modelConditionedResearchCount === V5_MODEL_CONDITIONED_RESEARCH_ARTIFACTS.length
        && (
          !isObject(modelConditionedArtifact)
          || modelConditionedArtifact.row_count !== expectedModelConditionedRows
        )
      ) {
        errors.push("v5 model-conditioned asset statistics artifact 행 수가 올바르지 않습니다.");
      }
      const leaderboardArtifact = researchArtifacts.directional_model_leaderboard;
      if (
        isObject(leaderboardArtifact)
        && isObject(directional)
        && Array.isArray(directional.leaderboard)
        && leaderboardArtifact.row_count !== directional.leaderboard.length
      ) {
        errors.push("v5 directional leaderboard artifact 행 수가 payload와 일치하지 않습니다.");
      }
      const diagnosticsArtifact = researchArtifacts.directional_selection_diagnostics;
      if (
        isObject(diagnosticsArtifact)
        && isObject(directional)
        && Array.isArray(directional.selection_diagnostics)
        && diagnosticsArtifact.row_count !== directional.selection_diagnostics.length
      ) {
        errors.push("v5 directional diagnostics artifact 행 수가 payload와 일치하지 않습니다.");
      }
      const fxFeaturesArtifact = researchArtifacts.fx_features;
      const fxCoverageArtifact = researchArtifacts.fx_coverage;
      const fxAblationOosArtifact = researchArtifacts.fx_ablation_oos;
      if (
        fxResearchCount === V5_FX_RESEARCH_ARTIFACTS.length
        && isObject(fxFeaturesArtifact)
        && isObject(fxCoverageArtifact)
        && (
          fxFeaturesArtifact.row_count !== fxCoverageArtifact.row_count
          || !isObject(fxAblationOosArtifact)
          || fxAblationOosArtifact.row_count !== (
            fxAblation?.status === "evaluated"
              ? fxAblation.common_evaluation_origins.count * FX_ABLATION_VARIANTS.length
              : 0
          )
        )
      ) {
        errors.push("v5 FX research artifact 행 수가 일치하지 않습니다.");
      }
    }

    const evidence = model.evidence_artifacts;
    if (!hasExactKeys(evidence, ["state_membership_history", "weekly_state_forecasts"])) {
      errors.push("v5 model.evidence_artifacts 필드가 계약과 일치하지 않습니다.");
    } else {
      const membership = evidence.state_membership_history;
      if (
        !hasExactKeys(membership, [
          "path", "row_count", "sha256", "label_fit_weeks", "label_fit_end",
          "initial_state", "method",
        ])
        || membership.path !== "state-membership-history.csv"
        || !Number.isInteger(membership.row_count)
        || membership.row_count < 520
        || !isLowerSha256(membership.sha256)
        || membership.label_fit_weeks !== 520
        || !isIsoTimestamp(membership.label_fit_end)
        || !STATE_ORDER.includes(membership.initial_state)
        || membership.method !== "risk_score_anchor_membership"
      ) {
        errors.push("v5 state_membership_history artifact 계약이 올바르지 않습니다.");
      }
      const forecasts = evidence.weekly_state_forecasts;
      if (
        !hasExactKeys(forecasts, ["path", "row_count", "sha256"])
        || forecasts.path !== "weekly-state-forecasts-v5.csv"
        || forecasts.row_count !== payload.weekly.length
        || !isLowerSha256(forecasts.sha256)
      ) {
        errors.push("v5 weekly_state_forecasts artifact 계약이 올바르지 않습니다.");
      }
    }
  }

  function validateV5WeekContract(item, path, errors, model) {
    const current = item.current;
    const currentFields = ["state", "memberships", "primary_membership", "membership_entropy", "method"];
    if (!hasExactKeys(current, currentFields)) {
      errors.push(`${path}.current 필드가 v5 계약과 정확히 일치하지 않습니다.`);
      return;
    }
    const memberships = validateStateVector(current.memberships, `${path}.current.memberships`, errors, "소속도");
    const primary = strictProbability(current.primary_membership);
    if (primary === null) errors.push(`${path}.current.primary_membership가 0–1 범위가 아닙니다.`);
    if (memberships && primary !== null && Math.abs(primary - memberships[current.state]) > 0.00000001) {
      errors.push(`${path}.current.primary_membership는 hard state 소속도와 일치해야 합니다.`);
    }
    if (strictProbability(current.membership_entropy) === null) errors.push(`${path}.current.membership_entropy가 0–1 범위가 아닙니다.`);
    if (current.method !== "risk_score_anchor_membership") errors.push(`${path}.current.method가 올바르지 않습니다.`);

    const forecast = item.next_week;
    const forecastFields = [
      "state", "probabilities", "confidence", "entropy", "date",
      "method", "model", "fallback", "fallback_reason",
    ];
    if (!hasExactKeys(forecast, forecastFields)) {
      errors.push(`${path}.next_week 필드가 v5 계약과 정확히 일치하지 않습니다.`);
      return;
    }
    const forecastProbabilities = validateStateVector(forecast.probabilities, `${path}.next_week.probabilities`, errors, "예측확률");
    const confidence = strictProbability(forecast.confidence);
    if (confidence === null) errors.push(`${path}.next_week.confidence가 0–1 범위가 아닙니다.`);
    if (forecastProbabilities && confidence !== null && Math.abs(confidence - forecastProbabilities[forecast.state]) > 0.00000001) {
      errors.push(`${path}.next_week.confidence는 예측 state 확률과 일치해야 합니다.`);
    }
    if (strictProbability(forecast.entropy) === null) errors.push(`${path}.next_week.entropy가 0–1 범위가 아닙니다.`);
    if (forecast.date !== isoDateOffset(item.date, 7)) errors.push(`${path}.next_week.date가 1주 horizon과 일치하지 않습니다.`);
    for (const field of ["method", "model", "fallback_reason"]) {
      if (typeof forecast[field] !== "string") errors.push(`${path}.next_week.${field}는 문자열이어야 합니다.`);
    }
    if (typeof forecast.fallback !== "boolean") errors.push(`${path}.next_week.fallback은 boolean이어야 합니다.`);

    const forecastComparison = isObject(model) ? model.forecast_comparison : null;
    if (!forecastComparison) {
      errors.push(`${path}.model_forecasts에는 forecast_comparison 메타데이터가 필요합니다.`);
    } else {
      const comparisonModels = Array.isArray(forecastComparison.models)
        ? forecastComparison.models
        : [];
      const modelForecasts = item.model_forecasts;
      if (!Array.isArray(modelForecasts) || modelForecasts.length !== comparisonModels.length) {
        errors.push(`${path}.model_forecasts 모델 수가 올바르지 않습니다.`);
      } else {
        let championForecast = null;
        modelForecasts.forEach((row, index) => {
          const rowPath = `${path}.model_forecasts[${index}]`;
          const comparisonFields = [
            "state", "probabilities", "confidence", "entropy", "date",
            "method", "model", "fallback", "fallback_reason",
          ];
          if (!hasExactKeys(row, comparisonFields)) {
            errors.push(`${rowPath} 필드가 v5 계약과 정확히 일치하지 않습니다.`);
            return;
          }
          if (row.model !== comparisonModels[index]) {
            errors.push(`${rowPath}.model 순서가 forecast_comparison과 일치하지 않습니다.`);
          }
          if (row.method !== "model_comparison_walk_forward_probability") {
            errors.push(`${rowPath}.method가 올바르지 않습니다.`);
          }
          if (!STATE_ORDER.includes(row.state)) errors.push(`${rowPath}.state가 올바르지 않습니다.`);
          const probabilities = validateStateVector(row.probabilities, `${rowPath}.probabilities`, errors, "예측확률");
          const rowConfidence = strictProbability(row.confidence);
          if (rowConfidence === null) errors.push(`${rowPath}.confidence가 0–1 범위가 아닙니다.`);
          if (
            probabilities
            && rowConfidence !== null
            && Math.abs(rowConfidence - probabilities[row.state]) > 0.00000001
          ) {
            errors.push(`${rowPath}.confidence는 예측 state 확률과 일치해야 합니다.`);
          }
          if (strictProbability(row.entropy) === null) errors.push(`${rowPath}.entropy가 0–1 범위가 아닙니다.`);
          if (row.date !== isoDateOffset(item.date, 7)) errors.push(`${rowPath}.date가 1주 horizon과 일치하지 않습니다.`);
          if (typeof row.model !== "string" || typeof row.fallback_reason !== "string") {
            errors.push(`${rowPath}.model과 fallback_reason은 문자열이어야 합니다.`);
          }
          if (typeof row.fallback !== "boolean") errors.push(`${rowPath}.fallback은 boolean이어야 합니다.`);
          if (probabilities && STATE_ORDER.includes(row.state)) {
            const maximum = Math.max(...STATE_ORDER.map((code) => probabilities[code]));
            if (Math.abs(probabilities[row.state] - maximum) > 0.00000001) {
              errors.push(`${rowPath}.state가 최대 예측확률과 일치하지 않습니다.`);
            }
          }
          if (row.model === model.champion) championForecast = row;
        });
        if (!championForecast) {
          errors.push(`${path}.model_forecasts에 선정 모델이 없습니다.`);
        } else {
          const scalarParityFields = [
            "state", "confidence", "entropy", "date", "model", "fallback", "fallback_reason",
          ];
          const scalarMismatch = scalarParityFields.some((field) => championForecast[field] !== forecast[field]);
          const probabilityMismatch = !isObject(championForecast.probabilities)
            || !isObject(forecast.probabilities)
            || STATE_ORDER.some(
              (code) => championForecast.probabilities[code] !== forecast.probabilities[code],
            );
          if (scalarMismatch || probabilityMismatch) {
            errors.push(`${path}.model_forecasts 선정 모델이 공식 next_week와 일치하지 않습니다.`);
          }
        }
      }
    }

    const risk = item.transition_risk;
    if (!hasExactKeys(risk, ["1w", "4w", "13w"])) {
      errors.push(`${path}.transition_risk horizon이 v5 계약과 일치하지 않습니다.`);
      return;
    }
    const riskValues = {};
    for (const horizon of TRANSITION_HORIZONS) {
      const key = `${horizon}w`;
      const row = risk[key];
      const value = strictProbability(isObject(row) ? row.probability : null);
      if (value === null) errors.push(`${path}.transition_risk.${key}.probability가 0–1 범위가 아닙니다.`);
      if (!isObject(row) || row.target_end !== isoDateOffset(item.date, 7 * horizon)) {
        errors.push(`${path}.transition_risk.${key}.target_end가 horizon과 일치하지 않습니다.`);
      }
      riskValues[key] = value;
    }
    const transitionAlias = strictProbability(item.transition_probability);
    if (transitionAlias === null) errors.push(`${path}.transition_probability가 0–1 범위가 아닙니다.`);
    if (transitionAlias !== null && riskValues["1w"] !== null && Math.abs(transitionAlias - riskValues["1w"]) > 0.00000001) {
      errors.push(`${path}.transition_probability와 transition_risk.1w가 일치하지 않습니다.`);
    }

    const directional = item.directional_risk;
    if (!hasExactKeys(directional, ["1w", "4w", "13w"])) {
      errors.push(`${path}.directional_risk horizon이 v5 계약과 일치하지 않습니다.`);
    } else {
      for (const horizon of TRANSITION_HORIZONS) {
        const key = `${horizon}w`;
        const rowPath = `${path}.directional_risk.${key}`;
        const row = directional[key];
        const directionalFields = ["probability", "no_departure", "first_destination", "target_end", "model", "method"];
        if (!hasExactKeys(row, directionalFields)) {
          errors.push(`${rowPath} 필드가 v5 계약과 일치하지 않습니다.`);
          continue;
        }
        const departure = strictProbability(row.probability);
        const noDeparture = strictProbability(row.no_departure);
        if (departure === null || riskValues[key] === null || Math.abs(departure - riskValues[key]) > 0.00000001) {
          errors.push(`${rowPath}.probability가 transition risk와 일치하지 않습니다.`);
        }
        if (departure === null || noDeparture === null || Math.abs(noDeparture - (1 - departure)) > 0.00000001) {
          errors.push(`${rowPath}.no_departure가 이탈 확률과 일치하지 않습니다.`);
        }
        const destinations = row.first_destination;
        if (!hasExactKeys(destinations, STATE_ORDER)) {
          errors.push(`${rowPath}.first_destination 키가 올바르지 않습니다.`);
        } else {
          let total = 0;
          let validDestinations = true;
          for (const code of STATE_ORDER) {
            const value = strictProbability(destinations[code]);
            if (value === null) {
              errors.push(`${rowPath}.first_destination.${code}가 0–1 범위가 아닙니다.`);
              validDestinations = false;
            } else {
              total += value;
            }
          }
          if (validDestinations && departure !== null && Math.abs(total - departure) > 0.000001) errors.push(`${rowPath}.first_destination 합계가 이탈 확률과 일치하지 않습니다.`);
          if (strictProbability(destinations[current.state]) !== 0) errors.push(`${rowPath}는 현재 state로 최초 이탈할 수 없습니다.`);
        }
        if (row.target_end !== isoDateOffset(item.date, 7 * horizon)) errors.push(`${rowPath}.target_end가 horizon과 일치하지 않습니다.`);
        if (typeof row.model !== "string" || !row.model) errors.push(`${rowPath}.model이 없습니다.`);
        if (row.method !== "first_departure_state_within_h_or_no_departure") errors.push(`${rowPath}.method가 올바르지 않습니다.`);
      }
    }

    const duration = item.duration_context;
    if (!isObject(duration)) {
      errors.push(`${path}.duration_context 객체가 없습니다.`);
    } else {
      if (!["ok", "insufficient_history", "unavailable"].includes(duration.status)) errors.push(`${path}.duration_context.status가 올바르지 않습니다.`);
      if (duration.method !== "state_specific_kaplan_meier" || duration.state !== current.state) errors.push(`${path}.duration_context state/method가 올바르지 않습니다.`);
      validateInteger(duration.elapsed_weeks, `${path}.duration_context.elapsed_weeks`, errors, 1);
      validateInteger(duration.episodes, `${path}.duration_context.episodes`, errors, 1);
      validateInteger(duration.completed_spells, `${path}.duration_context.completed_spells`, errors);
      validateInteger(duration.censored_spells, `${path}.duration_context.censored_spells`, errors);
      validateInteger(duration.minimum_completed_spells, `${path}.duration_context.minimum_completed_spells`, errors, 1);
      validateOptionalFinite(duration.median_remaining_weeks, `${path}.duration_context.median_remaining_weeks`, errors, 0);
      validateOptionalFinite(duration.restricted_mean_remaining_weeks, `${path}.duration_context.restricted_mean_remaining_weeks`, errors, 0);
      if (duration.restriction_weeks !== 52) errors.push(`${path}.duration_context.restriction_weeks는 52여야 합니다.`);
      const survival = duration.conditional_survival;
      const departure = duration.departure_probability;
      if (!hasExactKeys(survival, ["4w", "13w"]) || !hasExactKeys(departure, ["4w", "13w"])) {
        errors.push(`${path}.duration_context 4w/13w 생존·이탈 키가 올바르지 않습니다.`);
      } else {
        for (const key of ["4w", "13w"]) {
          validateOptionalFinite(survival[key], `${path}.duration_context.conditional_survival.${key}`, errors, 0, 1);
          validateOptionalFinite(departure[key], `${path}.duration_context.departure_probability.${key}`, errors, 0, 1);
          const survivalValue = survival[key] === null ? null : strictProbability(survival[key]);
          const departureValue = departure[key] === null ? null : strictProbability(departure[key]);
          if ((survivalValue === null) !== (departureValue === null)) errors.push(`${path}.duration_context.${key} 생존·이탈 null 상태가 다릅니다.`);
          if (survivalValue !== null && departureValue !== null && Math.abs(survivalValue + departureValue - 1) > 0.00000001) errors.push(`${path}.duration_context.${key} 생존·이탈 합계가 1이 아닙니다.`);
        }
      }
      const bootstrap = duration.bootstrap;
      if (!isObject(bootstrap) || bootstrap.unit !== "episode") {
        errors.push(`${path}.duration_context.bootstrap.unit이 올바르지 않습니다.`);
      } else {
        validateInteger(bootstrap.resamples, `${path}.duration_context.bootstrap.resamples`, errors);
        validateInteger(bootstrap.valid_resamples, `${path}.duration_context.bootstrap.valid_resamples`, errors);
        validateInteger(bootstrap.seed, `${path}.duration_context.bootstrap.seed`, errors);
        validateOptionalFinite(bootstrap.interval, `${path}.duration_context.bootstrap.interval`, errors, 0, 1);
      }
      if (duration.ci95 !== null && duration.ci95 !== undefined && !isObject(duration.ci95)) errors.push(`${path}.duration_context.ci95는 객체 또는 null이어야 합니다.`);
    }

    const fx = item.fx_context;
    if (!isObject(fx)) {
      errors.push(`${path}.fx_context 객체가 없습니다.`);
    } else {
      if (!["ok", "partial", "degraded", "stale", "insufficient_history", "unavailable"].includes(fx.status)) errors.push(`${path}.fx_context.status가 올바르지 않습니다.`);
      if (fx.method !== "fed_h10_usd_strength") errors.push(`${path}.fx_context.method가 올바르지 않습니다.`);
      if (!Array.isArray(fx.bilateral_panel) || fx.bilateral_panel.length !== FX_BILATERAL_PANEL.length || fx.bilateral_panel.some((code, index) => code !== FX_BILATERAL_PANEL[index])) {
        errors.push(`${path}.fx_context.bilateral_panel이 고정 9통화 panel과 일치하지 않습니다.`);
      }
      if (!isObject(fx.coverage)) {
        errors.push(`${path}.fx_context.coverage 객체가 없습니다.`);
      } else {
        validateInteger(fx.coverage.available_pairs, `${path}.fx_context.coverage.available_pairs`, errors);
        validateInteger(fx.coverage.required_pairs, `${path}.fx_context.coverage.required_pairs`, errors);
        if (fx.coverage.required_pairs !== FX_BILATERAL_PANEL.length || fx.coverage.available_pairs > fx.coverage.required_pairs) {
          errors.push(`${path}.fx_context coverage가 9통화 panel과 일치하지 않습니다.`);
        }
      }
      for (const blockName of ["indexes", "bilateral"]) {
        if (!isObject(fx[blockName])) errors.push(`${path}.fx_context.${blockName} 객체가 없습니다.`);
        else for (const [name, value] of Object.entries(fx[blockName])) validateOptionalFinite(value, `${path}.fx_context.${blockName}.${name}`, errors);
      }
    }

    const contextScores = item.context_scores;
    if (!hasExactKeys(contextScores, Object.keys(FACTOR_META))) {
      errors.push(`${path}.context_scores 키가 올바르지 않습니다.`);
    } else {
      for (const [name, value] of Object.entries(contextScores)) {
        const number = strictFiniteNumber(value);
        if (number === null || number < -1 || number > 1) errors.push(`${path}.context_scores.${name}가 -1–1 범위가 아닙니다.`);
      }
    }
    if (!Array.isArray(item.extreme_context)) {
      errors.push(`${path}.extreme_context 배열이 없습니다.`);
    } else {
      item.extreme_context.forEach((row, index) => {
        const rowPath = `${path}.extreme_context[${index}]`;
        if (!hasExactKeys(row, ["feature", "label", "z_score", "position", "method"])) errors.push(`${rowPath} 필드가 올바르지 않습니다.`);
        if (isObject(row) && ("impact" in row || "direction" in row)) errors.push(`${rowPath}에 attribution 필드가 있습니다.`);
        if (!isObject(row) || strictFiniteNumber(row.z_score) === null || !["high", "low"].includes(row.position)) errors.push(`${rowPath} z-score/position이 올바르지 않습니다.`);
      });
    }
    if ("scores" in item || "top_drivers" in item) errors.push(`${path}에 제거된 v4 semantic field가 있습니다.`);
    if (typeof item.summary !== "string" || !item.summary.trim()) errors.push(`${path}.summary가 없습니다.`);
    if (!isObject(item.market) || !isObject(item.health)) errors.push(`${path}.market/health 객체가 없습니다.`);
  }

  function validateV5ResearchContract(research, model, errors) {
    const stats = isObject(research) ? research.conditional_asset_stats : null;
    if (!isObject(stats)) {
      errors.push("v5 research.conditional_asset_stats 객체가 없습니다.");
      return;
    }
    if (stats.method !== "state_conditioned_forward_total_return") errors.push("v5 conditional asset method가 올바르지 않습니다.");
    if (stats.role !== "descriptive_only") errors.push("v5 conditional asset role은 descriptive_only여야 합니다.");
    if (stats.execution_lag_weeks !== 1 || stats.return_currency !== "USD") errors.push("v5 conditional asset 실행시점/통화가 올바르지 않습니다.");
    if (!Array.isArray(stats.horizons_weeks) || stats.horizons_weeks.length !== TRANSITION_HORIZONS.length || stats.horizons_weeks.some((value, index) => value !== TRANSITION_HORIZONS[index])) {
      errors.push("v5 conditional asset horizons가 올바르지 않습니다.");
    }
    if (!Array.isArray(stats.assets) || stats.assets.length !== OUTCOME_ASSETS.length || stats.assets.some((value, index) => value !== OUTCOME_ASSETS[index])) {
      errors.push("v5 conditional asset 목록이 올바르지 않습니다.");
    }
    if (!Array.isArray(stats.rows)) {
      errors.push("v5 conditional asset rows 배열이 없습니다.");
      return;
    }
    const forbidden = ["weight", "allocation", "position", "signal", "target_weight"];
    const metrics = [
      "mean_return", "median_return", "positive_rate", "annualized_volatility",
      "downside_volatility", "cvar_5", "mean_max_drawdown",
    ];
    const combinations = new Set();
    stats.rows.forEach((row, index) => {
      const path = `research.conditional_asset_stats.rows[${index}]`;
      if (!isObject(row)) {
        errors.push(`${path}는 객체여야 합니다.`);
        return;
      }
      if (forbidden.some((field) => field in row)) errors.push(`${path}에 allocation 의미 필드가 있습니다.`);
      if (!OUTCOME_ASSETS.includes(row.asset) || !STATE_ORDER.includes(row.state) || !TRANSITION_HORIZONS.includes(row.horizon_weeks)) {
        errors.push(`${path} asset/state/horizon이 올바르지 않습니다.`);
      } else {
        const combination = `${row.asset}|${row.state}|${row.horizon_weeks}`;
        if (combinations.has(combination)) errors.push(`${path} asset/state/horizon이 중복됩니다.`);
        combinations.add(combination);
      }
      if (row.execution_lag_weeks !== 1 || row.return_currency !== "USD") errors.push(`${path} 실행시점/통화가 올바르지 않습니다.`);
      if (row.bootstrap_method !== "episode_bounded_circular_block" || row.bootstrap_block_weeks !== 13) errors.push(`${path} bootstrap 방법이 올바르지 않습니다.`);
      if (row.bootstrap_resamples !== model.execution_parameters.conditional_outcome_bootstrap_resamples) errors.push(`${path} bootstrap 횟수가 실행 계약과 일치하지 않습니다.`);
      if (row.minimum_observations !== 20 || row.minimum_unique_episodes !== 5) errors.push(`${path} 지원 기준이 올바르지 않습니다.`);
      validateInteger(row.n, `${path}.n`, errors);
      validateInteger(row.unique_episodes, `${path}.unique_episodes`, errors);
      if (!["ok", "insufficient_support"].includes(row.status)) errors.push(`${path}.status가 올바르지 않습니다.`);
      for (const metric of metrics) {
        validateOptionalFinite(row[metric], `${path}.${metric}`, errors);
        const lower = row[`${metric}_ci95_lower`];
        const upper = row[`${metric}_ci95_upper`];
        validateOptionalFinite(lower, `${path}.${metric}_ci95_lower`, errors);
        validateOptionalFinite(upper, `${path}.${metric}_ci95_upper`, errors);
        if ((lower == null) !== (upper == null)) errors.push(`${path}.${metric} CI null 상태가 다릅니다.`);
        if (lower != null && upper != null && lower > upper) errors.push(`${path}.${metric} CI 순서가 뒤집혔습니다.`);
      }
    });
    if (combinations.size !== OUTCOME_ASSETS.length * STATE_ORDER.length * TRANSITION_HORIZONS.length) {
      errors.push("v5 conditional asset rows는 모든 asset/state/horizon 조합을 포함해야 합니다.");
    }

    const modelConditioned = isObject(research)
      ? research.model_conditioned_asset_stats
      : null;
    const researchArtifacts = isObject(model) && isObject(model.research_artifacts)
      ? model.research_artifacts
      : {};
    const conditionedArtifactCount = V5_MODEL_CONDITIONED_RESEARCH_ARTIFACTS
      .filter((key) => Object.hasOwn(researchArtifacts, key)).length;
    if (modelConditioned == null) {
      errors.push("v5 research.model_conditioned_asset_stats 객체가 필요합니다.");
      return;
    }
    if (!isObject(modelConditioned)) {
      errors.push("v5 research.model_conditioned_asset_stats 객체가 올바르지 않습니다.");
      return;
    }
    if (conditionedArtifactCount !== V5_MODEL_CONDITIONED_RESEARCH_ARTIFACTS.length) {
      errors.push("v5 model-conditioned research artifact는 완전한 pair여야 합니다.");
    }

    if (modelConditioned.method !== "oos_one_week_forecast_conditioned_forward_total_return") {
      errors.push("v5 model-conditioned asset method가 올바르지 않습니다.");
    }
    if (modelConditioned.role !== "retrospective_model_diagnostic") {
      errors.push("v5 model-conditioned asset role이 올바르지 않습니다.");
    }
    if (
      modelConditioned.conditioning !== "hard_argmax_oos_forecast"
      || modelConditioned.forecast_horizon_weeks !== 1
      || modelConditioned.execution_lag_weeks !== 1
      || modelConditioned.return_currency !== "USD"
    ) {
      errors.push("v5 model-conditioned asset conditioning/horizon/통화가 올바르지 않습니다.");
    }
    if (
      !Array.isArray(modelConditioned.horizons_weeks)
      || modelConditioned.horizons_weeks.length !== TRANSITION_HORIZONS.length
      || modelConditioned.horizons_weeks.some((value, index) => value !== TRANSITION_HORIZONS[index])
    ) {
      errors.push("v5 model-conditioned asset horizons가 올바르지 않습니다.");
    }
    if (
      !Array.isArray(modelConditioned.assets)
      || modelConditioned.assets.length !== OUTCOME_ASSETS.length
      || modelConditioned.assets.some((value, index) => value !== OUTCOME_ASSETS[index])
    ) {
      errors.push("v5 model-conditioned asset 목록이 올바르지 않습니다.");
    }
    const expectedModels = isObject(model.forecast_comparison)
      && Array.isArray(model.forecast_comparison.models)
      ? model.forecast_comparison.models
      : [];
    if (
      !Array.isArray(modelConditioned.models)
      || modelConditioned.models.length !== expectedModels.length
      || modelConditioned.models.some((value, index) => value !== expectedModels[index])
    ) {
      errors.push("v5 model-conditioned asset models가 forecast_comparison과 일치하지 않습니다.");
    }
    if (!Array.isArray(modelConditioned.rows)) {
      errors.push("v5 model-conditioned asset rows 배열이 없습니다.");
      return;
    }

    const expectedConditionedRows = expectedModels.length
      * OUTCOME_ASSETS.length
      * STATE_ORDER.length
      * TRANSITION_HORIZONS.length;
    if (modelConditioned.rows.length !== expectedConditionedRows) {
      errors.push("v5 model-conditioned asset rows 행 수가 올바르지 않습니다.");
    }
    const conditionedCombinations = new Set();
    modelConditioned.rows.forEach((row, index) => {
      const path = `research.model_conditioned_asset_stats.rows[${index}]`;
      if (!isObject(row)) {
        errors.push(`${path}는 객체여야 합니다.`);
        return;
      }
      if (forbidden.some((field) => field in row)) errors.push(`${path}에 allocation 의미 필드가 있습니다.`);
      const conditioningModel = row.conditioning_model;
      if (!expectedModels.includes(conditioningModel)) {
        errors.push(`${path}.conditioning_model이 올바르지 않습니다.`);
      }
      if (!OUTCOME_ASSETS.includes(row.asset) || !STATE_ORDER.includes(row.state) || !TRANSITION_HORIZONS.includes(row.horizon_weeks)) {
        errors.push(`${path} model/asset/state/horizon이 올바르지 않습니다.`);
      } else if (expectedModels.includes(conditioningModel)) {
        const combination = `${conditioningModel}|${row.asset}|${row.state}|${row.horizon_weeks}`;
        if (conditionedCombinations.has(combination)) errors.push(`${path} model/asset/state/horizon이 중복됩니다.`);
        conditionedCombinations.add(combination);
      }
      if (row.execution_lag_weeks !== 1 || row.return_currency !== "USD") errors.push(`${path} 실행시점/통화가 올바르지 않습니다.`);
      if (row.bootstrap_method !== "episode_bounded_circular_block" || row.bootstrap_block_weeks !== 13) errors.push(`${path} bootstrap 방법이 올바르지 않습니다.`);
      if (row.bootstrap_resamples !== model.execution_parameters.conditional_outcome_bootstrap_resamples) errors.push(`${path} bootstrap 횟수가 실행 계약과 일치하지 않습니다.`);
      if (row.minimum_observations !== 20 || row.minimum_unique_episodes !== 5) errors.push(`${path} 지원 기준이 올바르지 않습니다.`);
      validateInteger(row.n, `${path}.n`, errors);
      validateInteger(row.unique_episodes, `${path}.unique_episodes`, errors);
      if (!["ok", "insufficient_support"].includes(row.status)) errors.push(`${path}.status가 올바르지 않습니다.`);
      for (const metric of metrics) {
        validateOptionalFinite(row[metric], `${path}.${metric}`, errors);
        const lower = row[`${metric}_ci95_lower`];
        const upper = row[`${metric}_ci95_upper`];
        validateOptionalFinite(lower, `${path}.${metric}_ci95_lower`, errors);
        validateOptionalFinite(upper, `${path}.${metric}_ci95_upper`, errors);
        if ((lower == null) !== (upper == null)) errors.push(`${path}.${metric} CI null 상태가 다릅니다.`);
        if (lower != null && upper != null && lower > upper) errors.push(`${path}.${metric} CI 순서가 뒤집혔습니다.`);
      }
    });
    if (conditionedCombinations.size !== expectedConditionedRows) {
      errors.push("v5 model-conditioned asset rows는 모든 model/asset/state/horizon 조합을 포함해야 합니다.");
    }
  }

  function validatePayload(payload) {
    const errors = [];
    const warnings = [];

    if (!isObject(payload)) {
      return { errors: ["최상위 JSON은 객체여야 합니다."], warnings, weekly: [] };
    }

    if (!isObject(payload.meta)) errors.push("meta 객체가 없습니다.");
    if (!Array.isArray(payload.states)) errors.push("states 배열이 없습니다.");
    if (!isObject(payload.model)) errors.push("model 객체가 없습니다.");
    if (!Array.isArray(payload.weekly)) errors.push("weekly 배열이 없습니다.");
    if (!Array.isArray(payload.sources)) errors.push("sources 배열이 없습니다.");
    if (!Array.isArray(payload.feature_catalog)) errors.push("feature_catalog 배열이 없습니다.");

    if (errors.length || !Array.isArray(payload.weekly)) return { errors, warnings, weekly: [] };

    const declaredResultVersion = payload.meta.result_version;
    const isV3 = declaredResultVersion === V3_RESULT_VERSION;
    const isV4 = declaredResultVersion === V4_RESULT_VERSION;
    const isV5 = declaredResultVersion === V5_RESULT_VERSION;
    if (![V3_RESULT_VERSION, V4_RESULT_VERSION, V5_RESULT_VERSION].includes(declaredResultVersion)) {
      errors.push(`지원하지 않는 meta.result_version입니다: ${textValue(declaredResultVersion)}`);
    }
    const expectedSchemaVersion = isV5 ? V5_SCHEMA_VERSION : "1.0.0";
    if (payload.meta.schema_version !== expectedSchemaVersion) errors.push(`schema_version은 ${expectedSchemaVersion}이어야 합니다.`);
    if (isV5) {
      for (const field of ["label", "forecast", "selection", "research"]) {
        if (!isObject(payload[field])) errors.push(`v5 ${field} 객체가 없습니다.`);
      }
    }
    if (!payload.meta.generated_at) errors.push("meta.generated_at이 없습니다.");
    if (!payload.meta.data_as_of) errors.push("meta.data_as_of가 없습니다.");
    if (!isV5 && !payload.meta.mode) errors.push("meta.mode가 없습니다.");
    if (payload.meta.timezone !== "America/New_York") {
      errors.push("meta.timezone은 America/New_York이어야 합니다.");
    }
    const stateIds = payload.states.map((item) => isObject(item) ? item.id : null);
    if (stateIds.length !== STATE_ORDER.length || stateIds.some((item, index) => item !== STATE_ORDER[index])) {
      errors.push("states는 risk_on, transition, risk_off 순서여야 합니다.");
    }
    if (!payload.model.champion) errors.push("model.champion이 없습니다.");
    const expectedSelectionStatus = isV5 ? "selected_by_gate" : "provisional_predeployment";
    if (payload.model.selection_status !== expectedSelectionStatus) {
      errors.push(`model.selection_status는 ${expectedSelectionStatus}여야 합니다.`);
    }
    if (!Array.isArray(payload.model.leaderboard)) errors.push("model.leaderboard 배열이 없습니다.");
    if (!payload.feature_catalog.length || payload.feature_catalog.some((item) => !isObject(item))) {
      errors.push("feature_catalog는 비어 있지 않은 객체 배열이어야 합니다.");
    }

    const isTransitionContract = isV3 || isV4;
    if (isTransitionContract) {
      if (typeof payload.meta.generation_id !== "string" || !payload.meta.generation_id) {
        errors.push("구조적 결과 meta.generation_id는 비어 있지 않은 문자열이어야 합니다.");
      }
      const expectedModelVersion = isV4 ? V4_MODEL_VERSION : V3_MODEL_VERSION;
      const expectedLabelVersion = isV4 ? V4_LABEL_VERSION : V3_LABEL_VERSION;
      const expectedFeatureSetVersion = isV4 ? V4_FEATURE_SET_VERSION : V3_FEATURE_SET_VERSION;
      if (payload.model.version !== expectedModelVersion) {
        errors.push(`구조적 결과 model.version은 ${expectedModelVersion}이어야 합니다.`);
      }
      if (payload.model.label_version !== expectedLabelVersion) {
        errors.push(`구조적 결과 model.label_version은 ${expectedLabelVersion}이어야 합니다.`);
      }
      if (payload.model.feature_set_version !== expectedFeatureSetVersion) {
        errors.push(`구조적 결과 model.feature_set_version은 ${expectedFeatureSetVersion}이어야 합니다.`);
      }
      const horizons = payload.model.transition_horizons_weeks;
      if (
        !Array.isArray(horizons)
        || horizons.length !== TRANSITION_HORIZONS.length
        || horizons.some((value, index) => value !== TRANSITION_HORIZONS[index])
      ) {
        errors.push("v3 model.transition_horizons_weeks는 1, 4, 13 순서여야 합니다.");
      }
      if (payload.model.primary_horizon_weeks !== 1) {
        errors.push("v3 model.primary_horizon_weeks는 1이어야 합니다.");
      }
      if (!isIsoDate(payload.model.transition_selection_end)) {
        errors.push("v3 model.transition_selection_end는 YYYY-MM-DD 형식의 실제 날짜여야 합니다.");
      }
      const baseline = payload.model.baseline_v2;
      if (!isObject(baseline)) {
        errors.push("v3 model.baseline_v2 객체가 없습니다.");
      } else {
        if (baseline.result_version !== "weekly-regime-result-v2") {
          errors.push("v3 model.baseline_v2.result_version이 올바르지 않습니다.");
        }
        for (const field of ["label_version", "model_version", "champion"]) {
          if (typeof baseline[field] !== "string" || !baseline[field]) {
            errors.push(`v3 model.baseline_v2.${field}는 비어 있지 않은 문자열이어야 합니다.`);
          }
        }
        for (const field of ["payload_sha256", "artifacts_inventory_sha256"]) {
          if (typeof baseline[field] !== "string" || !/^[0-9a-f]{64}$/.test(baseline[field])) {
            errors.push(`v3 model.baseline_v2.${field}는 소문자 SHA-256이어야 합니다.`);
          }
        }
      }
      const transitionChampions = payload.model.transition_champions;
      const championKeys = isObject(transitionChampions) ? Object.keys(transitionChampions) : [];
      if (
        !isObject(transitionChampions)
        || championKeys.length !== 3
        || championKeys.some((key) => !["1w", "4w", "13w"].includes(key))
      ) {
        errors.push("v3 model.transition_champions 키는 1w, 4w, 13w와 정확히 일치해야 합니다.");
      } else if (championKeys.some((key) => typeof transitionChampions[key] !== "string" || !transitionChampions[key])) {
        errors.push("v3 model.transition_champions 값은 비어 있지 않은 문자열이어야 합니다.");
      }
      if (!Array.isArray(payload.model.transition_leaderboard)) {
        errors.push("v3 model.transition_leaderboard 배열이 없습니다.");
      } else {
        const coveredHorizons = new Set();
        payload.model.transition_leaderboard.forEach((row, index) => {
          const rowPath = `model.transition_leaderboard[${index}]`;
          if (!isObject(row)) {
            errors.push(`${rowPath}는 객체여야 합니다.`);
            return;
          }
          if (!TRANSITION_HORIZONS.includes(row.horizon_weeks)) {
            errors.push(`${rowPath}.horizon_weeks가 1, 4, 13 중 하나가 아닙니다.`);
          } else {
            coveredHorizons.add(row.horizon_weeks);
          }
          if (typeof row.model !== "string" || !row.model.trim()) errors.push(`${rowPath}.model이 없습니다.`);
          if (typeof row.selected !== "boolean") errors.push(`${rowPath}.selected는 boolean이어야 합니다.`);
          if (!["selection", "retrospective_diagnostic"].includes(row.evaluation_split)) {
            errors.push(`${rowPath}.evaluation_split이 지원 범위가 아닙니다.`);
          }
          const binaryLogLoss = strictFiniteNumber(row.binary_log_loss);
          if (binaryLogLoss === null || binaryLogLoss < 0) {
            errors.push(`${rowPath}.binary_log_loss가 0 이상의 유한한 숫자가 아닙니다.`);
          }
          for (const metric of ["brier", "precision", "recall"]) {
            if (strictProbability(row[metric]) === null) errors.push(`${rowPath}.${metric}이 0–1 범위의 숫자가 아닙니다.`);
          }
          const countMetrics = [
            "n_predictions", "event_count", "non_event_count", "fallback_count", "calibration_fallback_count",
          ];
          for (const metric of countMetrics) {
            const value = strictFiniteNumber(row[metric]);
            if (value === null || value < 0 || !Number.isInteger(value)) {
              errors.push(`${rowPath}.${metric}이 0 이상의 정수가 아닙니다.`);
            }
          }
          const falseAlarms = strictFiniteNumber(row.false_alarms_per_year);
          if (falseAlarms === null || falseAlarms < 0) {
            errors.push(`${rowPath}.false_alarms_per_year가 0 이상의 유한한 숫자가 아닙니다.`);
          }
          const eventCount = strictFiniteNumber(row.event_count);
          const nonEventCount = strictFiniteNumber(row.non_event_count);
          const predictionCount = strictFiniteNumber(row.n_predictions);
          if (
            Number.isInteger(eventCount)
            && Number.isInteger(nonEventCount)
            && Number.isInteger(predictionCount)
            && predictionCount !== eventCount + nonEventCount
          ) {
            errors.push(`${rowPath}의 n_predictions는 event_count와 non_event_count의 합이어야 합니다.`);
          }
          const noEventAveragePrecision = row.average_precision === null && eventCount === 0;
          if (!noEventAveragePrecision && strictProbability(row.average_precision) === null) {
            errors.push(`${rowPath}.average_precision은 0–1 범위 숫자이거나 무이벤트 구간의 null이어야 합니다.`);
          }
        });
        if (TRANSITION_HORIZONS.some((horizon) => !coveredHorizons.has(horizon))) {
          errors.push("v3 transition_leaderboard는 1, 4, 13주 결과를 모두 포함해야 합니다.");
        }
      }
      if (!isObject(payload.model.shadow_nowcast) || payload.model.shadow_nowcast.status !== "shadow_only") {
        errors.push("v3 model.shadow_nowcast는 shadow_only 요약이어야 합니다.");
      } else if (payload.model.shadow_nowcast.canonical_target !== false) {
        errors.push("v3 model.shadow_nowcast.canonical_target은 false여야 합니다.");
      }

      if (isV4) {
        const baselineV3Keys = [
          "result_version", "label_version", "model_version", "champion",
          "payload_sha256", "artifacts_inventory_sha256", "captured_at",
        ];
        const baselineV3 = payload.model.baseline_v3;
        if (!hasExactKeys(baselineV3, baselineV3Keys)) {
          errors.push("v4 model.baseline_v3 필드가 정확한 계약과 일치하지 않습니다.");
        } else {
          for (const [field, expected] of Object.entries(FROZEN_V4_BASELINE_V3)) {
            if (baselineV3[field] !== expected) errors.push(`v4 baseline_v3.${field}가 frozen 계약과 일치하지 않습니다.`);
          }
          for (const field of ["payload_sha256", "artifacts_inventory_sha256"]) {
            if (!isLowerSha256(baselineV3[field])) errors.push(`v4 baseline_v3.${field}는 소문자 SHA-256이어야 합니다.`);
          }
          if (!isIsoTimestamp(baselineV3.captured_at)) errors.push("v4 baseline_v3.captured_at은 ISO timestamp여야 합니다.");
        }

        const preregistration = payload.model.structural_preregistration;
        if (!hasExactKeys(preregistration, ["path", "sha256"])) {
          errors.push("v4 model.structural_preregistration 필드가 정확한 계약과 일치하지 않습니다.");
        } else {
          for (const [field, expected] of Object.entries(FROZEN_V4_STRUCTURAL_PREREGISTRATION)) {
            if (preregistration[field] !== expected) errors.push(`v4 structural_preregistration.${field}가 frozen 계약과 일치하지 않습니다.`);
          }
          if (!isLowerSha256(preregistration.sha256)) errors.push("v4 structural_preregistration.sha256은 소문자 SHA-256이어야 합니다.");
        }
        if (!isLowerSha256(payload.model.feature_manifest_sha256)) {
          errors.push("v4 model.feature_manifest_sha256은 소문자 SHA-256이어야 합니다.");
        }
        const evidenceArtifacts = payload.model.evidence_artifacts;
        if (!hasExactKeys(evidenceArtifacts, ["state_label_history", "weekly_state_forecasts"])) {
          errors.push("v4 model.evidence_artifacts 필드가 정확한 계약과 일치하지 않습니다.");
        } else {
          const labelHistory = evidenceArtifacts.state_label_history;
          if (
            !hasExactKeys(labelHistory, [
              "path", "row_count", "sha256", "label_fit_weeks", "label_fit_end", "initial_state",
            ])
            || labelHistory.path !== "state-label-history.csv"
            || !Number.isInteger(labelHistory.row_count)
            || labelHistory.row_count < 520
            || !isLowerSha256(labelHistory.sha256)
            || labelHistory.label_fit_weeks !== 520
            || !isIsoTimestamp(labelHistory.label_fit_end)
            || labelHistory.initial_state !== "transition"
          ) {
            errors.push("v4 state_label_history artifact 계약이 올바르지 않습니다.");
          }
          const weeklyForecasts = evidenceArtifacts.weekly_state_forecasts;
          if (
            !hasExactKeys(weeklyForecasts, ["path", "row_count", "sha256"])
            || weeklyForecasts.path !== "weekly-state-forecasts.csv"
            || !Number.isInteger(weeklyForecasts.row_count)
            || weeklyForecasts.row_count < 1
            || !isLowerSha256(weeklyForecasts.sha256)
          ) {
            errors.push("v4 weekly_state_forecasts artifact 계약이 올바르지 않습니다.");
          }
        }

        const structuralModels = payload.model.structural_models;
        const structuralModelKeys = ["xgb_hazard_destination", "causal_dynamic_ensemble", "joint_survival_hazard"];
        if (!hasExactKeys(structuralModels, structuralModelKeys)) {
          errors.push("v4 model.structural_models 키가 정확한 계약과 일치하지 않습니다.");
        } else {
          const hazard = structuralModels.xgb_hazard_destination;
          if (
            !hasExactKeys(hazard, ["hazard_model", "destination_model", "direct_jump_floor"])
            || hazard.hazard_model !== "binary_xgboost"
            || hazard.destination_model !== "xgboost"
            || hazard.direct_jump_floor !== 0.000001
          ) {
            errors.push("v4 xgb_hazard_destination 계약이 올바르지 않습니다.");
          }

          const ensemble = structuralModels.causal_dynamic_ensemble;
          const expectedExperts = ["markov", "xgboost", "xgb_hazard_destination"];
          if (
            !hasExactKeys(ensemble, ["experts", "half_life_weeks", "minimum_history_rows", "eligible_loss_rule"])
            || !Array.isArray(ensemble.experts)
            || ensemble.experts.length !== expectedExperts.length
            || ensemble.experts.some((expert, index) => expert !== expectedExperts[index])
            || ensemble.half_life_weeks !== 52
            || ensemble.minimum_history_rows !== 26
            || ensemble.eligible_loss_rule !== "target_date_strictly_before_origin"
          ) {
            errors.push("v4 causal_dynamic_ensemble 계약이 올바르지 않습니다.");
          }

          const survival = structuralModels.joint_survival_hazard;
          if (
            !hasExactKeys(survival, ["base_target_weeks", "horizons_weeks", "future_covariates", "identity"])
            || survival.base_target_weeks !== 1
            || !Array.isArray(survival.horizons_weeks)
            || survival.horizons_weeks.length !== TRANSITION_HORIZONS.length
            || survival.horizons_weeks.some((value, index) => value !== TRANSITION_HORIZONS[index])
            || survival.future_covariates !== "origin_values_frozen"
            || survival.identity !== "one_minus_product_one_minus_weekly_hazard"
          ) {
            errors.push("v4 joint_survival_hazard 계약이 올바르지 않습니다.");
          }
        }

        const ablation = payload.model.ablation;
        const ablationKeys = [
          "anchor_model", "reference_variant", "published_variant", "primary_period",
          "post_2023_role", "may_change_published_variant", "manifest_sha256",
        ];
        if (
          !hasExactKeys(ablation, ablationKeys)
          || ablation.anchor_model !== "xgboost"
          || ablation.reference_variant !== "legacy_v3"
          || ablation.published_variant !== "all_structural"
          || ablation.primary_period !== "pre_2023_selection_oos"
          || ablation.post_2023_role !== "retrospective_diagnostic_only"
          || ablation.may_change_published_variant !== false
          || !isLowerSha256(ablation.manifest_sha256)
        ) {
          errors.push("v4 model.ablation 계약이 올바르지 않습니다.");
        }
      }
    }
    if (isV5) validateV5ModelContract(payload, errors);

    const seenDates = new Set();
    const weekly = [];
    let previousDate = null;
    for (let index = 0; index < payload.weekly.length; index += 1) {
      const item = payload.weekly[index];
      const path = `weekly[${index}]`;
      if (!isObject(item)) {
        errors.push(`${path}는 객체여야 합니다.`);
        continue;
      }
      if (!isIsoDate(item.date)) {
        errors.push(`${path}.date는 YYYY-MM-DD 형식의 실제 날짜여야 합니다.`);
        continue;
      }
      if (seenDates.has(item.date)) {
        errors.push(`${item.date} 주간 결과가 중복되어 있습니다.`);
        continue;
      }
      seenDates.add(item.date);
      if (previousDate !== null && item.date <= previousDate) {
        errors.push("weekly 관측치는 중복 없이 날짜 오름차순이어야 합니다.");
      }
      previousDate = item.date;

      if (!isObject(item.current)) errors.push(`${path}.current 객체가 없습니다.`);
      if (!isObject(item.next_week)) errors.push(`${path}.next_week 객체가 없습니다.`);
      if (isObject(item.current) && !STATE_ORDER.includes(item.current.state)) {
        errors.push(`${item.date} 현재 국면 코드가 표준 세 상태와 일치하지 않습니다.`);
      }
      if (isObject(item.next_week) && !STATE_ORDER.includes(item.next_week.state)) {
        errors.push(`${item.date} 다음 주 국면 코드가 표준 세 상태와 일치하지 않습니다.`);
      }

      for (const horizon of isV5 ? [] : ["current", "next_week"]) {
        const probabilities = isObject(item[horizon]) ? item[horizon].probabilities : null;
        if (!isObject(probabilities)) {
          errors.push(`${item.date} ${horizon} 확률 객체가 없습니다.`);
          continue;
        }
        const probabilityKeys = Object.keys(probabilities);
        if (
          probabilityKeys.length !== STATE_ORDER.length
          || probabilityKeys.some((code) => !STATE_ORDER.includes(code))
        ) {
          errors.push(`${item.date} ${horizon} 확률 키는 표준 세 상태와 정확히 일치해야 합니다.`);
          continue;
        }
        let sum = 0;
        let count = 0;
        for (const code of STATE_ORDER) {
          const value = probability(probabilities[code]);
          if (value === null) {
            errors.push(`${item.date} ${horizon}.${code} 확률이 0–1 범위의 숫자가 아닙니다.`);
          } else {
            sum += value;
            count += 1;
          }
        }
        if (count === STATE_ORDER.length && Math.abs(sum - 1) > 0.00001) {
          errors.push(`${item.date} ${horizon} 확률 합계가 1이 아닙니다.`);
        }
      }
      if (probability(item.transition_probability) === null) {
        errors.push(`${item.date} transition_probability가 0–1 범위가 아닙니다.`);
      }
      if (isV5) validateV5WeekContract(item, path, errors, payload.model);
      if (isTransitionContract) {
        const transitionRisk = item.transition_risk;
        const expectedKeys = ["1w", "4w", "13w"];
        if (!isObject(transitionRisk)) {
          errors.push(`${item.date} v3 transition_risk 객체가 없습니다.`);
        } else {
          const horizonKeys = Object.keys(transitionRisk);
          if (
            horizonKeys.length !== expectedKeys.length
            || horizonKeys.some((key) => !expectedKeys.includes(key))
          ) {
            errors.push(`${item.date} transition_risk 키는 1w, 4w, 13w와 정확히 일치해야 합니다.`);
          }
          const exactRiskKeys = ["probability", "target_end", "model", "threshold", "fallback", "fallback_reason"];
          for (const key of expectedKeys) {
            const result = transitionRisk[key];
            const resultPath = `${item.date} transition_risk.${key}`;
            if (!isObject(result)) {
              errors.push(`${resultPath} 객체가 없습니다.`);
              continue;
            }
            const resultKeys = Object.keys(result);
            if (
              resultKeys.length !== exactRiskKeys.length
              || resultKeys.some((field) => !exactRiskKeys.includes(field))
            ) {
              errors.push(`${resultPath} 필드가 v3 계약과 정확히 일치하지 않습니다.`);
            }
            if (strictProbability(result.probability) === null) errors.push(`${resultPath}.probability가 0–1 범위가 아닙니다.`);
            if (!isIsoDate(result.target_end)) errors.push(`${resultPath}.target_end가 실제 날짜가 아닙니다.`);
            if (typeof result.model !== "string" || !result.model.trim()) errors.push(`${resultPath}.model이 없습니다.`);
            if (strictProbability(result.threshold) === null) errors.push(`${resultPath}.threshold가 0–1 범위가 아닙니다.`);
            if (typeof result.fallback !== "boolean") errors.push(`${resultPath}.fallback은 boolean이어야 합니다.`);
            if (typeof result.fallback_reason !== "string") {
              errors.push(`${resultPath}.fallback_reason은 문자열이어야 합니다.`);
            }
            const expectedTargetEnd = isoDateOffset(item.date, 7 * Number.parseInt(key, 10));
            if (result.target_end !== expectedTargetEnd) {
              errors.push(`${resultPath}.target_end는 관측일로부터 정확히 ${Number.parseInt(key, 10)}주 뒤여야 합니다.`);
            }
          }
          const primaryRisk = isObject(transitionRisk["1w"]) ? strictProbability(transitionRisk["1w"].probability) : null;
          const legacyRisk = strictProbability(item.transition_probability);
          if (primaryRisk !== null && legacyRisk !== null && Math.abs(primaryRisk - legacyRisk) > 0.00000001) {
            errors.push(`${item.date} transition_probability와 transition_risk.1w.probability가 일치하지 않습니다.`);
          }
          const currentState = isObject(item.current) ? item.current.state : null;
          const stayProbability = STATE_ORDER.includes(currentState)
            && isObject(item.next_week)
            && isObject(item.next_week.probabilities)
            ? strictProbability(item.next_week.probabilities[currentState])
            : null;
          if (primaryRisk !== null && stayProbability !== null && Math.abs(primaryRisk - (1 - stayProbability)) > 0.00000001) {
            errors.push(`${item.date} 1주 이탈 확률과 next_week 현재 국면 잔류 확률이 일치하지 않습니다.`);
          }
          if (
            isObject(item.next_week)
            && isObject(transitionRisk["1w"])
            && item.next_week.date !== transitionRisk["1w"].target_end
          ) {
            errors.push(`${item.date} next_week.date와 transition_risk.1w.target_end가 일치하지 않습니다.`);
          }
        }
      }
      if (!isV5 && !isObject(item.scores)) {
        errors.push(`${item.date} scores 객체가 없습니다.`);
      } else if (!isV5) {
        for (const scoreName of Object.keys(FACTOR_META)) {
          const score = finiteNumber(item.scores[scoreName]);
          if (score === null || score < -1 || score > 1) {
            errors.push(`${item.date} scores.${scoreName}이 -1–1 범위가 아닙니다.`);
          }
        }
      }
      weekly.push(item);
    }

    if (
      isV4
      && isObject(payload.model.evidence_artifacts)
      && isObject(payload.model.evidence_artifacts.weekly_state_forecasts)
      && payload.model.evidence_artifacts.weekly_state_forecasts.row_count !== weekly.length
    ) {
      errors.push("v4 weekly_state_forecasts.row_count는 weekly 길이와 일치해야 합니다.");
    }

    for (let index = 0; index < payload.sources.length; index += 1) {
      const source = payload.sources[index];
      if (!isObject(source) || typeof source.status !== "string" || !(source.status in HEALTH_LABELS)) {
        errors.push(`sources[${index}].status가 지원하는 상태가 아닙니다.`);
      }
    }
    if (isV5) validateV5ResearchContract(payload.research, payload.model, errors);

    return { errors: [...new Set(errors)], warnings: [...new Set(warnings)], weekly };
  }

  function snapToPriorDate(dates, targetDate) {
    if (!Array.isArray(dates) || !dates.length || !isIsoDate(targetDate)) return -1;
    let low = 0;
    let high = dates.length - 1;
    let answer = -1;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      if (dates[middle] <= targetDate) {
        answer = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    return answer;
  }

  function getProbability(result, code) {
    return probability(isObject(result) && isObject(result.probabilities) ? result.probabilities[code] : null);
  }

  function resultVersion() {
    return isObject(state.raw && state.raw.meta) ? state.raw.meta.result_version : null;
  }

  function isV5Payload() {
    return resultVersion() === V5_RESULT_VERSION;
  }

  function currentMeasureKind(version = resultVersion()) {
    return version === V5_RESULT_VERSION ? "membership" : "probability";
  }

  function getCurrentMeasure(result, code, version = resultVersion()) {
    const field = currentMeasureKind(version) === "membership" ? "memberships" : "probabilities";
    return probability(isObject(result) && isObject(result[field]) ? result[field][code] : null);
  }

  function extractCurrentStrength(result, version = resultVersion()) {
    return probability(isObject(result) ? (version === V5_RESULT_VERSION ? result.primary_membership : result.confidence) : null);
  }

  function extractConfidence(result) {
    return probability(isObject(result) ? result.confidence : null);
  }

  function selectedWeek() {
    return state.weekly[state.selectedIndex] || null;
  }

  function forecastComparisonModels(payload = state.raw) {
    const model = isObject(payload) && isObject(payload.model) ? payload.model : null;
    const comparison = model && isObject(model.forecast_comparison)
      ? model.forecast_comparison
      : null;
    return comparison && Array.isArray(comparison.models)
      ? comparison.models.filter((name) => typeof name === "string" && name)
      : [];
  }

  function operatingChampionName(payload = state.raw) {
    const selection = isObject(payload) && isObject(payload.selection) ? payload.selection : {};
    const model = isObject(payload) && isObject(payload.model) ? payload.model : {};
    return modelName(selection.operating_champion || model.champion);
  }

  function forecastForWeek(week, requestedModel = state.comparisonModel, payload = state.raw) {
    if (!isObject(week)) return null;
    const official = isObject(week.next_week) ? week.next_week : null;
    const model = isObject(payload) && isObject(payload.model) ? payload.model : {};
    const championName = modelName(model.champion);
    const name = typeof requestedModel === "string" && requestedModel
      ? requestedModel
      : championName;
    const forecasts = Array.isArray(week.model_forecasts) ? week.model_forecasts : [];
    const selected = forecasts.find(
      (row) => isObject(row) && row.model === name,
    );
    if (selected) return selected;
    return official;
  }

  function oneWeekDepartureProbability(week, forecast = null) {
    if (!isObject(week)) return null;
    const currentState = isObject(week.current) ? week.current.state : null;
    const selected = isObject(forecast) ? forecast : forecastForWeek(week);
    if (STATE_ORDER.includes(currentState) && isObject(selected)) {
      const stay = getProbability(selected, currentState);
      if (stay !== null) return Math.max(0, Math.min(1, 1 - stay));
    }
    const risk = isObject(week.transition_risk) && isObject(week.transition_risk["1w"])
      ? week.transition_risk["1w"].probability
      : week.transition_probability;
    return probability(risk);
  }

  function historyMeasureForWeek(
    week,
    code,
    series = "observed",
    requestedModel = null,
    payload = null,
    version = V5_RESULT_VERSION,
  ) {
    if (series === "forecast") {
      return getProbability(
        forecastForWeek(week, requestedModel, payload),
        code,
      );
    }
    return getCurrentMeasure(isObject(week) ? week.current : null, code, version);
  }

  function historyStateForWeek(
    week,
    series = "observed",
    requestedModel = null,
    payload = null,
  ) {
    const result = series === "forecast"
      ? forecastForWeek(week, requestedModel, payload)
      : isObject(week)
        ? week.current
        : null;
    return isObject(result) && STATE_ORDER.includes(result.state) ? result.state : null;
  }

  function observedHistoryMeasure(week, code) {
    return historyMeasureForWeek(
      week,
      code,
      "observed",
      state.comparisonModel,
      state.raw,
      resultVersion(),
    );
  }

  function forecastHistoryMeasure(week, code) {
    return historyMeasureForWeek(
      week,
      code,
      "forecast",
      state.comparisonModel,
      state.raw,
      resultVersion(),
    );
  }

  function actualNextWeekForWeek(
    week,
    requestedModel = state.comparisonModel,
    payload = state.raw,
    weekly = null,
  ) {
    const forecast = forecastForWeek(week, requestedModel, payload);
    const targetDate = firstValue(forecast, ["date", "target_date", "period_end"]);
    const rows = Array.isArray(weekly)
      ? weekly
      : isObject(payload) && Array.isArray(payload.weekly)
        ? payload.weekly
        : state.weekly;
    const matched = typeof targetDate === "string"
      ? rows.find((row) => isObject(row) && row.date === targetDate)
      : null;
    const actualState = isObject(matched) && isObject(matched.current)
      && STATE_ORDER.includes(matched.current.state)
      ? matched.current.state
      : null;
    const latestDate = rows.reduce(
      (latest, row) => isObject(row) && typeof row.date === "string" && row.date > latest ? row.date : latest,
      "",
    );
    const status = actualState
      ? "available"
      : typeof targetDate !== "string"
        ? "unavailable"
        : latestDate && targetDate > latestDate
          ? "pending"
          : "missing";
    return { date: typeof targetDate === "string" ? targetDate : null, state: actualState, status };
  }

  function forecastEntropyForWeek(
    week,
    requestedModel = state.comparisonModel,
    payload = state.raw,
  ) {
    const forecast = forecastForWeek(week, requestedModel, payload);
    return probability(isObject(forecast) ? forecast.entropy : null);
  }

  function isHistoricalSelection() {
    if (state.selectedIndex < 0 || !state.weekly.length) return false;
    return state.selectedIndex !== state.weekly.length - 1;
  }

  function resolveHistoryWindow(availableCount, requestedWindow) {
    const available = Number.isInteger(availableCount) && availableCount > 0 ? availableCount : 0;
    if (requestedWindow === "all") return "all";
    const requested = Number(requestedWindow);
    return Number.isInteger(requested) && requested > 0 && requested <= available ? requested : "all";
  }

  function syncHistoryWindowControl() {
    const available = Math.max(0, state.selectedIndex + 1);
    const select = dom["history-window"];
    const requested = state.preferredHistoryWindow;
    const resolved = resolveHistoryWindow(available, requested);

    for (const option of select.options) {
      if (option.value === "all") {
        option.textContent = available ? `전체 · ${available}주` : "전체";
        option.disabled = false;
        continue;
      }
      const weeks = Number(option.value);
      option.textContent = `${weeks}주`;
      option.disabled = weeks > available;
    }

    select.value = String(resolved);
    select.setAttribute("aria-label", `표시 기간 · 사용 가능 ${available}주`);
    state.historyWindow = resolved;
  }

  function selectedHistory() {
    if (state.selectedIndex < 0) return [];
    const end = state.selectedIndex + 1;
    if (state.historyWindow === "all") return state.weekly.slice(0, end);
    const count = Number(state.historyWindow) || 52;
    return state.weekly.slice(Math.max(0, end - count), end);
  }

  function initializeDom() {
    const ids = [
      "app-state", "loading-state", "error-state", "empty-state", "error-title", "error-detail", "retry-button",
      "dashboard", "header-result-identity", "header-analysis-date", "header-data-as-of", "header-model-health", "theme-toggle",
      "theme-toggle-text", "dashboard-subtitle", "date-form", "analysis-date", "week-select",
      "snap-note", "previous-week", "next-week", "latest-week", "history-window",
      "hero-results", "contract-overview-grid", "label-spec-identity", "membership-definition",
      "forecast-window-card", "forecast-contract-status", "forecast-origin-at", "forecast-decision-at",
      "forecast-target-at", "forecast-remaining-horizon", "forecast-expired-notice",
      "current-regime-card", "current-horizon", "current-regime-symbol", "current-regime-name",
      "current-regime-confidence", "current-probabilities", "current-entropy", "next-regime-card", "next-horizon",
      "next-regime-symbol", "next-regime-name", "next-regime-confidence",
      "next-probabilities", "next-entropy", "next-model-context-detail",
      "transition-card", "transition-value", "transition-value-label", "transition-meter",
      "transition-horizon-bars", "transition-risk-detail", "probability-chart",
      "probability-chart-wrap", "chart-tooltip", "history-caption",
      "chart-selection-readout", "chart-interaction-hint",
      "chart-readout-date", "chart-readout-target-date", "chart-readout-observed-label", "chart-readout-risk-on", "chart-readout-transition", "chart-readout-risk-off",
      "chart-readout-forecast-label", "chart-readout-forecast-risk-on", "chart-readout-forecast-transition",
      "chart-readout-forecast-risk-off", "chart-readout-predicted", "chart-readout-actual", "chart-readout-entropy",
      "history-data-body", "history-observed-group-label", "history-forecast-group-label", "probability-chart-title", "history-chart-legend",
      "history-table-scroll", "history-table-caption",
      "factor-title", "factor-axis", "factor-caption", "factor-scores", "timeline-title", "regime-timeline", "timeline-start",
      "timeline-end", "drivers-title", "drivers-caption", "top-drivers", "market-context",
      "duration-context-card", "duration-context-caption", "duration-context", "duration-baselines", "duration-research-detail",
      "fx-context-card", "fx-context-caption", "fx-coverage", "fx-ablation-status", "fx-context", "fx-context-detail",
      "conditional-stats-nav", "conditional-stats", "conditional-stats-caption", "conditional-basis-field", "conditional-basis-select",
      "conditional-asset-select", "conditional-horizon-select", "conditional-comparison-caption", "conditional-stat-grid",
      "conditional-stat-scroll", "conditional-stat-table-caption", "conditional-stat-body",
      "champion-summary", "model-evidence-summary", "model-caption", "model-loss-caption",
      "model-loss-chart", "model-loss-axis", "leaderboard-body",
      "model-role-grid", "operating-model-name", "operating-model-status", "frozen-baseline-name",
      "research-selector-summary", "research-selector-detail", "research-selection-status", "research-selection-detail",
      "model-forecast-field", "model-forecast-select", "model-forecast-explorer",
      "model-forecast-role", "model-forecast-title", "model-forecast-caption",
      "model-forecast-symbol", "model-forecast-state", "model-forecast-confidence",
      "model-forecast-probabilities", "model-forecast-rank", "model-forecast-log-loss",
      "model-forecast-brier", "model-forecast-calibration",
      "transition-model-section", "transition-model-caption", "transition-horizon-select",
      "transition-model-summary", "transition-leaderboard-caption", "transition-leaderboard-body",
      "research-evidence", "research-notice-summary",
      "source-freshness", "source-health-body", "feature-catalog", "footer-model-version", "footer-schema-version",
      "footer-generated-at", "screen-reader-status",
    ];
    for (const id of ids) dom[id] = document.getElementById(id);
  }

  function revealActiveProjectLink() {
    const links = document.querySelector(".site-nav-links");
    const active = links && links.querySelector('[aria-current="page"]');
    if (!links || !active || links.scrollWidth <= links.clientWidth) return;

    const linksRect = links.getBoundingClientRect();
    const activeRect = active.getBoundingClientRect();
    const edgePadding = 1;
    if (activeRect.left < linksRect.left) {
      links.scrollLeft -= linksRect.left - activeRect.left + edgePadding;
    } else if (activeRect.right > linksRect.right) {
      links.scrollLeft += activeRect.right - linksRect.right + edgePadding;
    }
  }

  function showAppState(kind, title, detail) {
    dom["app-state"].hidden = false;
    dom.dashboard.hidden = true;
    dom["loading-state"].hidden = kind !== "loading";
    dom["error-state"].hidden = kind !== "error";
    dom["empty-state"].hidden = kind !== "empty";
    dom["app-state"].setAttribute("aria-busy", kind === "loading" ? "true" : "false");
    if (kind === "error") {
      setText(dom["error-title"], title || "데이터를 표시할 수 없습니다");
      setText(dom["error-detail"], detail || "알 수 없는 오류가 발생했습니다.");
    }
  }

  function showDashboard() {
    dom["app-state"].hidden = true;
    dom["app-state"].setAttribute("aria-busy", "false");
    dom.dashboard.hidden = false;
  }

  function setupTheme() {
    let requested = null;
    try {
      requested = new URLSearchParams(window.location.search).get("theme");
    } catch (_error) {
      requested = null;
    }
    let stored = null;
    try {
      stored = localStorage.getItem(THEME_STORAGE_KEY);
      if (stored !== "dark" && stored !== "light") {
        for (const key of LEGACY_THEME_STORAGE_KEYS) {
          const legacy = localStorage.getItem(key);
          if (legacy === "dark" || legacy === "light") {
            stored = legacy;
            localStorage.setItem(THEME_STORAGE_KEY, legacy);
            break;
          }
        }
      }
    } catch (_error) {
      stored = null;
    }
    const preferredDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const theme = requested === "dark" || requested === "light"
      ? requested
      : stored === "dark" || stored === "light"
        ? stored
        : preferredDark
          ? "dark"
          : "light";
    applyTheme(theme);
  }

  function applyTheme(theme) {
    const next = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    dom["theme-toggle"].setAttribute("aria-pressed", next === "dark" ? "true" : "false");
    dom["theme-toggle"].setAttribute("aria-label", next === "dark" ? "라이트 모드로 전환" : "다크 모드로 전환");
    setText(dom["theme-toggle-text"], next === "dark" ? "라이트 모드" : "다크 모드");
  }

  function toggleTheme() {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch (_error) {
      // Theme still applies for the current session when storage is unavailable.
    }
  }

  function setSnapNote(message = DEFAULT_SNAP_NOTE, visible = false) {
    dom["snap-note"].textContent = message;
    dom["snap-note"].classList.toggle("sr-only", !visible);
    dom["snap-note"].classList.toggle("is-visible", visible);
  }

  function bindEvents() {
    dom["theme-toggle"].addEventListener("click", toggleTheme);
    dom["retry-button"].addEventListener("click", loadData);
    dom["date-form"].addEventListener("submit", (event) => event.preventDefault());
    dom["analysis-date"].addEventListener("change", () => {
      const requested = dom["analysis-date"].value;
      const dates = state.weekly.map((item) => item.date);
      let index = snapToPriorDate(dates, requested);
      if (index < 0 && dates.length) {
        index = 0;
        setSnapNote(`최초 관측일 ${dates[0]}로 이동했습니다.`, true);
      } else if (index >= 0 && dates[index] !== requested) {
        setSnapNote(`${dates[index]} 관측 주로 이동했습니다.`, true);
      } else {
        setSnapNote();
      }
      if (index >= 0) selectWeek(index, true, true);
    });
    dom["week-select"].addEventListener("change", () => {
      const index = state.weekly.findIndex((item) => item.date === dom["week-select"].value);
      if (index >= 0) {
        setSnapNote();
        selectWeek(index, true);
      }
    });
    dom["previous-week"].addEventListener("click", () => {
      if (state.selectedIndex > 0) selectWeek(state.selectedIndex - 1, true);
    });
    dom["next-week"].addEventListener("click", () => {
      if (state.selectedIndex < state.weekly.length - 1) selectWeek(state.selectedIndex + 1, true);
    });
    dom["latest-week"].addEventListener("click", () => {
      if (state.weekly.length) selectWeek(state.weekly.length - 1, true);
    });
    dom["history-window"].addEventListener("change", () => {
      state.preferredHistoryWindow = dom["history-window"].value === "all"
        ? "all"
        : Number(dom["history-window"].value);
      syncHistoryWindowControl();
      renderHistory();
      renderTimeline();
    });
    dom["transition-horizon-select"].addEventListener("change", () => {
      const requested = Number(dom["transition-horizon-select"].value);
      state.transitionHorizon = TRANSITION_HORIZONS.includes(requested) ? requested : 1;
      renderTransitionModels();
    });
    dom["model-forecast-select"].addEventListener("change", () => {
      const requested = dom["model-forecast-select"].value;
      const models = forecastComparisonModels(state.raw);
      if (!models.includes(requested)) return;
      state.comparisonModel = requested;
      renderForecastSurfaces();
      dom["screen-reader-status"].textContent = `${modelForecastLabel(requested)} 1주 예측으로 변경했습니다.`;
    });
    if (dom["conditional-basis-select"]) {
      dom["conditional-basis-select"].addEventListener("change", () => {
        const requested = dom["conditional-basis-select"].value;
        state.outcomeBasis = requested === "forecast"
          && modelConditionedAssetRowsComplete(state.raw, state.comparisonModel)
          ? "forecast"
          : "observed";
        syncConditionalBasisControl();
        renderConditionalComparison();
        renderConditionalDetail();
        dom["screen-reader-status"].textContent = `${conditionalBasisLabel()} 자산 성과로 변경했습니다.`;
      });
    }
    dom["conditional-asset-select"].addEventListener("change", () => {
      state.outcomeAsset = OUTCOME_ASSETS.includes(dom["conditional-asset-select"].value)
        ? dom["conditional-asset-select"].value
        : "SPY";
      renderConditionalDetail();
      dom["screen-reader-status"].textContent = `${state.outcomeAsset} ${OUTCOME_ASSET_LABELS[state.outcomeAsset]} 상세 성과 표로 변경했습니다.`;
    });
    dom["conditional-horizon-select"].addEventListener("change", () => {
      const requested = Number(dom["conditional-horizon-select"].value);
      state.outcomeHorizon = TRANSITION_HORIZONS.includes(requested) ? requested : 13;
      renderConditionalComparison();
      renderConditionalDetail();
      dom["screen-reader-status"].textContent = `${state.outcomeHorizon}주 보유 자산군 평균 수익률과 상세 성과 표를 갱신했습니다.`;
    });
    dom["probability-chart"].addEventListener("pointermove", (event) => previewChartDateFromPointer(event, false));
    dom["probability-chart"].addEventListener("click", (event) => previewChartDateFromPointer(event, true));
    dom["probability-chart-wrap"].addEventListener("pointerleave", resetChartPreview);
    dom["probability-chart-wrap"].addEventListener("pointercancel", resetChartPreview);
    dom["probability-chart-wrap"].addEventListener("keydown", handleChartKeydown);
    dom["probability-chart-wrap"].addEventListener("blur", resetChartPreview);
  }

  function populateDateControls() {
    const dates = state.weekly.map((item) => item.date);
    dom["week-select"].replaceChildren();
    for (const date of [...dates].reverse()) {
      const option = createElement("option", null, formatDate(date));
      option.value = date;
      dom["week-select"].append(option);
    }
    dom["analysis-date"].min = dates[0];
    dom["analysis-date"].max = dates[dates.length - 1];
  }

  function selectWeek(index, announce = false, preserveSnapNote = false) {
    if (!Number.isInteger(index) || index < 0 || index >= state.weekly.length) return;
    state.selectedIndex = index;
    syncHistoryWindowControl();
    const week = selectedWeek();
    if (!preserveSnapNote) {
      setSnapNote();
    }
    state.chartPinnedDate = week.date;
    state.chartPreviewDate = null;
    dom["analysis-date"].value = week.date;
    dom["week-select"].value = week.date;
    dom["previous-week"].disabled = index === 0;
    dom["next-week"].disabled = index === state.weekly.length - 1;
    renderSelectedWeek();
    if (announce) {
      dom["screen-reader-status"].textContent = `${formatDate(week.date)} 결과로 이동했습니다. 현재 국면은 ${stateMeta(week.current && week.current.state).ko}입니다.`;
    }
  }

  function renderNextForecastSurface(week) {
    const suppressed = suppressCurrentForecastSurface();
    dom["next-regime-card"].hidden = suppressed;
    if (suppressed) {
      renderNextModelContext(null);
      return null;
    }
    const forecast = forecastForWeek(week, state.comparisonModel);
    const forecastDate = firstValue(forecast, ["date", "target_date", "period_end"]);
    renderRegime("next", forecast, forecastDate);
    const selectedModel = textValue(isObject(forecast) ? forecast.model : null, "");
    if (selectedModel && dom["next-horizon"]) {
      dom["next-horizon"].textContent += ` · ${modelForecastLabel(selectedModel)}`;
    }
    if (dom["next-regime-card"]) {
      const championName = operatingChampionName();
      const roleLabel = selectedForecastIsHistorical()
        ? "과거 OOS"
        : selectedModel === championName
          ? "공식"
          : "연구";
      dom["next-regime-card"].classList.toggle(
        "is-comparison-model",
        Boolean(selectedModel) && selectedModel !== championName,
      );
      dom["next-regime-card"].dataset.forecastModel = selectedModel;
      dom["next-regime-card"].setAttribute(
        "aria-label",
        `${modelForecastLabel(selectedModel)} ${roleLabel} 1주 예측 · ${stateMeta(isObject(forecast) ? forecast.state : null).ko}`,
      );
    }
    renderNextModelContext(forecast);
    return forecast;
  }

  function renderForecastSurfaces() {
    const week = selectedWeek();
    if (!week) return;
    const forecast = renderNextForecastSurface(week);
    renderTransition(week, forecast);
    renderModelForecast();
    renderSemanticLabels();
    renderHistory();
    const outcomeWasForecast = state.outcomeBasis === "forecast";
    syncConditionalBasisControl();
    if (outcomeWasForecast || state.outcomeBasis === "forecast") {
      renderConditionalComparison();
      renderConditionalDetail();
    }
    applyExpiredForecastDomState(
      dom,
      forecastSurfacePolicy(state.raw, state.selectedIndex, state.weekly.length),
    );
  }

  function renderSelectedWeek() {
    const week = selectedWeek();
    if (!week) return;

    const cutoff = firstValue(week, ["data_as_of", "available_at", "cutoff_at"]) ||
      firstValue(state.raw.meta, ["data_as_of", "dataAsOf", "cutoff_at"]);
    setText(dom["header-analysis-date"], formatDate(week.date));
    renderHeaderDataAsOf(cutoff);
    dom["dashboard-subtitle"].textContent = `${formatDate(week.date)} 관측 주${cutoff ? ` · 컷오프 ${formatDateTime(cutoff)}` : ""}`;
    renderContractOverview();

    renderRegime("current", week.current, week.date);
    const forecast = renderNextForecastSurface(week);
    renderTransition(week, forecast);
    renderSemanticLabels();
    renderHistory();
    renderTimeline();
    renderFactors(isV5Payload() ? week.context_scores : week.scores);
    if (isV5Payload()) renderContextExtremes(week.extreme_context);
    else renderDrivers(week.top_drivers);
    renderMarket(week.market);
    renderDurationContext(week.duration_context);
    renderFxContext(week.fx_context);
    renderModelForecast();
    applyExpiredForecastDomState(
      dom,
      forecastSurfacePolicy(state.raw, state.selectedIndex, state.weekly.length),
    );
  }

  function historyComparisonMeta() {
    const membership = isV5Payload();
    const model = modelForecastLabel(state.comparisonModel);
    const lineStyles = { risk_on: "실선", transition: "파선", risk_off: "점선" };
    return {
      title: membership ? "관측 소속도와 1주 예측확률" : "관측 확률과 1주 예측확률",
      observedMeasure: membership ? "관측 소속도" : "관측 확률",
      model,
      legendLabel: `상하 패널의 ${STATE_ORDER.map((code) => `${stateMeta(code).label} ${lineStyles[code]}`).join(", ")} 범례`,
      tableLabel: `${model} ${membership ? "관측 소속도" : "관측 확률"}와 다음 주 예측확률 및 실제 결과 표`,
      tableCaption: `${model} ${membership ? "관측 소속도" : "관측 확률"}·1주 예측확률·실제 다음 주 결과·정규화 예측 엔트로피`,
      timelineTitle: "관측 국면 타임라인",
      timelineLabel: "주간 관측 국면",
    };
  }

  function renderSemanticLabels() {
    const membership = isV5Payload();
    const historyMeta = historyComparisonMeta();
    setText(dom["probability-chart-title"], historyMeta.title);
    setText(dom["factor-title"], membership ? "시장 맥락 점수" : "국면 팩터");
    setText(dom["factor-caption"], membership ? "52주 표준화 기반 합성점수" : "52주 표준점수");
    setText(dom["drivers-title"], membership ? "52주 극단값" : "주요 지표");
    setText(dom["drivers-caption"], membership ? "52주 표준점수 · 시장 맥락" : "52주 표준점수");
    dom["history-chart-legend"].setAttribute("aria-label", historyMeta.legendLabel);
    dom["history-table-scroll"].setAttribute("aria-label", historyMeta.tableLabel);
    dom["probability-chart-wrap"].setAttribute("aria-label", historyMeta.tableLabel);
    setText(dom["history-table-caption"], historyMeta.tableCaption);
    setText(dom["history-observed-group-label"], `${historyMeta.observedMeasure} · t`);
    setText(dom["chart-readout-observed-label"], `${historyMeta.observedMeasure} · t`);
    setText(dom["history-forecast-group-label"], `${historyMeta.model} 예측확률 · t→t+1`);
    setText(dom["chart-readout-forecast-label"], `${historyMeta.model} 예측확률 · t→t+1`);
    setText(dom["timeline-title"], historyMeta.timelineTitle);
    dom["regime-timeline"].setAttribute("aria-label", historyMeta.timelineLabel);
  }

  function renderRegime(prefix, result, horizonDate) {
    const card = dom[`${prefix}-regime-card`];
    const code = isObject(result) ? result.state : null;
    const meta = stateMeta(code);
    card.classList.remove("state-risk_on", "state-transition", "state-risk_off");
    if (STATE_ORDER.includes(code)) card.classList.add(`state-${code}`);

    setText(dom[`${prefix}-regime-symbol`], meta.symbol);
    setText(dom[`${prefix}-regime-name`], `${meta.ko} · ${meta.label}`);
    const isCurrent = prefix === "current";
    const semantic = isCurrent && isV5Payload() ? "소속도" : isCurrent ? "확률" : "예측확률";
    const confidence = isCurrent ? extractCurrentStrength(result) : extractConfidence(result);
    setText(dom[`${prefix}-regime-confidence`], `${semantic} ${formatPercent(confidence)}`);
    dom[`${prefix}-horizon`].textContent = `${prefix === "current" ? "t" : "t+1"}${horizonDate ? ` · ${formatDate(horizonDate, false)}` : ""}`;
    const probabilityContainer = dom[`${prefix}-probabilities`];
    probabilityContainer.replaceChildren();
    for (const stateCode of STATE_ORDER) {
      const stateDefinition = stateMeta(stateCode);
      const value = isCurrent ? getCurrentMeasure(result, stateCode) : getProbability(result, stateCode);
      const row = createElement("div", "probability-row");
      const label = createElement("span", "probability-label");
      const marker = createElement("span", `state-dot ${stateCode}`, stateDefinition.short);
      marker.setAttribute("aria-hidden", "true");
      label.append(marker, document.createTextNode(stateDefinition.label));
      const track = createElement("span", "probability-track");
      const fill = createElement("span", `probability-fill ${stateCode}`);
      fill.style.width = value === null ? "0" : `${(value * 100).toFixed(2)}%`;
      track.append(fill);
      const display = createElement("span", "probability-value", formatPercent(value));
      row.setAttribute("aria-label", `${stateDefinition.ko} ${semantic} ${formatPercent(value)}`);
      row.append(label, track, display);
      probabilityContainer.append(row);
    }

    const entropyField = isCurrent && isV5Payload() ? "membership_entropy" : "entropy";
    const entropy = finiteNumber(isObject(result) ? result[entropyField] : null);
    const entropyLabel = isCurrent && isV5Payload() ? "소속도" : isCurrent ? "불확실성" : "예측 불확실성";
    setText(dom[`${prefix}-entropy`], `${entropyLabel} 엔트로피 ${formatNumber(entropy, 3)}`);
  }

  function renderNextModelContext(result) {
    const model = textValue(isObject(result) ? result.model : null, "미기재");
    const details = [`모델 ${model}`];
    const inputLabel = {
      markov: "입력 현재 국면·과거 전이",
      persistence: "입력 현재 국면",
      xgboost: "입력 PIT 구조 피처",
      xgb_hazard_destination: "입력 PIT 피처·이탈/목적지",
      causal_dynamic_ensemble: "입력 완료 OOS 예측 풀",
      causal_multiscale_ensemble: "입력 완료 OOS 예측 풀 26·52·104주",
      pca_ridge_logistic: "입력 PIT 피처 · fold 내부 PCA",
      recency_weighted_xgboost_208w: "입력 PIT 피처 · 208주 최근 가중",
      recency_weighted_ridge_logistic_208w: "입력 PIT 피처 · 208주 최근 가중",
      discounted_markov_208w: "입력 현재 국면 · 최근 전이 가중",
    }[model];
    if (inputLabel) details.push(inputLabel);
    if (isObject(result) && result.fallback === true) details.push("fallback");
    const health = state.raw && isObject(state.raw.model)
      ? state.raw.model.model_health
      : null;
    const championName = operatingChampionName();
    if (isV5Payload() && model && model !== championName) {
      details.push("비교 예측");
    } else if (isV5Payload() && isObject(health)) {
      details.push(health.status === "review_due" ? "검토 필요" : "정상");
      details.push(...modelHealthReasonLabels(health.reasons));
    }
    setText(dom["next-model-context-detail"], details.join(" · "));
  }

  function renderTransition(week, forecast = null) {
    const suppressed = suppressCurrentForecastSurface();
    dom["transition-card"].hidden = suppressed;
    if (suppressed) {
      dom["transition-horizon-bars"].replaceChildren();
      dom["transition-risk-detail"].replaceChildren();
      return;
    }
    const riskByHorizon = isObject(week.transition_risk) ? week.transition_risk : null;
    const selectedForecast = isObject(forecast) ? forecast : forecastForWeek(week);
    const value = oneWeekDepartureProbability(week, selectedForecast);
    const selectedModel = textValue(
      isObject(selectedForecast) ? selectedForecast.model : null,
      "",
    );
    const selectableForecast = isV5Payload()
      && Array.isArray(week.model_forecasts)
      && Boolean(selectedModel);
    setText(dom["transition-value"], formatPercent(value));
    setText(
      dom["transition-value-label"],
      selectableForecast
        ? `1주 이탈 · ${modelForecastLabel(selectedModel)}`
        : riskByHorizon
          ? "1주 이탈 확률"
        : "다음 주 국면 변경 확률",
    );
    const fill = dom["transition-meter"].querySelector("span");
    dom["transition-meter"].setAttribute(
      "aria-label",
      selectableForecast
        ? `${modelForecastLabel(selectedModel)} 기준 다음 주 현재 국면 이탈 확률`
        : riskByHorizon
          ? "향후 1주 안에 한 번 이상 현재 국면에서 이탈할 확률"
          : "다음 주 국면 변경 확률",
    );
    fill.style.width = value === null ? "0" : `${(value * 100).toFixed(2)}%`;
    if (value === null) {
      dom["transition-meter"].removeAttribute("aria-valuenow");
      dom["transition-meter"].setAttribute("aria-valuetext", "국면 변경 확률 결과 없음");
    } else {
      dom["transition-meter"].setAttribute("aria-valuenow", String(Math.round(value * 100)));
      dom["transition-meter"].setAttribute("aria-valuetext", formatPercent(value));
    }
    renderTransitionHorizons(week);
  }

  function renderTransitionHorizons(week) {
    const container = dom["transition-horizon-bars"];
    const researchDetail = dom["transition-risk-detail"];
    container.replaceChildren();
    researchDetail.replaceChildren();
    const riskByHorizon = isObject(week.transition_risk) ? week.transition_risk : null;
    if (!isObject(riskByHorizon)) {
      container.hidden = true;
      return;
    }

    for (const horizon of [4, 13]) {
      const result = riskByHorizon[`${horizon}w`];
      const value = probability(isObject(result) ? result.probability : null);
      const row = createElement("div", "transition-horizon-row");
      const heading = createElement("div", "transition-horizon-heading");
      heading.append(
        createElement("span", null, `${horizon}주 이내 · 별도 이탈 모델`),
        createElement("strong", null, formatPercent(value)),
      );
      const meter = createElement("span", "transition-horizon-meter");
      const meterFill = createElement("span");
      meterFill.style.width = value === null ? "0" : `${(value * 100).toFixed(2)}%`;
      meter.append(meterFill);
      row.append(heading, meter);
      if (isV5Payload()) {
        const detail = createElement("div", "transition-horizon-detail");
        const kmDeparture = probability(
          week.duration_context
          && week.duration_context.departure_probability
          && week.duration_context.departure_probability[`${horizon}w`],
        );
        const difference = value === null || kmDeparture === null ? null : (value - kmDeparture) * 100;
        const comparison = createElement("div", "transition-horizon-comparison");
        comparison.append(
          createElement("span", null, `KM 기준 ${formatPercent(kmDeparture)}`),
          createElement("strong", null, difference === null ? "차이 —" : `${difference > 0 ? "+" : ""}${formatNumber(difference, 1)}%p`),
        );
        detail.append(comparison);

        const directional = isObject(week.directional_risk) ? week.directional_risk[`${horizon}w`] : null;
        const masses = createElement("div", "direction-mass-list");
        for (const code of STATE_ORDER) {
          if (code === week.current.state) continue;
          const mass = probability(directional && directional.first_destination ? directional.first_destination[code] : null);
          if (mass === null || mass === 0) continue;
          const item = createElement("span", "direction-mass");
          const definition = stateMeta(code);
          const marker = createElement("span", `state-dot ${code}`, definition.short);
          marker.setAttribute("aria-hidden", "true");
          item.append(marker, document.createTextNode(`${definition.label} ${formatPercent(mass)}`));
          masses.append(item);
        }
        if (masses.childElementCount) detail.append(masses);
        const researchItem = createElement("div", "research-transition-item");
        researchItem.append(createElement("strong", null, `${horizon}주 이내`), detail);
        researchItem.title = directional && directional.model ? `최초 이탈 방향 모델 · ${directional.model}` : "";
        researchDetail.append(researchItem);
      }
      row.setAttribute(
        "aria-label",
        `향후 ${horizon}주 안에 한 번 이상 현재 국면에서 이탈할 확률 ${formatPercent(value)}`,
      );
      container.append(row);
    }
    container.hidden = false;
  }

  function makeLinePath(points) {
    let path = "";
    let open = false;
    for (const point of points) {
      if (point.value === null) {
        open = false;
        continue;
      }
      path += `${open ? "L" : "M"}${point.x.toFixed(2)},${point.y.toFixed(2)} `;
      open = true;
    }
    return path.trim();
  }

  function chartIndexForDate(date) {
    return state.chartHistory.findIndex((week) => week.date === date);
  }

  function chartX(index, count) {
    const { width, margin } = CHART_DIMENSIONS;
    const plotWidth = width - margin.left - margin.right;
    return margin.left + (count === 1 ? plotWidth / 2 : (index / (count - 1)) * plotWidth);
  }

  function chartPanelLayout() {
    const { height, margin, panelGap, outcomeOffset } = CHART_DIMENSIONS;
    const panelHeight = (height - margin.top - margin.bottom - panelGap) / 2;
    const observedTop = margin.top;
    const forecastTop = observedTop + panelHeight + panelGap;
    return {
      panelHeight,
      observedTop,
      forecastTop,
      outcomeY: forecastTop + panelHeight + outcomeOffset,
    };
  }

  function updateChartCursor(index) {
    const cursor = dom["probability-chart"].querySelector('[data-chart-cursor="true"]');
    if (!cursor || index < 0 || index >= state.chartHistory.length) return;
    const cursorX = chartX(index, state.chartHistory.length);
    cursor.setAttribute("x1", cursorX.toFixed(2));
    cursor.setAttribute("x2", cursorX.toFixed(2));
  }

  function scrollChartDateIntoView(date, behavior = "auto") {
    const index = chartIndexForDate(date);
    const wrap = dom["probability-chart-wrap"];
    const svg = dom["probability-chart"];
    if (!wrap || !svg || index < 0 || wrap.scrollWidth <= wrap.clientWidth) return;
    const cursorX = chartX(index, state.chartHistory.length);
    const scaledX = (cursorX / CHART_DIMENSIONS.width) * svg.scrollWidth;
    const maxLeft = Math.max(0, wrap.scrollWidth - wrap.clientWidth);
    const targetLeft = Math.max(0, Math.min(maxLeft, scaledX - wrap.clientWidth / 2));
    wrap.scrollTo({ left: targetLeft, behavior });
  }

  function renderChartReadout(date) {
    const index = chartIndexForDate(date);
    const week = index >= 0 ? state.chartHistory[index] : null;
    const historyMeta = historyComparisonMeta();
    const forecast = week ? forecastForWeek(week, state.comparisonModel, state.raw) : null;
    const actual = week ? actualNextWeekForWeek(week) : { date: null, state: null, status: "unavailable" };
    const predictedState = isObject(forecast) && STATE_ORDER.includes(forecast.state) ? forecast.state : null;
    const entropy = week ? forecastEntropyForWeek(week) : null;
    setText(dom["chart-readout-date"], week ? formatDate(week.date) : "—");
    setText(
      dom["chart-readout-target-date"],
      actual.date ? `예측 대상 ${formatDate(actual.date, false)}` : "예측 대상 —",
    );
    for (const code of STATE_ORDER) {
      const readout = dom[`chart-readout-${code.replaceAll("_", "-")}`];
      if (readout) setText(readout, week ? formatPercent(observedHistoryMeasure(week, code)) : "—");
      const forecastReadout = dom[`chart-readout-forecast-${code.replaceAll("_", "-")}`];
      if (forecastReadout) setText(forecastReadout, week ? formatPercent(forecastHistoryMeasure(week, code)) : "—");
    }
    setText(dom["chart-readout-predicted"], predictedState ? stateMeta(predictedState).label : "—");
    const actualText = actual.status === "available"
      ? `${stateMeta(actual.state).label}${predictedState ? ` · ${actual.state === predictedState ? "일치" : "불일치"}` : ""}`
      : actual.status === "pending"
        ? "결과 대기"
        : actual.status === "missing"
          ? "결과 없음"
          : "—";
    setText(dom["chart-readout-actual"], actualText);
    setText(dom["chart-readout-entropy"], formatNumber(entropy, 3));
    dom["chart-readout-entropy"].title = "0은 한 국면에 집중, 1은 세 국면에 균등";
    dom["chart-readout-actual"].classList.remove("is-match", "is-miss", "is-pending");
    if (actual.status === "available" && predictedState) {
      dom["chart-readout-actual"].classList.add(actual.state === predictedState ? "is-match" : "is-miss");
    } else if (actual.status === "pending") {
      dom["chart-readout-actual"].classList.add("is-pending");
    }
    if (week) updateChartCursor(index);
    for (const point of dom["probability-chart"].querySelectorAll(".chart-point")) {
      point.classList.toggle("is-active", Boolean(week) && point.dataset.date === week.date);
    }
    for (const marker of dom["probability-chart"].querySelectorAll(".actual-outcome-marker")) {
      marker.classList.toggle("is-active", Boolean(week) && marker.dataset.date === week.date);
    }

    if (dom["chart-selection-readout"]) {
      const observedVector = STATE_ORDER
        .map((code) => `${stateMeta(code).label} ${formatPercent(observedHistoryMeasure(week, code))}`)
        .join(", ");
      const forecastVector = STATE_ORDER
        .map((code) => `${stateMeta(code).label} ${formatPercent(forecastHistoryMeasure(week, code))}`)
        .join(", ");
      const summary = week
        ? `${formatDate(week.date)} 관측 기준. ${historyMeta.observedMeasure}: ${observedVector}. ${historyMeta.model} 1주 예측확률: ${forecastVector}. 실제 다음 주 ${actualText}. 정규화 예측 엔트로피 ${formatNumber(entropy, 3)}.`
        : "선택된 차트 날짜가 없습니다.";
      dom["chart-selection-readout"].setAttribute("aria-label", summary);
    }
  }

  function showChartTooltipForWeek(event, week) {
    const tooltip = dom["chart-tooltip"];
    const forecast = forecastForWeek(week, state.comparisonModel, state.raw);
    const actual = actualNextWeekForWeek(week);
    const predictedState = isObject(forecast) && STATE_ORDER.includes(forecast.state) ? forecast.state : null;
    const actualText = actual.status === "available"
      ? `${stateMeta(actual.state).label}${predictedState ? ` (${actual.state === predictedState ? "일치" : "불일치"})` : ""}`
      : actual.status === "pending"
        ? "결과 대기"
        : actual.status === "missing"
          ? "결과 없음"
          : "—";
    tooltip.textContent = `${formatDate(week.date)} 관측\n관측 · ${STATE_ORDER.map((code) => `${stateMeta(code).label} ${formatPercent(observedHistoryMeasure(week, code))}`).join(" · ")}\n${modelForecastLabel(state.comparisonModel)} 예측 · ${STATE_ORDER.map((code) => `${stateMeta(code).label} ${formatPercent(forecastHistoryMeasure(week, code))}`).join(" · ")}\n실제 t+1 ${actualText} · 엔트로피 ${formatNumber(forecastEntropyForWeek(week), 3)}`;
    tooltip.hidden = false;
    const wrapRect = dom["probability-chart-wrap"].getBoundingClientRect();
    const left = finiteNumber(event.clientX) === null ? 8 : event.clientX - wrapRect.left + 10;
    const top = finiteNumber(event.clientY) === null ? 8 : event.clientY - wrapRect.top - 42;
    const maxLeft = Math.max(8, wrapRect.width - 380);
    tooltip.style.left = `${Math.max(8, Math.min(left, maxLeft))}px`;
    tooltip.style.top = `${Math.max(4, top)}px`;
  }

  function previewChartDateFromPointer(event, pin) {
    if (!state.chartHistory.length) return;
    const rect = dom["probability-chart"].getBoundingClientRect();
    if (!rect.width || finiteNumber(event.clientX) === null) return;
    const { width, margin } = CHART_DIMENSIONS;
    const plotWidth = width - margin.left - margin.right;
    const viewBoxX = ((event.clientX - rect.left) / rect.width) * width;
    const ratio = Math.max(0, Math.min(1, (viewBoxX - margin.left) / plotWidth));
    const index = state.chartHistory.length === 1 ? 0 : Math.round(ratio * (state.chartHistory.length - 1));
    const week = state.chartHistory[index];
    state.chartPreviewDate = week.date;
    if (pin) {
      state.chartPinnedDate = week.date;
      state.chartPreviewDate = null;
      dom["probability-chart-wrap"].focus({ preventScroll: true });
    }
    renderChartReadout(week.date);
    showChartTooltipForWeek(event, week);
  }

  function resetChartPreview() {
    state.chartPreviewDate = null;
    const pinnedIndex = chartIndexForDate(state.chartPinnedDate);
    const fallback = pinnedIndex >= 0
      ? state.chartPinnedDate
      : state.chartHistory.length
        ? state.chartHistory[state.chartHistory.length - 1].date
        : null;
    renderChartReadout(fallback);
    hideChartTooltip();
  }

  function handleChartKeydown(event) {
    if (!state.chartHistory.length) return;
    const keys = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "Escape"]);
    if (!keys.has(event.key)) return;
    event.preventDefault();

    if (event.key === "Escape") {
      const selected = selectedWeek();
      state.chartPinnedDate = selected && chartIndexForDate(selected.date) >= 0
        ? selected.date
        : state.chartHistory[state.chartHistory.length - 1].date;
      state.chartPreviewDate = null;
      renderChartReadout(state.chartPinnedDate);
      scrollChartDateIntoView(state.chartPinnedDate);
      hideChartTooltip();
      return;
    }

    const activeDate = state.chartPreviewDate || state.chartPinnedDate;
    let index = chartIndexForDate(activeDate);
    if (index < 0) index = state.chartHistory.length - 1;
    if (event.key === "Home") index = 0;
    if (event.key === "End") index = state.chartHistory.length - 1;
    if (event.key === "ArrowLeft" || event.key === "ArrowUp") index = Math.max(0, index - 1);
    if (event.key === "ArrowRight" || event.key === "ArrowDown") index = Math.min(state.chartHistory.length - 1, index + 1);
    state.chartPinnedDate = state.chartHistory[index].date;
    state.chartPreviewDate = null;
    renderChartReadout(state.chartPinnedDate);
    scrollChartDateIntoView(state.chartPinnedDate);
    hideChartTooltip();
  }

  function createActualOutcomeMarker(actual, x, y, date) {
    const common = {
      class: `actual-outcome-marker ${actual.state || actual.status}`,
      "data-date": date,
      focusable: "false",
      "aria-hidden": "true",
    };
    if (actual.state === "transition") {
      return createSvg("path", {
        ...common,
        d: `M${x.toFixed(2)},${(y - 4.5).toFixed(2)} L${(x + 4.5).toFixed(2)},${y.toFixed(2)} L${x.toFixed(2)},${(y + 4.5).toFixed(2)} L${(x - 4.5).toFixed(2)},${y.toFixed(2)} Z`,
      });
    }
    if (actual.state === "risk_off") {
      return createSvg("path", {
        ...common,
        d: `M${(x - 4.8).toFixed(2)},${(y - 3.6).toFixed(2)} L${(x + 4.8).toFixed(2)},${(y - 3.6).toFixed(2)} L${x.toFixed(2)},${(y + 5).toFixed(2)} Z`,
      });
    }
    return createSvg("circle", {
      ...common,
      cx: x,
      cy: y,
      r: actual.state === "risk_on" ? 4 : 3.8,
    });
  }

  function renderHistory() {
    const history = selectedHistory();
    const historyMeta = historyComparisonMeta();
    state.chartHistory = history;
    state.chartPreviewDate = null;
    renderHistoryTable(history);
    const svg = dom["probability-chart"];
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${CHART_DIMENSIONS.width} ${CHART_DIMENSIONS.height}`);
    dom["chart-tooltip"].hidden = true;

    if (!history.length) {
      renderChartReadout(null);
      const empty = createSvg("text", {
        x: CHART_DIMENSIONS.width / 2,
        y: CHART_DIMENSIONS.height / 2,
        "text-anchor": "middle",
        class: "chart-axis-label",
      });
      empty.textContent = "표시할 관측·예측 히스토리가 없습니다.";
      svg.append(empty);
      return;
    }

    const { width, height, margin } = CHART_DIMENSIONS;
    const layout = chartPanelLayout();
    const x = (index) => chartX(index, history.length);
    const panels = [
      {
        key: "observed",
        top: layout.observedTop,
        title: `${historyMeta.observedMeasure} · t`,
        measure: observedHistoryMeasure,
      },
      {
        key: "forecast",
        top: layout.forecastTop,
        title: `${historyMeta.model} 1주 예측확률 · t→t+1`,
        measure: forecastHistoryMeasure,
      },
    ];

    for (const panel of panels) {
      const title = createSvg("text", {
        x: margin.left,
        y: panel.top - 15,
        class: "chart-panel-title",
      });
      title.textContent = panel.title;
      svg.append(title);
      for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
        const tickY = panel.top + (1 - tick) * layout.panelHeight;
        svg.append(createSvg("line", {
          x1: margin.left,
          y1: tickY,
          x2: width - margin.right,
          y2: tickY,
          class: "chart-grid-line",
        }));
        const label = createSvg("text", {
          x: margin.left - 10,
          y: tickY + 4,
          "text-anchor": "end",
          class: "chart-axis-label",
        });
        label.textContent = `${Math.round(tick * 100)}%`;
        svg.append(label);
      }
    }

    svg.append(createSvg("line", {
      x1: margin.left,
      y1: layout.observedTop + layout.panelHeight + CHART_DIMENSIONS.panelGap / 2,
      x2: width - margin.right,
      y2: layout.observedTop + layout.panelHeight + CHART_DIMENSIONS.panelGap / 2,
      class: "chart-panel-separator",
      "aria-hidden": "true",
    }));

    const desiredTicks = Math.min(7, history.length);
    const tickIndexes = new Set();
    for (let step = 0; step < desiredTicks; step += 1) {
      tickIndexes.add(desiredTicks === 1 ? 0 : Math.round((step / (desiredTicks - 1)) * (history.length - 1)));
    }
    for (const index of tickIndexes) {
      const label = createSvg("text", {
        x: x(index), y: height - 8, "text-anchor": index === 0 ? "start" : index === history.length - 1 ? "end" : "middle", class: "chart-date-label",
      });
      label.textContent = history[index].date;
      svg.append(label);
    }

    if (chartIndexForDate(state.chartPinnedDate) < 0) state.chartPinnedDate = history[history.length - 1].date;
    const pinnedIndex = chartIndexForDate(state.chartPinnedDate);
    const cursorX = x(pinnedIndex);
    svg.append(createSvg("line", {
      x1: cursorX,
      y1: margin.top,
      x2: cursorX,
      y2: layout.outcomeY + 8,
      class: "chart-selected-line",
      "data-chart-cursor": "true",
      "aria-hidden": "true",
    }));

    for (const panel of panels) {
      let validPointCount = 0;
      const y = (value) => panel.top + (1 - value) * layout.panelHeight;
      for (const code of STATE_ORDER) {
        const points = history.map((week, index) => {
          const value = panel.measure(week, code);
          return { week, value, x: x(index), y: value === null ? null : y(value), index };
        });
        const pathData = makeLinePath(points);
        if (pathData) {
          svg.append(createSvg("path", {
            d: pathData,
            class: `chart-series ${code} ${panel.key}`,
            "data-chart-panel": panel.key,
          }));
        }
        for (const point of points) {
          if (point.value === null) continue;
          validPointCount += 1;
          svg.append(createSvg("circle", {
            cx: point.x,
            cy: point.y,
            r: 2.5,
            class: `chart-point ${code} ${panel.key}`,
            "data-date": point.week.date,
            focusable: "false",
            "aria-hidden": "true",
          }));
        }
      }
      if (!validPointCount) {
        const empty = createSvg("text", {
          x: width / 2,
          y: panel.top + layout.panelHeight / 2,
          "text-anchor": "middle",
          class: "chart-axis-label",
        });
        empty.textContent = panel.key === "observed" ? "관측 값이 없습니다." : "예측 값이 없습니다.";
        svg.append(empty);
      }
    }

    const actualLabel = createSvg("text", {
      x: margin.left - 10,
      y: layout.outcomeY + 4,
      "text-anchor": "end",
      class: "chart-outcome-label",
    });
    actualLabel.textContent = "실제 t+1";
    svg.append(actualLabel);
    svg.append(createSvg("line", {
      x1: margin.left,
      y1: layout.outcomeY,
      x2: width - margin.right,
      y2: layout.outcomeY,
      class: "chart-outcome-baseline",
      "aria-hidden": "true",
    }));
    for (const [index, week] of history.entries()) {
      svg.append(createActualOutcomeMarker(actualNextWeekForWeek(week), x(index), layout.outcomeY, week.date));
    }

    const range = `${formatDate(history[0].date)}–${formatDate(history[history.length - 1].date)}`;
    dom["history-caption"].textContent = `${range} · ${history.length}주 · 상단 ${historyMeta.observedMeasure}, 하단 ${historyMeta.model} 1주 예측확률 · 하단 마커는 실제 다음 주 국면 · 두 패널 모두 0–100% 축`;
    renderChartReadout(state.chartPinnedDate);
    requestAnimationFrame(() => scrollChartDateIntoView(state.chartPinnedDate));
  }

  function hideChartTooltip() {
    dom["chart-tooltip"].hidden = true;
  }

  function renderHistoryTable(history) {
    dom["history-data-body"].replaceChildren();
    if (!history.length) {
      const row = createElement("tr");
      const cell = createElement("td", null, "표시할 값이 없습니다.");
      cell.colSpan = 11;
      row.append(cell);
      dom["history-data-body"].append(row);
      return;
    }
    for (const week of history) {
      const row = createElement("tr");
      row.append(createElement("td", null, week.date));
      for (const code of STATE_ORDER) row.append(createElement("td", null, formatPercent(observedHistoryMeasure(week, code))));
      const forecast = forecastForWeek(week, state.comparisonModel, state.raw);
      const actual = actualNextWeekForWeek(week);
      row.append(createElement("td", null, actual.date || "—"));
      for (const code of STATE_ORDER) row.append(createElement("td", null, formatPercent(forecastHistoryMeasure(week, code))));
      const predictedState = isObject(forecast) && STATE_ORDER.includes(forecast.state) ? forecast.state : null;
      row.append(createElement("td", null, predictedState ? stateMeta(predictedState).label : "—"));
      const actualText = actual.status === "available"
        ? `${stateMeta(actual.state).label}${predictedState ? ` · ${actual.state === predictedState ? "일치" : "불일치"}` : ""}`
        : actual.status === "pending"
          ? "결과 대기"
          : actual.status === "missing"
            ? "결과 없음"
            : "—";
      row.append(createElement("td", null, actualText));
      row.append(createElement("td", null, formatNumber(forecastEntropyForWeek(week), 3)));
      dom["history-data-body"].append(row);
    }
  }

  function setTimelineTabStop(button) {
    const buttons = [...dom["regime-timeline"].querySelectorAll("button.timeline-cell")];
    for (const item of buttons) item.tabIndex = item === button ? 0 : -1;
  }

  function focusTimelineDate(date, scroll = false) {
    const button = [...dom["regime-timeline"].querySelectorAll("button.timeline-cell")]
      .find((item) => item.dataset.date === date);
    if (!button) return;
    setTimelineTabStop(button);
    button.focus({ preventScroll: true });
    if (scroll && typeof button.scrollIntoView === "function") {
      button.scrollIntoView({ block: "nearest", inline: "center" });
    }
  }

  function handleTimelineKeydown(event) {
    const keys = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "Escape", "Enter", " "]);
    if (!keys.has(event.key)) return;
    const buttons = [...dom["regime-timeline"].querySelectorAll("button.timeline-cell")];
    if (!buttons.length) return;
    event.preventDefault();

    if (["Enter", " "].includes(event.key)) {
      event.currentTarget.click();
      return;
    }

    if (event.key === "Escape") {
      focusTimelineDate(selectedWeek() ? selectedWeek().date : buttons[buttons.length - 1].dataset.date, true);
      return;
    }

    let index = Math.max(0, buttons.indexOf(event.currentTarget));
    if (event.key === "Home") index = 0;
    if (event.key === "End") index = buttons.length - 1;
    if (["ArrowLeft", "ArrowUp"].includes(event.key)) index = Math.max(0, index - 1);
    if (["ArrowRight", "ArrowDown"].includes(event.key)) index = Math.min(buttons.length - 1, index + 1);
    setTimelineTabStop(buttons[index]);
    buttons[index].focus({ preventScroll: true });
    if (typeof buttons[index].scrollIntoView === "function") {
      buttons[index].scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }

  function renderTimeline() {
    const history = selectedHistory();
    dom["regime-timeline"].replaceChildren();
    if (!history.length) {
      dom["regime-timeline"].append(createElement("p", "empty-inline", "표시할 국면 타임라인이 없습니다."));
      setText(dom["timeline-start"], "—");
      setText(dom["timeline-end"], "—");
      return;
    }

    for (const week of history) {
      const code = historyStateForWeek(week, "observed", state.comparisonModel, state.raw);
      const meta = stateMeta(code);
      const button = createElement("button", `timeline-cell ${STATE_ORDER.includes(code) ? code : "unknown"}`, week.date);
      button.type = "button";
      button.dataset.date = week.date;
      const seriesLabel = "관측 국면";
      button.setAttribute("aria-label", `${formatDate(week.date)} ${seriesLabel} ${meta.ko}`);
      button.title = `${week.date} · ${seriesLabel} · ${meta.ko}`;
      const isSelected = week.date === selectedWeek().date;
      button.tabIndex = isSelected ? 0 : -1;
      if (isSelected) button.setAttribute("aria-current", "date");
      button.addEventListener("keydown", handleTimelineKeydown);
      button.addEventListener("click", () => {
        const index = state.weekly.findIndex((item) => item.date === week.date);
        if (index >= 0) {
          selectWeek(index, true);
          focusTimelineDate(week.date, true);
        }
      });
      dom["regime-timeline"].append(button);
    }
    setText(dom["timeline-start"], history[0].date);
    setText(dom["timeline-end"], history[history.length - 1].date);
  }

  function renderFactors(scores) {
    dom["factor-scores"].replaceChildren();
    const entries = Object.entries(FACTOR_META).map(([key, label]) => {
      const raw = isObject(scores) ? scores[key] : null;
      const value = finiteNumber(isObject(raw) ? firstValue(raw, ["value", "score"]) : raw);
      return { key, label, value };
    });
    const available = entries.filter((entry) => entry.value !== null);
    if (!available.length) {
      dom["factor-scores"].append(createElement("p", "empty-inline", "이 관측 주에는 팩터 점수가 없습니다."));
      dom["factor-axis"].replaceChildren(createElement("span", null, "−1"), createElement("span", null, "0"), createElement("span", null, "+1"));
      return;
    }

    const maxAbsolute = Math.max(1, ...available.map((entry) => Math.abs(entry.value)));
    const scale = Math.ceil(maxAbsolute * 10) / 10;
    dom["factor-axis"].replaceChildren(
      createElement("span", null, `−${formatNumber(scale, 1)}`),
      createElement("span", null, "0"),
      createElement("span", null, `+${formatNumber(scale, 1)}`),
    );
    dom["factor-caption"].textContent = `52주 표준화 기반 합성점수 · −${formatNumber(scale, 1)}~+${formatNumber(scale, 1)}`;

    for (const entry of entries) {
      const row = createElement("div", "factor-row");
      const label = createElement("span", "factor-label", entry.label);
      const track = createElement("div", "factor-track");
      const fill = createElement("span", `factor-fill ${entry.value !== null && entry.value < 0 ? "negative" : "positive"}`);
      if (entry.value !== null) fill.style.width = `${Math.min(50, (Math.abs(entry.value) / scale) * 50).toFixed(2)}%`;
      track.append(fill);
      const value = createElement("span", "factor-value", entry.value === null ? "—" : `${entry.value > 0 ? "+" : ""}${formatNumber(entry.value, 2)}`);
      row.setAttribute("aria-label", `${entry.label} 점수 ${entry.value === null ? "없음" : formatNumber(entry.value, 2)}`);
      row.append(label, track, value);
      dom["factor-scores"].append(row);
    }
  }

  function driverValue(driver) {
    return finiteNumber(firstValue(driver, ["contribution", "impact", "shap_value", "importance"]));
  }

  function renderDrivers(drivers) {
    dom["top-drivers"].replaceChildren();
    if (!Array.isArray(drivers) || !drivers.length) {
      dom["top-drivers"].append(createElement("li", "empty-inline", "이 관측 주에는 드라이버 설명 값이 없습니다."));
      return;
    }
    const sorted = [...drivers]
      .filter((driver) => isObject(driver))
      .sort((left, right) => Math.abs(driverValue(right) || 0) - Math.abs(driverValue(left) || 0))
      .slice(0, 8);
    const maxAbsolute = Math.max(4, ...sorted.map((driver) => Math.abs(driverValue(driver) || 0)));

    for (const driver of sorted) {
      const label = textValue(firstValue(driver, ["label", "feature", "name", "id"]), "이름 없는 feature");
      const direction = textValue(firstValue(driver, ["direction", "target_state", "state"]), "unknown");
      const directionMeta = stateMeta(direction);
      const contribution = driverValue(driver);
      const observed = firstValue(driver, ["value", "observed_value", "feature_value"]);
      const item = createElement("li", "driver-item");
      const copy = createElement("div", "driver-copy");
      copy.append(
        createElement("strong", null, label),
        createElement("span", null, `${STATE_ORDER.includes(direction) ? `${directionMeta.ko} 방향` : "방향 미지정"}${observed !== null ? ` · 관측값 ${textValue(observed)}` : ""}`),
      );
      const track = createElement("span", "driver-track");
      const signClass = contribution !== null && contribution < 0 ? "negative" : "positive";
      const fill = createElement("span", `driver-fill ${STATE_ORDER.includes(direction) ? direction : ""} ${signClass}`);
      fill.style.width = contribution === null || maxAbsolute === 0 ? "0" : `${(Math.abs(contribution) / maxAbsolute * 50).toFixed(2)}%`;
      track.append(fill);
      const impact = createElement("span", "driver-impact", contribution === null ? "—" : `${contribution > 0 ? "+" : ""}${formatNumber(contribution, 3)}`);
      item.setAttribute("aria-label", `${label}, ${STATE_ORDER.includes(direction) ? `${directionMeta.ko} 방향, ` : ""}evidence ${formatNumber(contribution, 3)}`);
      item.append(copy, track, impact);
      dom["top-drivers"].append(item);
    }
  }

  function renderContextExtremes(extremes) {
    dom["top-drivers"].replaceChildren();
    if (!Array.isArray(extremes) || !extremes.length) {
      dom["top-drivers"].append(createElement("li", "empty-inline", "이 관측 주에는 52주 극단값이 없습니다."));
      return;
    }
    const rows = [...extremes]
      .filter((row) => isObject(row))
      .sort((left, right) => Math.abs(right.z_score) - Math.abs(left.z_score))
      .slice(0, 8);
    const scale = Math.max(4, ...rows.map((row) => Math.abs(row.z_score)));
    for (const extreme of rows) {
      const item = createElement("li", "driver-item");
      const copy = createElement("div", "driver-copy");
      copy.append(
        createElement("strong", null, textValue(extreme.label, textValue(extreme.feature))),
        createElement("span", null, extreme.position === "high" ? "52주 분포 상단" : "52주 분포 하단"),
      );
      const track = createElement("span", "driver-track");
      const fill = createElement("span", `driver-fill ${extreme.z_score < 0 ? "negative" : "positive"}`);
      fill.style.width = `${Math.min(50, Math.abs(extreme.z_score) / scale * 50).toFixed(2)}%`;
      track.append(fill);
      const value = createElement("span", "driver-impact", `${extreme.z_score > 0 ? "+" : ""}${formatNumber(extreme.z_score, 2)}`);
      item.setAttribute("aria-label", `${textValue(extreme.label, extreme.feature)}, 52주 분포 ${extreme.position === "high" ? "상단" : "하단"}, z-score ${formatNumber(extreme.z_score, 2)}`);
      item.append(copy, track, value);
      dom["top-drivers"].append(item);
    }
  }

  function appendMetric(container, label, value) {
    const wrapper = createElement("div");
    wrapper.append(createElement("dt", null, label), createElement("dd", null, value));
    container.append(wrapper);
  }

  function renderDurationContext(duration) {
    const card = dom["duration-context-card"];
    if (!isV5Payload() || !isObject(duration)) {
      card.hidden = true;
      setText(dom["duration-research-detail"], "지속 기간 통계가 없습니다.");
      return;
    }
    const supported = duration.status === "ok";
    const median = finiteNumber(duration.median_remaining_weeks);
    const rmst = finiteNumber(duration.restricted_mean_remaining_weeks);
    dom["duration-context"].replaceChildren();
    appendMetric(dom["duration-context"], "현재 지속", `${formatNumber(duration.elapsed_weeks, 0)}주`);
    appendMetric(
      dom["duration-context"],
      median === null ? "52주 제한 잔여기간" : "중앙 잔여기간",
      `${formatNumber(median === null ? rmst : median, 1)}주`,
    );
    setText(dom["duration-context-caption"], supported ? "현재 상태의 과거 지속 패턴" : "표본 축적 중");
    setText(
      dom["duration-research-detail"],
      `상태별 Kaplan–Meier · 완료 구간 ${formatNumber(duration.completed_spells, 0)}개 · 검열 구간 ${formatNumber(duration.censored_spells, 0)}개`,
    );
    dom["duration-baselines"].replaceChildren();
    for (const horizon of [4, 13]) {
      const departure = probability(duration.departure_probability && duration.departure_probability[`${horizon}w`]);
      const block = createElement("div", "duration-baseline");
      block.append(createElement("span", null, `${horizon}주 이탈 · 과거 KM`), createElement("strong", null, formatPercent(departure)));
      dom["duration-baselines"].append(block);
    }
    card.hidden = false;
  }

  function formatFxMetric(key, value) {
    const number = finiteNumber(value);
    if (number === null) return "—";
    if (/(return|vol|share|mad|minus_afe)/i.test(key) && Math.abs(number) <= 1) return formatSignedPercent(number);
    return `${number > 0 ? "+" : ""}${formatNumber(number, 3)}`;
  }

  function renderFxContext(fx) {
    const card = dom["fx-context-card"];
    if (!isV5Payload() || !isObject(fx)) {
      card.hidden = true;
      dom["fx-context-detail"].replaceChildren();
      return;
    }
    const available = finiteNumber(fx.coverage && fx.coverage.available_pairs);
    const availableIndexes = finiteNumber(fx.coverage && fx.coverage.available_indexes);
    dom["fx-coverage"].replaceChildren(
      createElement("span", null, `환율 ${formatNumber(available, 0)}개 · 달러 지수 ${formatNumber(availableIndexes, 0)}개`),
      createElement("span", `support-chip ${fx.status === "ok" ? "is-ok" : "is-limited"}`, fxStatusLabel(fx.status)),
    );
    const model = state.raw && isObject(state.raw.model) ? state.raw.model : {};
    const ablation = isObject(model.fx_ablation) ? model.fx_ablation : {};
    const eligibleWeeks = finiteNumber(ablation.eligible_common_weeks);
    const minimumWeeks = finiteNumber(ablation.minimum_common_weeks);
    const requiredPairs = finiteNumber(ablation.common_origin_required_pairs);
    const evaluationOrigins = finiteNumber(
      isObject(ablation.common_evaluation_origins)
        ? ablation.common_evaluation_origins.count
        : null,
    );
    const comparisons = isObject(ablation.gate) && Array.isArray(ablation.gate.comparisons)
      ? ablation.gate.comparisons
      : [];
    const passedVariants = isObject(ablation.gate) && Array.isArray(ablation.gate.passed_variants)
      ? ablation.gate.passed_variants.length
      : 0;
    const ablationLabel = fxStatusLabel(ablation.status);
    const coverageSummary = [
      `FX 평가 ${ablationLabel}`,
      `가용 공통 ${formatNumber(eligibleWeeks, 0)}/${formatNumber(minimumWeeks, 0)}주`,
      `실제 OOS ${formatNumber(evaluationOrigins, 0)}개`,
      `통화 ${formatNumber(requiredPairs, 0)}/9`,
      ablation.role === "prospective_shadow" ? "전향적 shadow" : "shadow 상태 확인",
    ].join(" · ");
    const gateSummary = [
      `FX 후보 gate ${formatNumber(passedVariants, 0)}/${formatNumber(comparisons.length, 0)} 통과`,
      ablation.core_champion_promoted === true ? "core 승격" : "core 비승격",
    ].join(" · ");
    dom["fx-ablation-status"].replaceChildren(
      createElement("span", null, coverageSummary),
      createElement("strong", null, gateSummary),
    );
    dom["fx-ablation-status"].setAttribute("aria-label", `${coverageSummary} · ${gateSummary}`);
    dom["fx-context"].replaceChildren();
    dom["fx-context-detail"].replaceChildren();
    for (const metric of FX_DISPLAY_METRICS) {
      const block = isObject(fx[metric.block]) ? fx[metric.block] : {};
      const formatted = formatFxMetric(metric.key, block[metric.key]);
      if (metric.primary) appendMetric(dom["fx-context"], metric.label, formatted);
      appendMetric(dom["fx-context-detail"], metric.label, formatted);
    }
    const observationWeek = typeof fx.observation_week === "string" && fx.observation_week
      ? `관측 ${formatDate(fx.observation_week, false)}`
      : null;
    const observationAge = finiteNumber(fx.observation_age_days);
    const observationLag = observationAge === null
      ? null
      : observationAge === 0
        ? "동일 주"
        : `${formatNumber(observationAge, 0)}일 전`;
    setText(
      dom["fx-context-caption"],
      ["Federal Reserve H.10", observationWeek, observationLag].filter(Boolean).join(" · "),
    );
    card.hidden = false;
  }

  function modelConditionedAssetRows(payload, requestedModel) {
    const research = isObject(payload) && isObject(payload.research) ? payload.research : null;
    const stats = research && isObject(research.model_conditioned_asset_stats)
      ? research.model_conditioned_asset_stats
      : null;
    if (!stats || !Array.isArray(stats.rows) || typeof requestedModel !== "string" || !requestedModel) return [];
    return stats.rows.filter(
      (row) => isObject(row) && row.conditioning_model === requestedModel,
    );
  }

  function modelConditionedAssetRowsComplete(payload, requestedModel) {
    if (!forecastComparisonModels(payload).includes(requestedModel)) return false;
    const rows = modelConditionedAssetRows(payload, requestedModel);
    const expectedCount = OUTCOME_ASSETS.length * STATE_ORDER.length * TRANSITION_HORIZONS.length;
    if (rows.length !== expectedCount) return false;
    const combinations = new Set();
    for (const row of rows) {
      if (
        !OUTCOME_ASSETS.includes(row.asset)
        || !STATE_ORDER.includes(row.state)
        || !TRANSITION_HORIZONS.includes(row.horizon_weeks)
      ) return false;
      combinations.add(`${row.asset}|${row.state}|${row.horizon_weeks}`);
    }
    return combinations.size === expectedCount;
  }

  function conditionalStatsRowsForBasis(payload, basis = "observed", requestedModel = null) {
    const research = isObject(payload) && isObject(payload.research) ? payload.research : null;
    if (basis === "forecast") return modelConditionedAssetRows(payload, requestedModel);
    const stats = research && isObject(research.conditional_asset_stats)
      ? research.conditional_asset_stats
      : null;
    return stats && Array.isArray(stats.rows) ? stats.rows : [];
  }

  function conditionalStatsRows() {
    return conditionalStatsRowsForBasis(
      state.raw,
      state.outcomeBasis,
      state.comparisonModel,
    );
  }

  function conditionalBasisLabel() {
    return state.outcomeBasis === "forecast"
      && modelConditionedAssetRowsComplete(state.raw, state.comparisonModel)
      ? `${modelForecastLabel(state.comparisonModel)} OOS 예측 국면 기준`
      : "관측 국면 기준";
  }

  function syncConditionalBasisControl() {
    const field = dom["conditional-basis-field"];
    const select = dom["conditional-basis-select"];
    if (!field || !select) {
      state.outcomeBasis = "observed";
      return;
    }
    const supported = modelConditionedAssetRowsComplete(state.raw, state.comparisonModel);
    const forecastOption = [...select.options].find((option) => option.value === "forecast");
    if (forecastOption) forecastOption.disabled = !supported;
    if (!supported && state.outcomeBasis === "forecast") state.outcomeBasis = "observed";
    field.hidden = !supported;
    select.value = state.outcomeBasis;
    select.setAttribute("aria-label", `국면 기준 · ${conditionalBasisLabel()}`);
  }

  function conditionalDetailRows() {
    return conditionalStatsRows().filter(
      (row) => row.asset === state.outcomeAsset && row.horizon_weeks === state.outcomeHorizon,
    );
  }

  function conditionalComparisonRows() {
    return conditionalStatsRows().filter((row) => row.horizon_weeks === state.outcomeHorizon);
  }

  function conditionalInterval(row) {
    return row.mean_return_ci95_lower !== null && row.mean_return_ci95_lower !== undefined
      && row.mean_return_ci95_upper !== null && row.mean_return_ci95_upper !== undefined
      ? `${formatSignedPercent(row.mean_return_ci95_lower)}–${formatSignedPercent(row.mean_return_ci95_upper)}`
      : "—";
  }

  function renderConditionalComparison() {
    dom["conditional-horizon-select"].value = String(state.outcomeHorizon);
    const comparisonRows = conditionalComparisonRows();
    const comparisonDigits = state.outcomeHorizon === 1 ? 2 : 1;
    const displayValue = (value) => {
      const number = finiteNumber(value);
      if (number === null) return null;
      const factor = 10 ** (comparisonDigits + 2);
      const rounded = Math.round(number * factor) / factor;
      return Object.is(rounded, -0) ? 0 : rounded;
    };
    const comparisonValues = comparisonRows
      .filter((row) => row.status === "ok")
      .map((row) => displayValue(row.mean_return))
      .filter((value) => value !== null);
    const comparisonScale = Math.max(0.001, ...comparisonValues.map((value) => Math.abs(value)));
    const basisLabel = conditionalBasisLabel();
    dom["conditional-stat-grid"].replaceChildren();
    dom["conditional-stat-grid"].setAttribute("aria-label", `${basisLabel} 자산군 평균 수익률 비교`);
    setText(
      dom["conditional-stats-caption"],
      `${basisLabel} · 자산군별 과거 평균 수익률 · USD · ${publicationSnapshotLabel()}`,
    );
    setText(dom["conditional-comparison-caption"], `${basisLabel} · ${state.outcomeHorizon}주 보유 · 동일 축 · 0 기준`);
    for (const asset of OUTCOME_ASSETS) {
      const card = createElement("article", "conditional-asset-card");
      const heading = createElement("div", "conditional-asset-heading");
      heading.append(
        createElement("strong", null, asset),
        createElement("span", null, OUTCOME_ASSET_LABELS[asset]),
      );
      const regimeList = createElement("div", "conditional-regime-list");
      for (const code of STATE_ORDER) {
        const row = comparisonRows.find((candidate) => candidate.state === code && candidate.asset === asset);
        const supportOk = Boolean(row && row.status === "ok");
        const rawValue = supportOk ? finiteNumber(row.mean_return) : null;
        const value = displayValue(rawValue);
        const available = value !== null;
        const statusLabel = !row
          ? "데이터 없음"
          : !supportOk
            ? "표본 부족"
            : !available
              ? "값 없음"
              : "";
        const sample = row ? formatNumber(row.n, 0) : "—";
        const regimeRow = createElement(
          "div",
          `conditional-regime-row state-${code}${available ? "" : " is-limited"}`,
        );
        const label = createElement("span", "conditional-regime-label");
        label.append(
          createElement("b", null, stateMeta(code).short),
          createElement("span", null, stateMeta(code).label),
        );
        const track = createElement("span", "conditional-return-track");
        track.setAttribute("aria-hidden", "true");
        if (available && value !== 0) {
          const fill = createElement(
            "span",
            `conditional-return-fill state-${code} ${value < 0 ? "is-negative" : "is-positive"}`,
          );
          fill.style.width = `${Math.min(50, (Math.abs(value) / comparisonScale) * 50).toFixed(2)}%`;
          track.append(fill);
        }
        const formatted = value === null
          ? "—"
          : `${value > 0 ? "+" : ""}${formatSignedPercent(value, comparisonDigits)}`;
        const output = createElement("span", `conditional-return-value ${value !== null && value < 0 ? "is-negative" : "is-positive"}`);
        output.append(
          createElement("strong", null, formatted),
          createElement("small", null, statusLabel || `n ${sample}`),
        );
        regimeRow.setAttribute(
          "aria-label",
          `${basisLabel}, ${stateMeta(code).label}, ${asset} ${OUTCOME_ASSET_LABELS[asset]}, ${state.outcomeHorizon}주 평균 수익률 ${formatted}, 표본 ${sample}개${statusLabel ? `, ${statusLabel}` : ""}`,
        );
        regimeRow.append(label, track, output);
        regimeList.append(regimeRow);
      }
      card.setAttribute("aria-label", `${basisLabel}, ${asset} ${OUTCOME_ASSET_LABELS[asset]} 국면별 ${state.outcomeHorizon}주 평균 수익률`);
      card.append(heading, regimeList);
      dom["conditional-stat-grid"].append(card);
    }
  }

  function renderConditionalDetail() {
    dom["conditional-asset-select"].value = state.outcomeAsset;
    const detailRows = conditionalDetailRows();
    dom["conditional-stat-body"].replaceChildren();
    for (const code of STATE_ORDER) {
      const row = detailRows.find((candidate) => candidate.state === code) || { state: code, n: 0, unique_episodes: 0, status: "insufficient_support" };
      const statusLabel = row.status !== "ok"
        ? "표본 부족"
        : finiteNumber(row.mean_return) === null
          ? "값 없음"
          : "";
      const tableRow = createElement("tr");
      tableRow.append(
        createElement("td", null, stateMeta(code).label),
        createElement("td", null, formatSignedPercent(row.mean_return)),
        createElement("td", null, formatSignedPercent(row.median_return)),
        createElement("td", null, formatPercent(row.positive_rate)),
        createElement("td", null, conditionalInterval(row)),
        createElement("td", null, formatPercent(row.downside_volatility)),
        createElement("td", null, formatSignedPercent(row.cvar_5)),
        createElement("td", null, formatSignedPercent(row.mean_max_drawdown)),
        createElement("td", null, `${formatNumber(row.n, 0)} / ${formatNumber(row.unique_episodes, 0)}${statusLabel ? ` · ${statusLabel}` : ""}`),
      );
      dom["conditional-stat-body"].append(tableRow);
    }
    const basisLabel = conditionalBasisLabel();
    setText(dom["conditional-stat-table-caption"], `${basisLabel} · ${state.outcomeAsset} ${state.outcomeHorizon}주 국면별 조건부 성과`);
    if (dom["conditional-stat-scroll"]) {
      dom["conditional-stat-scroll"].setAttribute(
        "aria-label",
        `${basisLabel} · 선택 자산과 보유 기간의 국면별 조건부 성과 표 · 가로 스크롤 가능`,
      );
    }
  }

  function renderConditionalStats() {
    const section = dom["conditional-stats"];
    if (!isV5Payload()) {
      section.hidden = true;
      dom["conditional-stats-nav"].hidden = true;
      return;
    }
    dom["conditional-stats-nav"].hidden = false;
    syncConditionalBasisControl();
    setText(
      dom["conditional-stats-caption"],
      `${conditionalBasisLabel()} · 자산군별 과거 평균 수익률 · USD · ${publicationSnapshotLabel()}`,
    );
    renderConditionalComparison();
    renderConditionalDetail();
    section.hidden = false;
  }

  function metricParts(key, raw) {
    if (isObject(raw)) {
      return {
        value: firstValue(raw, ["value", "score", "amount"]),
        unit: firstValue(raw, ["unit", "units"]),
        format: firstValue(raw, ["format", "display_format"]),
        label: firstValue(raw, ["label", "name"]) || MARKET_LABELS[key] || humanizeKey(key),
        percentile: probability(firstValue(raw, ["percentile_52w"])),
      };
    }
    return { value: raw, unit: null, format: null, label: MARKET_LABELS[key] || humanizeKey(key), percentile: null };
  }

  function formatMetric(key, raw) {
    const metric = metricParts(key, raw);
    const number = finiteNumber(metric.value);
    if (number === null) return textValue(metric.value);
    const format = String(metric.format || "").toLowerCase();
    const unit = String(metric.unit || "");
    if (format === "signed_percent") return formatSignedPercent(number);
    if (format === "plain_percent") return formatPercent(number);
    if (format === "signed_number") return `${number > 0 ? "+" : ""}${formatNumber(number, 3)}`;
    if (format === "percent") return formatSignedPercent(number);
    if (format === "probability") return formatPercent(number);
    if (unit === "%") return `${formatNumber(number, 2)}%`;
    if (format === "currency" || ["USD", "$"].includes(unit)) return `$${formatCompactNumber(number, 2)}`;
    if (/(^|_)(return|drawdown|vol|volatility)(_|$)/i.test(key) && Math.abs(number) <= 1) return formatSignedPercent(number);
    return `${formatCompactNumber(number, 3)}${unit ? ` ${unit}` : ""}`;
  }

  function legacySpyClose(index) {
    const week = state.weekly[index];
    if (!week || !isObject(week.market)) return null;
    return finiteNumber(metricParts("spy_close", week.market.spy_close).value);
  }

  function legacyMarketContext(market) {
    const currentClose = legacySpyClose(state.selectedIndex);
    const halfYearClose = legacySpyClose(state.selectedIndex - 26);
    const returnFromCloses = (prior) => (
      currentClose === null || prior === null || prior <= 0
        ? null
        : currentClose / prior - 1
    );
    return [
      ["spy_trend_26w", { value: returnFromCloses(halfYearClose), format: "signed_percent" }],
      ["spy_realized_vol_13w", { value: metricParts("realized_vol_13w", market.realized_vol_13w).value, format: "plain_percent" }],
      ["spy_drawdown_52w", { value: metricParts("drawdown_52w", market.drawdown_52w).value, format: "signed_percent" }],
    ];
  }

  function marketContextEntries(market) {
    const keys = [
      "spy_trend_26w",
      "spy_realized_vol_13w",
      "spy_drawdown_52w",
      "gics_sector_breadth_4w",
      "hyg_lqd_relative_13w",
      "anfci_change_4w",
    ];
    if (keys.some((key) => Object.hasOwn(market, key))) {
      return keys.filter((key) => Object.hasOwn(market, key)).map((key) => [key, market[key]]);
    }
    return legacyMarketContext(market);
  }

  function renderMarket(market) {
    dom["market-context"].replaceChildren();
    if (!isObject(market) || !Object.keys(market).length) {
      const empty = createElement("div", "empty-inline");
      empty.append(createElement("dt", "sr-only", "데이터 상태"), createElement("dd", null, "이 관측 주에는 시장 맥락 값이 없습니다."));
      dom["market-context"].append(empty);
      return;
    }
    const entries = marketContextEntries(market);
    for (const [key, raw] of entries) {
      const parts = metricParts(key, raw);
      const wrapper = createElement("div");
      wrapper.append(createElement("dt", null, parts.label), createElement("dd", null, formatMetric(key, raw)));
      if (parts.percentile !== null) {
        wrapper.append(createElement("small", "metric-percentile", `52주 ${Math.round(parts.percentile * 100)}백분위`));
      }
      dom["market-context"].append(wrapper);
    }
  }

  function modelName(value) {
    if (typeof value === "string") return value;
    return textValue(firstValue(value, ["name", "model", "id", "label"]), "선정 모델 없음");
  }

  function metricValue(row, keys) {
    const direct = firstValue(row, keys);
    if (direct !== null) return finiteNumber(direct);
    return finiteNumber(firstValue(isObject(row) ? row.metrics : null, keys));
  }

  function transitionSplitLabel(value) {
    return value === "selection" ? "선정 구간" : "2023+ 진단";
  }

  function renderTransitionModels() {
    const model = state.raw && isObject(state.raw.model) ? state.raw.model : {};
    const section = dom["transition-model-section"];
    const allRows = Array.isArray(model.transition_leaderboard) ? model.transition_leaderboard : [];
    const resultVersion = isObject(state.raw && state.raw.meta)
      ? state.raw.meta.result_version
      : null;
    const hasTransitionContract = [V3_RESULT_VERSION, V4_RESULT_VERSION].includes(resultVersion);
    if (!hasTransitionContract || !allRows.length) {
      section.hidden = true;
      return;
    }

    const horizon = TRANSITION_HORIZONS.includes(state.transitionHorizon) ? state.transitionHorizon : 1;
    dom["transition-horizon-select"].value = String(horizon);
    const rows = allRows
      .filter((row) => isObject(row) && row.horizon_weeks === horizon)
      .sort((left, right) => {
        if (Boolean(left.selected) !== Boolean(right.selected)) return left.selected ? -1 : 1;
        if (left.evaluation_split !== right.evaluation_split) {
          return left.evaluation_split === "retrospective_diagnostic" ? -1 : 1;
        }
        return (finiteNumber(left.binary_log_loss) || 0) - (finiteNumber(right.binary_log_loss) || 0);
      });

    setText(dom["transition-leaderboard-caption"], `${horizon}주 이탈 모델 진단 지표`);
    setText(
      dom["transition-model-caption"],
      `${horizon}주 이탈 · 선정 구간 / 2023+ 진단`,
    );

    const preferred = rows.find((row) => row.selected && row.evaluation_split === "retrospective_diagnostic")
      || rows.find((row) => row.selected)
      || rows.find((row) => row.evaluation_split === "retrospective_diagnostic")
      || rows[0];
    const summary = dom["transition-model-summary"];
    summary.replaceChildren();
    if (preferred) {
      const identity = createElement("div", "transition-model-identity");
      identity.append(
        createElement("span", null, preferred.selected ? "선정 모델" : "표시 모델"),
        createElement("strong", null, modelName(preferred)),
        createElement("small", null, transitionSplitLabel(preferred.evaluation_split)),
      );
      const metricGrid = createElement("dl", "transition-model-metrics");
      const metrics = [
        ["Average precision", formatPercent(metricValue(preferred, ["average_precision"]))],
        ["Precision", formatPercent(metricValue(preferred, ["precision"]))],
        ["Recall", formatPercent(metricValue(preferred, ["recall"]))],
        ["False alarms / 연", formatNumber(metricValue(preferred, ["false_alarms_per_year"]), 2)],
      ];
      for (const [label, value] of metrics) {
        const wrapper = createElement("div");
        wrapper.append(createElement("dt", null, label), createElement("dd", null, value));
        metricGrid.append(wrapper);
      }
      summary.append(identity, metricGrid);
    }

    dom["transition-leaderboard-body"].replaceChildren();
    if (!rows.length) {
      const row = createElement("tr");
      const cell = createElement("td", null, `${horizon}주 모델 비교 결과가 없습니다.`);
      cell.colSpan = 9;
      row.append(cell);
      dom["transition-leaderboard-body"].append(row);
    } else {
      for (const rowData of rows) {
        const row = createElement("tr");
        if (rowData.selected) row.classList.add("is-selected-transition-model");
        const nameCell = createElement("td", null, modelName(rowData));
        if (rowData.selected) nameCell.append(createElement("span", "champion-label", "선정"));
        const splitCell = createElement("td", null, transitionSplitLabel(rowData.evaluation_split));
        const sampleText = `${formatNumber(rowData.n_predictions, 0)} / ${formatNumber(rowData.event_count, 0)}`;
        const sampleCell = createElement("td", null, sampleText);
        row.append(
          nameCell,
          splitCell,
          createElement("td", null, formatNumber(metricValue(rowData, ["binary_log_loss"]), 4)),
          createElement("td", null, formatNumber(metricValue(rowData, ["brier"]), 4)),
          createElement("td", null, formatPercent(metricValue(rowData, ["average_precision"]))),
          createElement("td", null, formatPercent(metricValue(rowData, ["precision"]))),
          createElement("td", null, formatPercent(metricValue(rowData, ["recall"]))),
          createElement("td", null, formatNumber(metricValue(rowData, ["false_alarms_per_year"]), 2)),
          sampleCell,
        );
        dom["transition-leaderboard-body"].append(row);
      }
    }
    section.hidden = false;
  }

  function renderModelLossChart(rows, championName, holdoutBestName) {
    const container = dom["model-loss-chart"];
    container.replaceChildren();
    const ranked = rows
      .map((item) => (isObject(item) ? item : { name: textValue(item) }))
      .map((row) => ({
        row,
        name: modelName(row),
        rank: finiteNumber(firstValue(row, ["rank", "position"])),
        selection: metricValue(row, ["selection_log_loss"]),
        holdout: metricValue(row, ["log_loss", "multiclass_log_loss"]),
      }))
      .filter((item) => item.holdout !== null)
      .sort((left, right) => left.holdout - right.holdout);
    const champion = ranked.find((item) => item.name === championName);
    const eligible = ranked.slice(0, 6);
    if (champion && !eligible.some((item) => item.name === champion.name)) {
      eligible.splice(5, 1, champion);
      eligible.sort((left, right) => left.holdout - right.holdout);
    }

    if (!eligible.length) {
      container.append(createElement("p", "empty-inline", "표시할 Log loss 비교 값이 없습니다."));
      setText(dom["model-loss-caption"], "선정 구간·2023+ 진단 값 없음");
      return;
    }

    const values = eligible.flatMap((item) => [item.selection, item.holdout]).filter((value) => value !== null);
    const axisMax = Math.max(1.1, Math.ceil(Math.max(...values) * 10) / 10);
    const includesExtraChampion = champion && champion.rank > 6;
    setText(
      dom["model-loss-caption"],
      `${includesExtraChampion ? "2023+ 상위 5개 + 선정 모델" : `2023+ 상위 ${eligible.length}개`} · 낮을수록 좋음`,
    );
    dom["model-loss-axis"].replaceChildren(
      createElement("span", null, "0"),
      createElement("span", null, formatNumber(axisMax / 2, 2)),
      createElement("span", null, formatNumber(axisMax, 2)),
    );

    for (const item of eligible) {
      const isChampion = item.name === championName || Boolean(item.row.is_champion || item.row.champion);
      const isHoldoutBest = Boolean(holdoutBestName) && item.name === holdoutBestName;
      const chartRow = createElement("div", "model-loss-row");
      if (isChampion) chartRow.classList.add("is-champion");
      if (isHoldoutBest) chartRow.classList.add("is-holdout-best");

      const label = createElement("span", "model-loss-label");
      label.title = item.name;
      label.append(
        createElement("strong", null, modelForecastLabel(item.name)),
        createElement(
          "span",
          null,
          isChampion
            ? `선정 · 2023+ #${formatNumber(item.rank, 0)}`
            : isHoldoutBest
              ? "2023+ #1"
              : `2023+ #${formatNumber(item.rank, 0)}`,
        ),
      );

      const track = createElement("span", "model-loss-track");
      if (item.selection !== null) {
        const selectionPosition = Math.max(0, Math.min(100, (item.selection / axisMax) * 100));
        const holdoutPosition = Math.max(0, Math.min(100, (item.holdout / axisMax) * 100));
        const connector = createElement("span", "model-loss-connector");
        connector.style.left = `${Math.min(selectionPosition, holdoutPosition).toFixed(2)}%`;
        connector.style.width = `${Math.abs(holdoutPosition - selectionPosition).toFixed(2)}%`;
        const selectionMarker = createElement("span", "model-loss-point selection");
        selectionMarker.style.left = `${selectionPosition.toFixed(2)}%`;
        const holdoutMarker = createElement("span", "model-loss-point holdout");
        holdoutMarker.style.left = `${holdoutPosition.toFixed(2)}%`;
        track.append(connector, selectionMarker, holdoutMarker);
      } else {
        const holdoutMarker = createElement("span", "model-loss-point holdout");
        holdoutMarker.style.left = `${Math.max(0, Math.min(100, (item.holdout / axisMax) * 100)).toFixed(2)}%`;
        track.append(holdoutMarker);
      }

      const exact = createElement("span", "model-loss-values");
      exact.append(
        createElement("strong", null, `${formatNumber(item.selection, 4)} → ${formatNumber(item.holdout, 4)}`),
        createElement("span", null, item.selection === null ? "선정 값 없음" : `${item.holdout - item.selection >= 0 ? "+" : ""}${formatNumber(item.holdout - item.selection, 4)}`),
      );
      chartRow.setAttribute(
        "aria-label",
        `${item.name}, 선정 구간 Log loss ${formatNumber(item.selection, 4)}, 2023+ 진단 Log loss ${formatNumber(item.holdout, 4)}${isChampion ? ", 선정 모델" : ""}${isHoldoutBest ? ", 2023+ 진단 1위" : ""}`,
      );
      chartRow.append(label, track, exact);
      container.append(chartRow);
    }
  }

  function modelForecastLabel(name) {
    return {
      majority: "다수 국면",
      persistence: "직전 국면 유지",
      markov: "Markov",
      elastic_net_logistic: "Elastic-net Logistic",
      calibrated_linear_svm: "보정 Linear SVM",
      random_forest: "Random Forest",
      extra_trees: "Extra Trees",
      hist_gradient_boosting: "Histogram Gradient Boosting",
      ridge_logistic: "Ridge Logistic",
      transition_logistic: "전환 Logistic",
      duration_tvtp_hurdle: "Duration TVTP Hurdle",
      shrinkage_lda: "Shrinkage LDA",
      spline_logistic: "Spline Logistic",
      xgboost: "XGBoost",
      xgb_hazard_destination: "XGBoost · 이탈/목적지",
      causal_dynamic_ensemble: "동적 앙상블",
      causal_multiscale_ensemble: "멀티스케일 앙상블",
      recency_weighted_xgboost_208w: "XGBoost · 최근 가중",
      recency_weighted_ridge_logistic_208w: "Ridge Logistic · 최근 가중",
      pca_ridge_logistic: "PCA · Ridge Logistic",
      discounted_markov_208w: "Markov · 최근 가중",
      direct_jump_tvtp_hurdle: "Direct-jump TVTP",
      filtered_hsmm: "Filtered HSMM",
      dynamic_factor_tvtp: "Dynamic-factor TVTP",
      bayesian_online_changepoint: "BOCPD",
    }[name] || textValue(name, "모델");
  }

  function renderModelForecastProbabilities(forecast) {
    const container = dom["model-forecast-probabilities"];
    container.replaceChildren();
    for (const code of STATE_ORDER) {
      const definition = stateMeta(code);
      const value = getProbability(forecast, code);
      const row = createElement("div", "probability-row");
      const label = createElement("span", "probability-label");
      const marker = createElement("span", `state-dot ${code}`, definition.short);
      marker.setAttribute("aria-hidden", "true");
      label.append(marker, document.createTextNode(definition.label));
      const track = createElement("span", "probability-track");
      const fill = createElement("span", `probability-fill ${code}`);
      fill.style.width = value === null ? "0" : `${(value * 100).toFixed(2)}%`;
      track.append(fill);
      const display = createElement("span", "probability-value", formatPercent(value));
      row.setAttribute("aria-label", `${definition.ko} 예측확률 ${formatPercent(value)}`);
      row.append(label, track, display);
      container.append(row);
    }
  }

  function renderModelForecast() {
    const model = state.raw && isObject(state.raw.model) ? state.raw.model : {};
    const models = forecastComparisonModels(state.raw);
    const week = selectedWeek() || state.weekly[state.weekly.length - 1] || null;
    const forecasts = week && Array.isArray(week.model_forecasts)
      ? week.model_forecasts
      : [];
    const supported = isV5Payload()
      && models.length > 0
      && forecasts.length === models.length
      && !suppressCurrentForecastSurface();
    dom["model-forecast-field"].hidden = !supported;
    dom["model-forecast-explorer"].hidden = !supported;
    if (!supported) return;

    const championName = modelName(model.champion);
    const operatingName = operatingChampionName();
    if (!models.includes(state.comparisonModel)) {
      state.comparisonModel = models.includes(operatingName)
        ? operatingName
        : models.includes(championName)
          ? championName
          : models[0];
    }
    const leaderboard = Array.isArray(model.leaderboard) ? model.leaderboard : [];
    const select = dom["model-forecast-select"];
    const optionSignature = models.map((name) => {
      const row = leaderboard.find((candidate) => modelName(candidate) === name);
      return `${name}:${textValue(firstValue(row, ["rank", "position"]), "")}:${name === operatingName}`;
    }).join("|");
    if (select.dataset.models !== optionSignature) {
      select.replaceChildren();
      for (const name of models) {
        const row = leaderboard.find((candidate) => modelName(candidate) === name);
        const rank = finiteNumber(firstValue(row, ["rank", "position"]));
        const role = name === operatingName
          ? "공식"
          : name === championName
            ? "로컬 선정"
            : rank === null
              ? "연구"
              : `연구 · 2023+ #${formatNumber(rank, 0)}`;
        const option = createElement("option", null, `${modelForecastLabel(name)} · ${role}`);
        option.value = name;
        select.append(option);
      }
      select.dataset.models = optionSignature;
    }
    select.value = state.comparisonModel;
    select.setAttribute("aria-label", "1주 예측 모델");

    const forecast = forecastForWeek(week, state.comparisonModel);
    if (!forecast) {
      dom["model-forecast-explorer"].hidden = true;
      return;
    }
    const leaderboardRow = leaderboard.find((row) => modelName(row) === state.comparisonModel) || {};
    const isChampion = state.comparisonModel === operatingName;
    const selectedRole = isChampion
      ? "공식 모델"
      : state.comparisonModel === championName
        ? "로컬 선정 모델"
        : "연구 모델";
    const role = dom["model-forecast-role"];
    role.classList.toggle("is-comparison", !isChampion);
    role.classList.toggle("is-fallback", forecast.fallback === true);
    setText(role, forecast.fallback === true ? `${selectedRole} · 보조값` : selectedRole);
    setText(dom["model-forecast-title"], `${modelForecastLabel(state.comparisonModel)} 주간 예측`);

    const officialForecast = forecastForWeek(week, operatingName);
    const officialState = officialForecast && officialForecast.state;
    const officialModelLabel = modelForecastLabel(operatingName);
    const agreement = forecast.state === officialState
      ? "공식 예측과 국면 일치"
      : `공식 ${officialModelLabel} ${stateMeta(officialState).ko}`;
    setText(
      dom["model-forecast-caption"],
      `${formatDate(week.date, false)} 관측 → ${formatDate(forecast.date, false)} 예측 · ${agreement}`,
    );
    const meta = stateMeta(forecast.state);
    setText(dom["model-forecast-symbol"], meta.symbol);
    setText(dom["model-forecast-state"], `${meta.ko} · ${meta.label}`);
    setText(dom["model-forecast-confidence"], `예측확률 ${formatPercent(forecast.confidence)}`);
    renderModelForecastProbabilities(forecast);

    const rank = finiteNumber(firstValue(leaderboardRow, ["rank", "position"]));
    setText(dom["model-forecast-rank"], rank === null ? "—" : `${formatNumber(rank, 0)} / ${formatNumber(leaderboard.length, 0)}`);
    const selectionLogLoss = metricValue(leaderboardRow, ["selection_log_loss"]);
    const diagnosticLogLoss = metricValue(leaderboardRow, ["log_loss", "multiclass_log_loss"]);
    setText(
      dom["model-forecast-log-loss"],
      selectionLogLoss === null
        ? formatNumber(diagnosticLogLoss, 4)
        : `${formatNumber(selectionLogLoss, 4)} → ${formatNumber(diagnosticLogLoss, 4)}`,
    );
    setText(dom["model-forecast-brier"], formatNumber(metricValue(leaderboardRow, ["brier", "brier_score"]), 4));
    setText(dom["model-forecast-calibration"], formatNumber(metricValue(leaderboardRow, ["calibration_error"]), 4));
    dom["model-forecast-explorer"].setAttribute(
      "aria-label",
      `${modelForecastLabel(state.comparisonModel)} ${formatDate(week.date, false)} 기준 다음 주 ${meta.ko} 예측, 예측확률 ${formatPercent(forecast.confidence)}`,
    );
  }

  function renderModelRoles(model) {
    const grid = dom["model-role-grid"];
    if (!isV5Payload()) {
      grid.hidden = true;
      return;
    }
    grid.hidden = false;
    const selection = isObject(state.raw.selection) ? state.raw.selection : {};
    const selectionEvidence = selectionEvidenceForDisplay(
      state.raw,
      state.selectionFamilyAudit,
    );
    const lifecycle = isObject(model.lifecycle) ? model.lifecycle : {};
    const deployment = isObject(lifecycle.deployment) ? lifecycle.deployment.status : "candidate";
    const operatingName = textValue(selection.operating_champion, modelName(model.champion));
    const operatingStatus = deployment === "operating"
      ? "현재 payload · 공개 운영"
      : "기존 reviewed 운영 기준 유지";
    const researchRunStatus = {
      operating: "운영 반영 완료",
      reviewed: "검토 완료 · 미배포",
      candidate: "candidate · 미배포",
    }[deployment] || "run 상태 확인 필요";
    setText(dom["operating-model-name"], modelForecastLabel(operatingName));
    setText(dom["operating-model-status"], operatingStatus);

    const baseline = isObject(model.baseline_v4) ? model.baseline_v4 : {};
    setText(dom["frozen-baseline-name"], modelForecastLabel(baseline.champion));

    const selectors = selectionEvidence.candidates;
    const researchModels = selectors.filter((name) => name !== operatingName);
    setText(
      dom["research-selector-summary"],
      `${formatNumber(researchModels.length, 0)}개 비교 모델`,
    );
    setText(
      dom["research-selector-detail"],
      researchModels.length
        ? [
          researchModels.slice(0, 3).map(modelForecastLabel).join(" · ")
            + (researchModels.length > 3 ? ` 외 ${researchModels.length - 3}개` : ""),
          selectionEvidence.source === "selection-family-audit/v2"
            ? `generic audit · matched origin ${formatNumber(selectionEvidence.originCount, 0)}`
            : "payload 계약 fallback",
        ].join(" · ")
        : "등록된 비교 모델 없음",
    );

    const reasonLabel = {
      best_gate_passing_log_loss: "gate 통과 최저 Log loss",
      simplicity_tiebreak_within_tolerance: "0.01 이내 단순성 tie-break",
      reference_fallback_no_challenger_passed: "기준 모델 유지",
    }[selectionEvidence.selectionReason] || "선정 근거 확인 필요";
    setText(dom["research-selection-status"], reasonLabel);
    setText(
      dom["research-selection-detail"],
      [
        researchRunStatus,
        selectionEvidence.runnerUp
          ? `runner-up ${modelForecastLabel(selectionEvidence.runnerUp)}`
          : "runner-up 없음",
      ].join(" · "),
    );
  }

  function renderModel() {
    const model = state.raw.model || {};
    const champion = model.champion;
    const championName = modelName(champion);
    const holdoutDiagnostic = isObject(model.holdout_diagnostic) ? model.holdout_diagnostic : null;
    const holdoutBestName = holdoutDiagnostic ? textValue(holdoutDiagnostic.best_model, "") : "";
    const lifecycle = isObject(model.lifecycle) ? model.lifecycle : {};
    const deployment = isObject(lifecycle.deployment) ? lifecycle.deployment.status : null;
    dom["champion-summary"].replaceChildren(
      createElement("span", null, deployment === "operating" ? "공식 운영 모델" : "로컬 선정 모델"),
      createElement("strong", null, modelForecastLabel(championName)),
    );
    renderModelRoles(model);
    renderModelEvidenceSummary(model, championName, holdoutDiagnostic);
    renderTransitionModels();
    const selection = firstValue(model, ["selection_period"]);
    const holdout = firstValue(model, ["holdout_period", "validation_period", "evaluation_period", "oos_period"]);
    const comparisonLabel = selection && holdout
      ? "선정 구간과 2023+ 진단 비교"
      : holdout
        ? "2023+ 진단 비교"
        : "모델 진단 비교";
    dom["model-caption"].textContent = `${comparisonLabel} · ${publicationSnapshotLabel()}`;

    dom["leaderboard-body"].replaceChildren();
    const rows = Array.isArray(model.leaderboard) ? model.leaderboard : [];
    renderModelLossChart(rows, championName, holdoutBestName);
    renderModelForecast();
    if (!rows.length) {
      const row = createElement("tr");
      const cell = createElement("td", null, "모델 비교 결과가 없습니다.");
      cell.colSpan = 8;
      row.append(cell);
      dom["leaderboard-body"].append(row);
      return;
    }

    rows.forEach((item, index) => {
      const rowData = isObject(item) ? item : { name: textValue(item) };
      const name = modelName(rowData);
      const row = createElement("tr");
      const isChampion = name === championName || Boolean(rowData.is_champion || rowData.champion);
      const isHoldoutBest = Boolean(holdoutBestName) && name === holdoutBestName;
      if (isChampion) row.classList.add("is-champion");
      if (isHoldoutBest) row.classList.add("is-holdout-best");
      const rank = firstValue(rowData, ["rank", "position"]) || index + 1;
      row.append(createElement("td", null, rank));
      const nameCell = createElement("td", null, modelForecastLabel(name));
      nameCell.classList.add("model-name-cell");
      if (modelForecastLabel(name) !== name) nameCell.append(createElement("small", "model-code", name));
      if (isChampion) nameCell.append(createElement("span", "champion-label", "선정"));
      if (isHoldoutBest) nameCell.append(createElement("span", "holdout-label", "2023+ 1위"));
      row.append(nameCell);
      row.append(
        createElement("td", null, formatNumber(metricValue(rowData, ["log_loss", "multiclass_log_loss"]), 4)),
        createElement("td", null, formatNumber(metricValue(rowData, ["brier", "brier_score"]), 4)),
        createElement("td", null, formatPercent(metricValue(rowData, ["accuracy", "acc"]))),
        createElement("td", null, formatPercent(metricValue(rowData, ["balanced_accuracy", "balanced_acc"]))),
        createElement("td", null, formatPercent(metricValue(rowData, ["macro_f1", "f1_macro"]))),
        createElement("td", null, formatPercent(metricValue(rowData, ["transition_recall", "regime_change_recall"]))),
      );
      dom["leaderboard-body"].append(row);
    });
  }

  function renderModelEvidenceSummary(model, championName, holdoutDiagnostic) {
    const container = dom["model-evidence-summary"];
    container.replaceChildren();
    if (!isV5Payload()) {
      container.hidden = true;
      dom["research-evidence"].hidden = true;
      return;
    }
    dom["research-evidence"].hidden = false;
    const appendEvidence = (label, value, className = "") => {
      const item = createElement("div", `model-evidence-item ${className}`.trim());
      item.append(createElement("span", null, label), createElement("strong", null, value));
      container.append(item);
    };
    const diagnostics = Array.isArray(model.selection_diagnostics) ? model.selection_diagnostics : [];
    const selection = firstValue(model, ["selection_period"]);
    const holdout = firstValue(model, ["holdout_period", "validation_period", "evaluation_period", "oos_period"]);
    if (selection || holdout) {
      appendEvidence(
        "검증 구간",
        [selection ? `선정 ${textValue(selection)}` : null, holdout ? `2023+ ${textValue(holdout)}` : null]
          .filter(Boolean)
          .join(" · "),
      );
    }
    const championDiagnostic = diagnostics.find(
      (row) => isObject(row) && modelName(row.model) === championName,
    );
    if (isObject(championDiagnostic)) {
      const referenceName = modelName(championDiagnostic.reference_model);
      const retainedReference = referenceName === championName;
      appendEvidence(
        "공식 모델 선정",
        retainedReference
          ? `${modelForecastLabel(championName)} · 비교 기준 모델 유지`
          : `${modelForecastLabel(championName)} · Log loss 개선 ${formatNumber(championDiagnostic.absolute_log_loss_improvement, 4)} · 기준 ${formatNumber(championDiagnostic.minimum_log_loss_improvement, 4)} · ${championDiagnostic.gate_passed === true ? "통과" : "미통과"}`,
        championDiagnostic.gate_passed === true ? "is-ok" : "is-review",
      );
    }
    if (isObject(holdoutDiagnostic) && holdoutDiagnostic.applicable === true) {
      appendEvidence(
        "2023+ 진단",
        `${championName} ${formatNumber(holdoutDiagnostic.champion_rank, 0)}/${formatNumber(holdoutDiagnostic.model_count, 0)}위 · ${textValue(holdoutDiagnostic.best_model)} 대비 Log loss +${formatNumber(holdoutDiagnostic.absolute_regret, 4)}`,
        holdoutDiagnostic.status === "ok" ? "is-ok" : "is-review",
      );
    }
    if (isObject(state.comparisonSummary) && state.comparisonSummary.exactParity === true) {
      appendEvidence(
        "V4 기준 비교",
        `공통 OOS ${formatNumber(state.comparisonSummary.commonKeys, 0)}개 · Markov 확률 완전 일치`,
      );
    }
    const featureQuality = isObject(model.feature_quality_artifact)
      ? model.feature_quality_artifact
      : null;
    if (
      featureQuality
      && Number.isInteger(featureQuality.feature_count)
      && Number.isInteger(featureQuality.warning_feature_count)
      && Number.isInteger(featureQuality.unavailable_feature_count)
    ) {
      appendEvidence(
        "입력 피처 품질",
        `${formatNumber(featureQuality.feature_count, 0)}개 · 경고 ${formatNumber(featureQuality.warning_feature_count, 0)} · 사용 불가 ${formatNumber(featureQuality.unavailable_feature_count, 0)}`,
        featureQuality.warning_feature_count > 0 || featureQuality.unavailable_feature_count > 0
          ? "is-review"
          : "is-ok",
      );
    }
    container.hidden = false;
  }

  function sourceStatus(source) {
    return normalizeStatus(firstValue(source, ["status", "health", "quality_status"]));
  }

  function sourceRightsLabel(source) {
    const value = String(firstValue(source, ["license_class", "license", "rights", "usage_scope"]) || "")
      .trim()
      .toLowerCase();
    if (value === "private_noncommercial") return "개인 · 비상업";
    if (value.startsWith("user_confirmed")) return "사용자 확인";
    if (value === "federal_reserve_board_public_domain_citation_requested") return "미 연준 공개 · 출처 표기";
    if (value === "rights_unconfirmed") return "공개 미확인";
    if (value === "license_blocked") return "이용 차단";
    return textValue(firstValue(source, ["license_class", "license", "rights", "usage_scope"]), "미기재");
  }

  function renderSources() {
    const sources = state.raw.sources || [];
    dom["source-health-body"].replaceChildren();
    if (!sources.length) {
      const row = createElement("tr");
      const cell = createElement("td", null, "소스 메타데이터가 없습니다.");
      cell.colSpan = 5;
      row.append(cell);
      dom["source-health-body"].append(row);
      return;
    }

    for (const source of sources) {
      const rowData = isObject(source) ? source : { name: textValue(source) };
      const row = createElement("tr");
      const nameCell = createElement("td", "source-name", textValue(firstValue(rowData, ["name", "provider", "source", "id"]), "이름 없음"));
      const identifier = firstValue(rowData, ["series_id", "dataset", "endpoint", "id"]);
      if (identifier) nameCell.append(createElement("span", "source-detail", identifier));
      row.append(nameCell);
      const statusCell = createElement("td");
      const badge = createElement("span");
      const rawStatus = firstValue(rowData, ["status", "health", "quality_status"]);
      setStatusBadge(badge, sourceStatus(rowData), healthLabel(rawStatus));
      statusCell.append(badge);
      row.append(statusCell);
      row.append(createElement("td", null, formatDateTime(firstValue(rowData, ["available_at", "data_as_of", "updated_at", "retrieved_at"]))));
      const coverage = [firstValue(rowData, ["coverage", "history", "window"]), firstValue(rowData, ["frequency", "cadence"])]
        .filter((value) => value !== null)
        .map((value) => textValue(value))
        .join(" · ");
      row.append(createElement("td", null, coverage || "—"));
      row.append(createElement("td", null, sourceRightsLabel(rowData)));
      dom["source-health-body"].append(row);
    }
  }

  function renderFeatureCatalog() {
    const catalog = state.raw.feature_catalog || [];
    dom["feature-catalog"].replaceChildren();
    if (!catalog.length) {
      dom["feature-catalog"].append(createElement("p", "empty-inline", "Feature catalog가 비어 있습니다."));
      return;
    }

    const groups = new Map();
    for (const item of catalog) {
      const entry = isObject(item) ? item : { name: textValue(item) };
      const groupName = textValue(firstValue(entry, ["category", "group", "block", "family"]), "기타");
      if (!groups.has(groupName)) groups.set(groupName, []);
      groups.get(groupName).push(entry);
    }

    for (const [groupName, items] of groups.entries()) {
      const card = createElement("section", "feature-group");
      card.append(createElement("h3", null, `${groupName} · ${items.length}개`));
      const context = [firstValue(items[0], ["frequency", "cadence"]), firstValue(items[0], ["source", "provider"])]
        .filter((value) => value !== null)
        .map((value) => textValue(value))
        .join(" · ");
      if (context) card.append(createElement("p", null, context));
      const tags = createElement("div", "feature-tags");
      for (const item of items.slice(0, 18)) {
        const name = textValue(firstValue(item, ["label", "name", "feature", "id", "series_id"]), "이름 없음");
        const tag = createElement("span", "feature-tag", name);
        tag.title = name;
        tags.append(tag);
      }
      if (items.length > 18) tags.append(createElement("span", "feature-tag", `+${items.length - 18}개`));
      card.append(tags);
      dom["feature-catalog"].append(card);
    }
  }

  function renderGlobalMetadata() {
    const meta = state.raw.meta || {};
    const dataAsOf = firstValue(meta, ["data_as_of", "dataAsOf", "cutoff_at"]);
    renderResultIdentity();
    renderHeaderDataAsOf(dataAsOf);
    const freshness = isObject(meta.freshness) ? meta.freshness : null;
    const currentFreshness = freshness
      ? displayFreshness(dataAsOf, freshness.maximum_age_days)
      : null;
    const freshnessText = dataAsOf ? formatDateTime(dataAsOf) : "기준 시점 미기재";
    setText(
      dom["source-freshness"],
      currentFreshness
        ? `${freshnessText} · ${currentFreshness.status === "stale" ? `지연 ${formatNumber(currentFreshness.age_days, 0)}일` : `최신 ${formatNumber(currentFreshness.age_days, 0)}일`}`
        : freshnessText,
    );

    const model = state.raw.model || {};
    const champion = model.champion;
    setText(dom["footer-model-version"], firstValue(isObject(champion) ? champion : {}, ["version", "model_version"]) || firstValue(model, ["version", "model_version"]) || "미기재");
    setText(dom["footer-schema-version"], firstValue(meta, ["schema_version", "schemaVersion"]) || "미기재");
    setText(dom["footer-generated-at"], formatDateTime(firstValue(meta, ["generated_at", "generatedAt"])));
    renderHeaderModelHealth();
  }

  function renderResultIdentity() {
    const identity = resultIdentity(state.raw);
    const element = dom["header-result-identity"];
    const meta = isObject(state.raw && state.raw.meta) ? state.raw.meta : {};
    const model = isObject(state.raw && state.raw.model) ? state.raw.model : {};
    const visibleLabel = identity.mode === "demo"
      ? "모의자료"
      : meta.publication_status === V5_PUBLICATION_STATUS
        ? "공개 운영"
        : model.selection_status === "selected_by_gate"
          ? "연구 후보"
          : identity.mode === "live"
            ? "실데이터"
            : identity.label;
    element.className = `result-identity-chip is-${identity.mode}`;
    element.setAttribute("aria-label", identity.label);
    element.title = identity.label;
    setText(element, visibleLabel);
  }

  function renderHeaderDataAsOf(value) {
    const container = dom["header-data-as-of"];
    container.replaceChildren(
      createElement("span", null, value ? formatDateTime(value) : "미기재"),
    );
    if (isHistoricalSelection()) {
      container.append(createElement("span", "model-health-chip", "과거 조회"));
      return;
    }
    const freshness = state.raw && isObject(state.raw.meta) && isObject(state.raw.meta.freshness)
      ? state.raw.meta.freshness
      : null;
    if (!isV5Payload() || !freshness) return;
    const currentFreshness = displayFreshness(value, freshness.maximum_age_days);
    if (!currentFreshness) return;
    const stale = currentFreshness.status === "stale";
    container.append(
      createElement(
        "span",
        `model-health-chip ${stale ? "is-review" : "is-ok"}`,
        stale ? `지연 ${formatNumber(currentFreshness.age_days, 0)}일` : "최신",
      ),
    );
  }

  function renderHeaderModelHealth() {
    const model = state.raw && isObject(state.raw.model) ? state.raw.model : {};
    const selection = state.raw && isObject(state.raw.selection) ? state.raw.selection : {};
    const champion = isV5Payload()
      ? modelName(selection.operating_champion || model.champion)
      : modelName(model.champion);
    const championLabel = modelForecastLabel(champion);
    dom["header-model-health"].replaceChildren(createElement("span", null, championLabel));
    if (isV5Payload() && isObject(model.model_health)) {
      const lifecycle = isObject(model.lifecycle) ? model.lifecycle : {};
      const deployment = isObject(lifecycle.deployment) ? lifecycle.deployment.status : null;
      const review = model.model_health.status === "review_due";
      const label = model.model_health.status === "review_due"
        ? "검토 필요"
        : deployment === "operating"
          ? "공식"
          : "공식 유지";
      const chip = createElement("span", `model-health-chip ${review ? "is-review" : "is-ok"}`, label);
      dom["header-model-health"].append(chip);
      const reasons = modelHealthReasonLabels(model.model_health.reasons);
      dom["header-model-health"].setAttribute(
        "aria-label",
        [championLabel, label, ...reasons].join(" · "),
      );
    }
  }

  function renderStaticSections() {
    renderGlobalMetadata();
    renderContractOverview();
    renderModel();
    renderSources();
    renderFeatureCatalog();
    renderConditionalStats();
  }

  async function loadData() {
    showAppState("loading");
    dom["header-result-identity"].className = "result-identity-chip is-loading";
    setText(dom["header-result-identity"], "결과 상태 불러오는 중");
    setText(dom["header-analysis-date"], "불러오는 중");
    setText(dom["header-data-as-of"], "불러오는 중");
    setText(dom["header-model-health"], "불러오는 중");

    try {
      const response = await fetch(DATA_URL, {
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error(`결과 파일 요청이 실패했습니다 (HTTP ${response.status}).`);
      const payloadText = await response.text();
      const payload = JSON.parse(payloadText);
      const validation = validatePayload(payload);
      if (validation.errors.length) throw new DataContractError(validation.errors);

      state.raw = payload;
      applyPayloadStateTheme(payload);
      [state.comparisonSummary, state.selectionFamilyAudit] = await Promise.all([
        loadV5ComparisonSummary(payload, payloadText),
        loadSelectionFamilyAudit(payload),
      ]);
      state.weekly = validation.weekly;
      state.validationWarnings = validation.warnings;
      state.comparisonModel = operatingChampionName(payload);
      state.outcomeBasis = "observed";
      state.preferredHistoryWindow = dom["history-window"].value === "all"
        ? "all"
        : Number(dom["history-window"].value);
      state.historyWindow = state.preferredHistoryWindow;

      if (!state.weekly.length) {
        renderGlobalMetadata();
        setText(dom["header-analysis-date"], "결과 없음");
        showAppState("empty");
        return;
      }

      populateDateControls();
      renderStaticSections();
      showDashboard();
      selectWeek(state.weekly.length - 1, false);
    } catch (error) {
      const detail = error instanceof DataContractError
        ? `데이터 계약 오류: ${error.messages.slice(0, 4).join(" ")}${error.messages.length > 4 ? ` 외 ${error.messages.length - 4}건` : ""}`
        : error instanceof SyntaxError
          ? "결과 파일이 올바른 JSON이 아닙니다."
          : textValue(error && error.message, "데이터 요청 중 알 수 없는 오류가 발생했습니다.");
      showAppState("error", "국면 결과를 표시할 수 없습니다", detail);
      dom["header-result-identity"].className = "result-identity-chip is-unknown";
      setText(dom["header-result-identity"], "결과 유형 확인 필요");
      setText(dom["header-analysis-date"], "사용 불가");
      setText(dom["header-data-as-of"], "사용 불가");
      setText(dom["header-model-health"], "사용 불가");
    }
  }

  function init() {
    initializeDom();
    setupTheme();
    revealActiveProjectLink();
    window.addEventListener("load", revealActiveProjectLink, { once: true });
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(revealActiveProjectLink);
    }
    window.addEventListener("resize", revealActiveProjectLink);
    bindEvents();
    loadData();
  }

  const dashboardApi = Object.freeze({
    DATA_URL,
    V5_COMPARISON_URL,
    V5_SELECTION_FAMILY_AUDIT_URL,
    STATE_ORDER,
    finiteNumber,
    strictFiniteNumber,
    probability,
    strictProbability,
    isIsoDate,
    isoDateOffset,
    normalizeStatus,
    snapToPriorDate,
    resolveHistoryWindow,
    championSelectionEvidence,
    validatePayload,
    resultIdentity,
    stateMeta,
    forecastAvailability,
    forecastSurfacePolicy,
    applyExpiredForecastDomState,
    validateV5ComparisonSummary,
    validateSelectionFamilyAuditSemantics,
    validateSelectionFamilyAudit,
    loadSelectionFamilyAudit,
    selectionEvidenceForDisplay,
    fxStatusLabel,
    displayFreshness,
    currentMeasureKind,
    getCurrentMeasure,
    extractCurrentStrength,
    forecastComparisonModels,
    forecastForWeek,
    oneWeekDepartureProbability,
    historyMeasureForWeek,
    historyStateForWeek,
    actualNextWeekForWeek,
    forecastEntropyForWeek,
    modelConditionedAssetRows,
    modelConditionedAssetRowsComplete,
    conditionalStatsRowsForBasis,
  });

  if (typeof window !== "undefined") window.__REGIME_DASHBOARD__ = dashboardApi;
  if (typeof module !== "undefined" && module.exports) module.exports = dashboardApi;
  if (typeof window !== "undefined" && typeof document !== "undefined") init();
})();
