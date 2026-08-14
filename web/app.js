(function () {
  "use strict";

  const DATA_URL = "./data/regime-results.json";
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
    height: 300,
    margin: Object.freeze({ top: 18, right: 18, bottom: 42, left: 54 }),
  });
  const STATE_META = Object.freeze({
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
    spy_close: "SPY 종가",
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
    weekly: [],
    selectedIndex: -1,
    historyWindow: 52,
    preferredHistoryWindow: 52,
    validationWarnings: [],
    chartHistory: [],
    chartPinnedDate: null,
    chartPreviewDate: null,
    transitionHorizon: 1,
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
    return new Intl.NumberFormat("ko-KR", {
      style: "percent",
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).format(number);
  }

  function formatNumber(value, digits = 2) {
    const number = finiteNumber(value);
    if (number === null) return "—";
    return new Intl.NumberFormat("ko-KR", {
      maximumFractionDigits: digits,
      minimumFractionDigits: 0,
    }).format(number);
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

  function humanizeKey(key) {
    return String(key)
      .replace(/_/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function stateMeta(code) {
    return STATE_META[code] || { label: textValue(code, "Unknown"), ko: "알 수 없음", symbol: "?", short: "?" };
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

    if (payload.meta.schema_version !== "1.0.0") errors.push("지원하지 않는 schema_version입니다.");
    if (!payload.meta.generated_at) errors.push("meta.generated_at이 없습니다.");
    if (!payload.meta.data_as_of) errors.push("meta.data_as_of가 없습니다.");
    if (!payload.meta.mode) errors.push("meta.mode가 없습니다.");
    if (payload.meta.timezone !== "America/New_York") {
      errors.push("meta.timezone은 America/New_York이어야 합니다.");
    }
    const stateIds = payload.states.map((item) => isObject(item) ? item.id : null);
    if (stateIds.length !== STATE_ORDER.length || stateIds.some((item, index) => item !== STATE_ORDER[index])) {
      errors.push("states는 risk_on, transition, risk_off 순서여야 합니다.");
    }
    if (!payload.model.champion) errors.push("model.champion이 없습니다.");
    if (payload.model.selection_status !== "provisional_predeployment") {
      errors.push("model.selection_status는 provisional_predeployment여야 합니다.");
    }
    if (!Array.isArray(payload.model.leaderboard)) errors.push("model.leaderboard 배열이 없습니다.");
    if (!payload.feature_catalog.length || payload.feature_catalog.some((item) => !isObject(item))) {
      errors.push("feature_catalog는 비어 있지 않은 객체 배열이어야 합니다.");
    }

    const declaredResultVersion = payload.meta.result_version;
    if (
      declaredResultVersion !== undefined
      && declaredResultVersion !== null
      && declaredResultVersion !== ""
      && declaredResultVersion !== V3_RESULT_VERSION
      && declaredResultVersion !== V4_RESULT_VERSION
    ) {
      errors.push(`지원하지 않는 meta.result_version입니다: ${textValue(declaredResultVersion)}`);
    }
    const isV3 = declaredResultVersion === V3_RESULT_VERSION;
    const isV4 = declaredResultVersion === V4_RESULT_VERSION;
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

      for (const horizon of ["current", "next_week"]) {
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
      if (!isObject(item.scores)) {
        errors.push(`${item.date} scores 객체가 없습니다.`);
      } else {
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

  function extractConfidence(result) {
    return probability(isObject(result) ? result.confidence : null);
  }

  function selectedWeek() {
    return state.weekly[state.selectedIndex] || null;
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
      "dashboard", "header-analysis-date", "header-data-as-of", "theme-toggle",
      "theme-toggle-text", "dashboard-subtitle", "date-form", "analysis-date", "week-select",
      "snap-note", "previous-week", "next-week", "latest-week", "history-window",
      "current-regime-card", "current-horizon", "current-regime-symbol", "current-regime-name",
      "current-regime-confidence", "current-probabilities", "current-entropy", "next-regime-card", "next-horizon",
      "next-regime-symbol", "next-regime-name", "next-regime-confidence",
      "next-probabilities", "next-entropy", "probability-shifts",
      "transition-card", "transition-value", "transition-value-label", "transition-meter",
      "transition-horizon-bars", "probability-chart",
      "probability-chart-wrap", "chart-tooltip", "history-caption",
      "chart-selection-readout", "chart-interaction-hint",
      "chart-readout-date", "chart-readout-risk-on", "chart-readout-transition", "chart-readout-risk-off",
      "history-data-body", "factor-axis", "factor-caption", "factor-scores", "regime-timeline", "timeline-start",
      "timeline-end", "top-drivers", "market-context", "champion-summary", "model-caption", "model-loss-caption",
      "model-loss-chart", "model-loss-axis", "leaderboard-body",
      "transition-model-section", "transition-model-caption", "transition-horizon-select",
      "transition-model-summary", "transition-leaderboard-caption", "transition-leaderboard-body",
      "research-notice-summary",
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

  function renderSelectedWeek() {
    const week = selectedWeek();
    if (!week) return;

    const cutoff = firstValue(week, ["data_as_of", "available_at", "cutoff_at"]) ||
      firstValue(state.raw.meta, ["data_as_of", "dataAsOf", "cutoff_at"]);
    setText(dom["header-analysis-date"], formatDate(week.date));
    setText(dom["header-data-as-of"], cutoff ? formatDateTime(cutoff) : "미기재");
    dom["dashboard-subtitle"].textContent = `${formatDate(week.date)} 관측 주${cutoff ? ` · 컷오프 ${formatDateTime(cutoff)}` : ""}`;

    renderRegime("current", week.current, week.date);
    renderRegime("next", week.next_week, firstValue(week.next_week, ["date", "target_date", "period_end"]));
    renderTransition(week);
    renderProbabilityShifts(week);
    renderHistory();
    renderTimeline();
    renderFactors(week.scores);
    renderDrivers(week.top_drivers);
    renderMarket(week.market);
  }

  function renderRegime(prefix, result, horizonDate) {
    const card = dom[`${prefix}-regime-card`];
    const code = isObject(result) ? result.state : null;
    const meta = stateMeta(code);
    card.classList.remove("state-risk_on", "state-transition", "state-risk_off");
    if (STATE_ORDER.includes(code)) card.classList.add(`state-${code}`);

    setText(dom[`${prefix}-regime-symbol`], meta.symbol);
    setText(dom[`${prefix}-regime-name`], `${meta.ko} · ${meta.label}`);
    const confidence = extractConfidence(result);
    setText(dom[`${prefix}-regime-confidence`], `확률 ${formatPercent(confidence)}`);
    dom[`${prefix}-horizon`].textContent = `${prefix === "current" ? "t" : "t+1"}${horizonDate ? ` · ${formatDate(horizonDate, false)}` : ""}`;
    const probabilityContainer = dom[`${prefix}-probabilities`];
    probabilityContainer.replaceChildren();
    for (const stateCode of STATE_ORDER) {
      const stateDefinition = STATE_META[stateCode];
      const value = getProbability(result, stateCode);
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
      row.setAttribute("aria-label", `${stateDefinition.ko} 확률 ${formatPercent(value)}`);
      row.append(label, track, display);
      probabilityContainer.append(row);
    }

    const entropy = finiteNumber(isObject(result) ? result.entropy : null);
    setText(dom[`${prefix}-entropy`], `${prefix === "current" ? "불확실성" : "예측 불확실성"} 엔트로피 ${formatNumber(entropy, 3)}`);
  }

  function renderProbabilityShifts(week) {
    const container = dom["probability-shifts"];
    container.replaceChildren();
    for (const code of STATE_ORDER) {
      const definition = STATE_META[code];
      const current = getProbability(week.current, code);
      const forecast = getProbability(week.next_week, code);
      const row = createElement("div", `probability-shift-row ${code}`);
      const label = createElement("span", "probability-shift-label");
      const marker = createElement("span", `state-dot ${code}`, definition.short);
      marker.setAttribute("aria-hidden", "true");
      label.append(marker, document.createTextNode(definition.label));

      const track = createElement("span", "probability-shift-track");
      if (current !== null && forecast !== null) {
        const start = Math.min(current, forecast) * 100;
        const distance = Math.abs(forecast - current) * 100;
        const connector = createElement("span", "probability-shift-connector");
        connector.style.left = `${start.toFixed(2)}%`;
        connector.style.width = `${distance.toFixed(2)}%`;
        const currentMarker = createElement("span", "probability-shift-point current");
        currentMarker.style.left = `${(current * 100).toFixed(2)}%`;
        const forecastMarker = createElement("span", "probability-shift-point forecast");
        forecastMarker.style.left = `${(forecast * 100).toFixed(2)}%`;
        track.append(connector, currentMarker, forecastMarker);
      }

      const values = createElement("span", "probability-shift-values");
      values.append(createElement("strong", null, `${formatPercent(current)} → ${formatPercent(forecast)}`));
      const delta = current === null || forecast === null ? null : (forecast - current) * 100;
      const deltaText = delta === null ? "변화 —" : `${delta > 0 ? "+" : ""}${formatNumber(delta, 1)}%p`;
      values.append(createElement("span", `probability-shift-delta ${delta !== null && delta > 0 ? "is-up" : delta !== null && delta < 0 ? "is-down" : ""}`, deltaText));
      row.setAttribute(
        "aria-label",
        `${definition.ko} 확률 현재 ${formatPercent(current)}, 다음 주 ${formatPercent(forecast)}, 변화 ${deltaText}`,
      );
      row.append(label, track, values);
      container.append(row);
    }
  }

  function renderTransition(week) {
    const riskByHorizon = isObject(week.transition_risk) ? week.transition_risk : null;
    const primaryRisk = riskByHorizon && isObject(riskByHorizon["1w"]) ? riskByHorizon["1w"] : null;
    const value = probability(primaryRisk ? primaryRisk.probability : week.transition_probability);
    setText(dom["transition-value"], formatPercent(value));
    setText(
      dom["transition-value-label"],
      riskByHorizon
        ? "1주 이탈 확률"
        : "다음 주 국면 변경 확률",
    );
    const fill = dom["transition-meter"].querySelector("span");
    dom["transition-meter"].setAttribute(
      "aria-label",
      riskByHorizon ? "향후 1주 안에 한 번 이상 현재 국면에서 이탈할 확률" : "다음 주 국면 변경 확률",
    );
    fill.style.width = value === null ? "0" : `${(value * 100).toFixed(2)}%`;
    if (value === null) {
      dom["transition-meter"].removeAttribute("aria-valuenow");
      dom["transition-meter"].setAttribute("aria-valuetext", "국면 변경 확률 결과 없음");
    } else {
      dom["transition-meter"].setAttribute("aria-valuenow", String(Math.round(value * 100)));
      dom["transition-meter"].setAttribute("aria-valuetext", formatPercent(value));
    }
    renderTransitionHorizons(riskByHorizon);
  }

  function renderTransitionHorizons(riskByHorizon) {
    const container = dom["transition-horizon-bars"];
    container.replaceChildren();
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
        createElement("span", null, `${horizon}주 이내`),
        createElement("strong", null, formatPercent(value)),
      );
      const meter = createElement("span", "transition-horizon-meter");
      const meterFill = createElement("span");
      meterFill.style.width = value === null ? "0" : `${(value * 100).toFixed(2)}%`;
      meter.append(meterFill);
      row.setAttribute(
        "aria-label",
        `향후 ${horizon}주 안에 한 번 이상 현재 국면에서 이탈할 확률 ${formatPercent(value)}`,
      );
      row.append(heading, meter);
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
    setText(dom["chart-readout-date"], week ? formatDate(week.date) : "—");
    for (const code of STATE_ORDER) {
      const readout = dom[`chart-readout-${code.replaceAll("_", "-")}`];
      if (readout) setText(readout, week ? formatPercent(getProbability(week.current, code)) : "—");
    }
    if (week) updateChartCursor(index);
    for (const point of dom["probability-chart"].querySelectorAll(".chart-point")) {
      point.classList.toggle("is-active", Boolean(week) && point.dataset.date === week.date);
    }

    if (dom["chart-selection-readout"]) {
      const summary = week
        ? `${formatDate(week.date)}. Risk-on ${formatPercent(getProbability(week.current, "risk_on"))}, Transition ${formatPercent(getProbability(week.current, "transition"))}, Risk-off ${formatPercent(getProbability(week.current, "risk_off"))}.`
        : "선택된 차트 날짜가 없습니다.";
      dom["chart-selection-readout"].setAttribute("aria-label", summary);
    }
  }

  function showChartTooltipForWeek(event, week) {
    const tooltip = dom["chart-tooltip"];
    tooltip.textContent = `${formatDate(week.date)} · Risk-on ${formatPercent(getProbability(week.current, "risk_on"))} · Transition ${formatPercent(getProbability(week.current, "transition"))} · Risk-off ${formatPercent(getProbability(week.current, "risk_off"))}`;
    tooltip.hidden = false;
    const wrapRect = dom["probability-chart-wrap"].getBoundingClientRect();
    const left = finiteNumber(event.clientX) === null ? 8 : event.clientX - wrapRect.left + 10;
    const top = finiteNumber(event.clientY) === null ? 8 : event.clientY - wrapRect.top - 42;
    const maxLeft = Math.max(8, wrapRect.width - 290);
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

  function renderHistory() {
    const history = selectedHistory();
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
      empty.textContent = "표시할 확률 히스토리가 없습니다.";
      svg.append(empty);
      return;
    }

    const { width, height, margin } = CHART_DIMENSIONS;
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const x = (index) => chartX(index, history.length);
    const y = (value) => margin.top + (1 - value) * plotHeight;

    for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
      const tickY = y(tick);
      svg.append(createSvg("line", { x1: margin.left, y1: tickY, x2: width - margin.right, y2: tickY, class: "chart-grid-line" }));
      const label = createSvg("text", { x: margin.left - 10, y: tickY + 4, "text-anchor": "end", class: "chart-axis-label" });
      label.textContent = `${Math.round(tick * 100)}%`;
      svg.append(label);
    }

    const desiredTicks = Math.min(7, history.length);
    const tickIndexes = new Set();
    for (let step = 0; step < desiredTicks; step += 1) {
      tickIndexes.add(desiredTicks === 1 ? 0 : Math.round((step / (desiredTicks - 1)) * (history.length - 1)));
    }
    for (const index of tickIndexes) {
      const label = createSvg("text", {
        x: x(index), y: height - 15, "text-anchor": index === 0 ? "start" : index === history.length - 1 ? "end" : "middle", class: "chart-date-label",
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
      y2: height - margin.bottom,
      class: "chart-selected-line",
      "data-chart-cursor": "true",
      "aria-hidden": "true",
    }));

    let validPointCount = 0;
    for (const code of STATE_ORDER) {
      const points = history.map((week, index) => {
        const value = getProbability(week.current, code);
        return { week, value, x: x(index), y: value === null ? null : y(value), index };
      });
      const pathData = makeLinePath(points);
      if (pathData) {
        svg.append(createSvg("path", { d: pathData, class: `chart-series ${code}` }));
      }

      for (const point of points) {
        if (point.value === null) continue;
        validPointCount += 1;
        const circle = createSvg("circle", {
          cx: point.x,
          cy: point.y,
          r: 2.5,
          class: `chart-point ${code}`,
          "data-date": point.week.date,
          focusable: "false",
          "aria-hidden": "true",
        });
        svg.append(circle);
      }
    }

    if (!validPointCount) {
      const empty = createSvg("text", { x: width / 2, y: height / 2, "text-anchor": "middle", class: "chart-axis-label" });
      empty.textContent = "이 기간에는 유효한 확률 값이 없습니다.";
      svg.append(empty);
    }

    const range = `${formatDate(history[0].date)}–${formatDate(history[history.length - 1].date)}`;
    dom["history-caption"].textContent = `${range} · ${history.length}주 관측 · 현재 국면 확률 · 0–100% 축`;
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
      cell.colSpan = 4;
      row.append(cell);
      dom["history-data-body"].append(row);
      return;
    }
    for (const week of history) {
      const row = createElement("tr");
      row.append(createElement("td", null, week.date));
      for (const code of STATE_ORDER) row.append(createElement("td", null, formatPercent(getProbability(week.current, code))));
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
      const code = isObject(week.current) ? week.current.state : null;
      const meta = stateMeta(code);
      const button = createElement("button", `timeline-cell ${STATE_ORDER.includes(code) ? code : "unknown"}`, week.date);
      button.type = "button";
      button.dataset.date = week.date;
      button.setAttribute("aria-label", `${formatDate(week.date)} 현재 국면 ${meta.ko}`);
      button.title = `${week.date} · ${meta.ko}`;
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
    dom["factor-caption"].textContent = `52주 표준점수 · ±${formatNumber(scale, 1)}`;

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

  function metricParts(key, raw) {
    if (isObject(raw)) {
      return {
        value: firstValue(raw, ["value", "score", "amount"]),
        unit: firstValue(raw, ["unit", "units"]),
        format: firstValue(raw, ["format", "display_format"]),
        label: firstValue(raw, ["label", "name"]) || MARKET_LABELS[key] || humanizeKey(key),
      };
    }
    return { value: raw, unit: null, format: null, label: MARKET_LABELS[key] || humanizeKey(key) };
  }

  function formatMetric(key, raw) {
    const metric = metricParts(key, raw);
    const number = finiteNumber(metric.value);
    if (number === null) return textValue(metric.value);
    const format = String(metric.format || "").toLowerCase();
    const unit = String(metric.unit || "");
    if (format === "percent") return formatSignedPercent(number);
    if (format === "probability") return formatPercent(number);
    if (unit === "%") return `${formatNumber(number, 2)}%`;
    if (format === "currency" || ["USD", "$"].includes(unit)) return `$${formatCompactNumber(number, 2)}`;
    if (/(^|_)(return|drawdown|vol|volatility)(_|$)/i.test(key) && Math.abs(number) <= 1) return formatSignedPercent(number);
    return `${formatCompactNumber(number, 3)}${unit ? ` ${unit}` : ""}`;
  }

  function renderMarket(market) {
    dom["market-context"].replaceChildren();
    if (!isObject(market) || !Object.keys(market).length) {
      const empty = createElement("div", "empty-inline");
      empty.append(createElement("dt", "sr-only", "데이터 상태"), createElement("dd", null, "이 관측 주에는 시장 맥락 값이 없습니다."));
      dom["market-context"].append(empty);
      return;
    }
    const entries = Object.entries(market).filter(([, value]) => !Array.isArray(value)).slice(0, 8);
    for (const [key, raw] of entries) {
      const parts = metricParts(key, raw);
      const wrapper = createElement("div");
      wrapper.append(createElement("dt", null, parts.label), createElement("dd", null, formatMetric(key, raw)));
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
      label.append(
        createElement("strong", null, item.name),
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

  function renderModel() {
    const model = state.raw.model || {};
    const champion = model.champion;
    const championName = modelName(champion);
    const holdoutDiagnostic = isObject(model.holdout_diagnostic) ? model.holdout_diagnostic : null;
    const holdoutBestName = holdoutDiagnostic ? textValue(holdoutDiagnostic.best_model, "") : "";
    dom["champion-summary"].replaceChildren(
      createElement("span", null, "선정 모델"),
      createElement("strong", null, championName),
    );
    renderTransitionModels();
    const selection = firstValue(model, ["selection_period"]);
    const holdout = firstValue(model, ["holdout_period", "validation_period", "evaluation_period", "oos_period"]);
    if (selection && holdout) {
      dom["model-caption"].textContent = `선정 구간 ${textValue(selection)} · 2023+ 진단 ${textValue(holdout)}`;
    } else if (holdout) {
      dom["model-caption"].textContent = `2023+ 진단 ${textValue(holdout)}`;
    }

    dom["leaderboard-body"].replaceChildren();
    const rows = Array.isArray(model.leaderboard) ? model.leaderboard : [];
    renderModelLossChart(rows, championName, holdoutBestName);
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
      const nameCell = createElement("td", null, name);
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

  function sourceStatus(source) {
    return normalizeStatus(firstValue(source, ["status", "health", "quality_status"]));
  }

  function sourceRightsLabel(source) {
    const value = String(firstValue(source, ["license_class", "license", "rights", "usage_scope"]) || "")
      .trim()
      .toLowerCase();
    if (value === "private_noncommercial") return "개인 · 비상업";
    if (value.startsWith("user_confirmed")) return "사용자 확인";
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
    setText(dom["header-data-as-of"], dataAsOf ? formatDateTime(dataAsOf) : "미기재");
    setText(dom["source-freshness"], dataAsOf ? formatDateTime(dataAsOf) : "기준 시점 미기재");

    const model = state.raw.model || {};
    const champion = model.champion;
    setText(dom["footer-model-version"], firstValue(isObject(champion) ? champion : {}, ["version", "model_version"]) || firstValue(model, ["version", "model_version"]) || "미기재");
    setText(dom["footer-schema-version"], firstValue(meta, ["schema_version", "schemaVersion"]) || "미기재");
    setText(dom["footer-generated-at"], formatDateTime(firstValue(meta, ["generated_at", "generatedAt"])));
  }

  function renderStaticSections() {
    renderGlobalMetadata();
    renderModel();
    renderSources();
    renderFeatureCatalog();
  }

  async function loadData() {
    showAppState("loading");
    setText(dom["header-analysis-date"], "불러오는 중");
    setText(dom["header-data-as-of"], "불러오는 중");

    try {
      const response = await fetch(DATA_URL, {
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error(`결과 파일 요청이 실패했습니다 (HTTP ${response.status}).`);
      const payload = await response.json();
      const validation = validatePayload(payload);
      if (validation.errors.length) throw new DataContractError(validation.errors);

      state.raw = payload;
      state.weekly = validation.weekly;
      state.validationWarnings = validation.warnings;
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
      setText(dom["header-analysis-date"], "사용 불가");
      setText(dom["header-data-as-of"], "사용 불가");
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
    validatePayload,
  });

  if (typeof window !== "undefined") window.__REGIME_DASHBOARD__ = dashboardApi;
  if (typeof module !== "undefined" && module.exports) module.exports = dashboardApi;
  if (typeof window !== "undefined" && typeof document !== "undefined") init();
})();
