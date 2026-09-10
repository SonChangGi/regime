/* Forecast comparison views are independent of the archived publication model. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.REGIME_ENHANCEMENTS = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  "use strict";
  const STATES = ["risk_on", "transition", "risk_off"];
  const STATE_LABELS = { risk_on: "위험 선호", transition: "전환", risk_off: "위험 회피" };
  const LABELS = {
    causal_dynamic_ensemble: "운영 앙상블", recency_weighted_xgboost_208w: "최근 가중 XGBoost",
    markov: "Markov", persistence: "현재 상태 유지", boundary_filtered_history: "경계 · 과거 잔차",
    boundary_student_t: "경계 · Student-t", boundary_asymmetric_ewma: "경계 · 비대칭 변동성",
    directional_duration_hazard: "방향·기간", markov_duration_path_baseline: "Markov 경로",
    direct_endpoint_ridge: "직접 국면 · Ridge", direct_endpoint_xgboost: "직접 국면 · XGBoost",
    evolving_boundary_ewma: "EWMA 경로", evolving_boundary_gjr_skewt: "GJR-GARCH 경로", markov_endpoint: "Markov · 기간 말",
    adaptive_shrink_after_coherence: "축소 보정", identity_after_coherence: "무보정·정합", frozen_stored_calibration: "발행 보정",
    weekly_threshold: "기존 주별 기준", episode_hysteresis_cooldown: "기존 구간 기준", bounded_episode_budget: "1주 만료 · 예산 제한",
    boundary_cleveland_inflation: "경계 + 물가", boundary_nyfed_growth: "경계 + 성장", boundary_reserve_elasticity: "경계 + 준비금",
    official: "기본 정의", wider_thresholds: "경계 넓힘", narrower_thresholds: "경계 좁힘", less_hysteresis: "완충 폭 축소", more_hysteresis: "완충 폭 확대", boundary_residual_raw: "경계 잔차 기준선",
    first_departure: "최초 이탈", risk_off_entry: "위험회피 진입", risk_off_occupancy: "위험회피 관측",
    endpoint_risk_on: "기간 말 위험선호", endpoint_transition: "기간 말 전환", endpoint_risk_off: "기간 말 위험회피", worsening: "기간 말 악화", recovery: "기간 말 회복",
    downside_state_only: "국면", downside_direct_market: "국면 + 시장 지표", downside_state_plus_direct_endpoint_ridge: "국면 + Ridge", downside_state_plus_evolving_boundary_gjr_skewt: "국면 + GJR 경로",
  };
  const state = { context: null, data: null, loading: null, generation: null, model: null, horizon: 1, labelHorizon: null, evaluationScope: "selected", alertModel: null, healthTarget: "worsening", healthScope: "current", error: null };
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const date = (value) => typeof value === "string" && Number.isFinite(Date.parse(value)) ? value.slice(0, 10) : null;
  const rows = (value) => Array.isArray(value) ? value.filter((row) => row && typeof row === "object") : [];
  const n = (value, digits = 3) => finite(value) ? new Intl.NumberFormat("ko-KR", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value) : "—";
  const pct = (value) => finite(value) ? `${n(value * 100, 1)}%` : "—";
  const signed = (value, digits = 4) => finite(value) ? `${value > 0 ? "+" : ""}${n(value, digits)}` : "—";
  const count = (value) => finite(value) ? n(value, 0) : "—";
  const label = (model, data = state.data) => LABELS[model] || data?.model_labels?.[model] || model.replaceAll("_", " ");
  const split = (value) => ({ selection: "선정", holdout: "과거 진단", retrospective_diagnostic: "과거 진단", retrospective_2023_2026: "2023년 이후", prospective: "발행 후", operational_oos: "발행 후" }[value] || value || "과거 진단");
  const key = (row) => `${date(row.origin_date)}|${row.model}|${row.horizon_weeks}`;

  function forecastRows(data) {
    const merged = new Map();
    for (const row of [...rows(data?.history), ...rows(data?.latest)]) merged.set(key(row), row);
    return [...merged.values()];
  }

  function validate(data, payload) {
    const errors = [];
    if (data?.schema_version !== "regime-forecast-enhancements/1") return ["예측 비교 형식이 다릅니다."];
    if (data.source_generation_id !== payload?.meta?.generation_id
      || Date.parse(data.data_as_of) !== Date.parse(payload?.meta?.data_as_of)) errors.push("예측 비교의 기준 시점이 다릅니다.");
    if (data.weekly_candidates && (data.weekly_candidates.schema_version !== "regime-weekly-candidates-summary/1"
      || data.weekly_candidates.source_generation_id !== data.source_generation_id
      || Date.parse(data.weekly_candidates.data_as_of) !== Date.parse(data.data_as_of))) errors.push("후보 주간 기록의 기준 시점이 다릅니다.");
    if (!Array.isArray(data.latest) || !Array.isArray(data.history) || !Array.isArray(data.model_metrics)) errors.push("예측 비교 표가 없습니다.");
    if (!forecastRows(data).length) errors.push("표시할 예측 비교가 없습니다.");
    for (const collection of [rows(data.history), rows(data.latest)]) {
      if (new Set(collection.map(key)).size !== collection.length) errors.push("예측 기준일이 중복되었습니다.");
    }
    for (const row of forecastRows(data)) {
      const probabilities = row.probabilities;
      if (!date(row.origin_date) || !date(row.target_date) || Date.parse(row.origin_date) > Date.parse(data.data_as_of)
        || typeof row.model !== "string" || !STATES.includes(row.current_state) || ![1, 4, 13].includes(row.horizon_weeks)
        || !probabilities || STATES.some((code) => !finite(probabilities[code]) || probabilities[code] < 0 || probabilities[code] > 1)
        || Math.abs(STATES.reduce((sum, code) => sum + probabilities[code], 0) - 1) > 1e-5
        || Date.parse(date(row.target_date)) - Date.parse(date(row.origin_date)) !== row.horizon_weeks * 7 * 86400000) errors.push("예측 확률 또는 대상 시점이 올바르지 않습니다.");
      for (const field of ["endpoint_change_probability", "worsening_probability", "recovery_probability", "first_departure_probability", "risk_off_entry_probability"]) {
        if (row[field] != null && (!finite(row[field]) || row[field] < 0 || row[field] > 1)) errors.push("사건 확률이 올바르지 않습니다.");
      }
    }
    for (const row of rows(data.model_metrics)) {
      if (typeof row.model !== "string" || ![1, 4, 13].includes(row.horizon_weeks)
        || !Number.isInteger(row.n_predictions) || row.n_predictions < 0) errors.push("평가 표본이 올바르지 않습니다.");
    }
    return [...new Set(errors)];
  }

  function view(data, { week, model, horizon = 1, modelOptions }) {
    const candidates = forecastRows(data);
    const models = modelOptions?.length ? modelOptions : [...new Set(candidates.map((row) => row.model))];
    const selectedModel = models.includes(model) ? model : models.includes(data.protocol?.display_model)
      ? data.protocol.display_model : models[0];
    const horizons = [...new Set(candidates.filter((row) => row.model === selectedModel).map((row) => row.horizon_weeks))].sort((a, b) => a - b);
    if (!horizons.length && models.includes(selectedModel)) horizons.push(1);
    const selectedHorizon = horizons.includes(Number(horizon)) ? Number(horizon) : horizons[0];
    const forecast = candidates.find((row) => date(row.origin_date) === date(week)
      && row.model === selectedModel && row.horizon_weeks === selectedHorizon) || null;
    const sameModelHorizon = (row) => row.model === selectedModel && (row.horizon_weeks == null || row.horizon_weeks === selectedHorizon);
    const sameOrigin = (row) => date(row.origin_date) === date(week);
    const evidenceRows = (block) => [...new Map([...rows(block?.history), ...rows(block?.latest)].map((row) => [`${key(row)}|${row.policy || ""}`, row])).values()];
    const metricRows = rows(data.model_metrics).filter((row) => row.horizon_weeks === selectedHorizon);
    return { models, horizons, model: selectedModel, horizon: selectedHorizon, forecast,
      metrics: metricRows,
      eventMetrics: rows(data.event_metrics).filter(sameModelHorizon),
      calibrationHealth: rows(data.calibration_health).filter(sameModelHorizon),
      economicIncremental: rows(data.economics?.incremental?.metrics).filter((row) => row.horizon_weeks === (selectedHorizon === 13 ? 13 : 4)),
      calibration: rows(data.calibration?.metrics).filter((row) => selectedHorizon === 1 || row.horizon_weeks === selectedHorizon),
      calibrationForecast: evidenceRows(data.calibration).filter((row) => sameOrigin(row) && (selectedHorizon === 1 || row.horizon_weeks === selectedHorizon)),
      alerts: selectedHorizon === 1 ? rows(data.alerts?.metrics).filter(sameModelHorizon) : [],
      alertForecasts: selectedHorizon === 1 ? evidenceRows(data.alerts).filter((row) => sameModelHorizon(row) && sameOrigin(row)) : [],
      economicHorizon: selectedHorizon === 13 ? 13 : 4,
      economics: rows(data.economics?.rows).filter((row) => (row.model == null || row.model === selectedModel) && row.horizon_weeks === (selectedHorizon === 13 ? 13 : 4)),
    };
  }

  function evaluate(data, { week, horizon, scope = "selected" }) {
    const cutoff = scope === "all" ? date(data.data_as_of) : date(week);
    const candidates = forecastRows(data).filter((row) => row.horizon_weeks === horizon
      && row.evaluation_split === "retrospective_diagnostic" && STATES.includes(row.actual)
      && date(row.target_date) <= cutoff);
    const baseline = new Map(candidates.filter((row) => row.model === "markov_endpoint").map((row) => [date(row.origin_date), row]));
    const score = (row) => ({ loss: -Math.log(Math.max(1e-15, row.probabilities[row.actual])),
      brier: STATES.reduce((sum, code) => sum + (row.probabilities[code] - Number(code === row.actual)) ** 2, 0) });
    return [...new Set(candidates.map((row) => row.model))].map((model) => {
      const matched = candidates.filter((row) => row.model === model && baseline.has(date(row.origin_date))
        && baseline.get(date(row.origin_date)).actual === row.actual).sort((a, b) => a.origin_date.localeCompare(b.origin_date));
      if (!matched.length) return null;
      const totals = matched.reduce((sum, row) => { const v = score(row), b = score(baseline.get(date(row.origin_date)));
        return { loss: sum.loss + v.loss, brier: sum.brier + v.brier, delta: sum.delta + v.loss - b.loss }; }, { loss: 0, brier: 0, delta: 0 });
      return { model, horizon_weeks: horizon, n_predictions: matched.length, evaluation_split: "retrospective_diagnostic",
        evaluation_start: date(matched[0].origin_date), evaluation_end: date(matched.at(-1).origin_date), evaluation_target_end: date(matched.at(-1).target_date),
        log_loss: totals.loss / matched.length, brier: totals.brier / matched.length, delta_log_loss: totals.delta / matched.length, baseline_model: "markov_endpoint" };
    }).filter(Boolean);
  }

  function appliedComparison(data, week) {
    const calibration = [...rows(data.calibration?.history), ...rows(data.calibration?.latest)];
    return [4, 13].map((horizon) => {
      const match = (model) => calibration.findLast((row) => date(row.origin_date) === week.date && row.horizon_weeks === horizon && row.model === model);
      const published = week.transition_risk?.[`${horizon}w`];
      const candidate = match("adaptive_shrink_after_coherence");
      const km = week.duration_context;
      return { horizon, target: published?.target_end || date(candidate?.target_date), applied: published?.probability,
        candidate: candidate?.probability, delta: finite(candidate?.probability) && finite(published?.probability) ? candidate.probability - published.probability : null,
        km: km?.departure_probability?.[`${horizon}w`], interval: km?.ci95?.departure_probability?.[`${horizon}w`] };
    });
  }

  function alertView(data, { week, model }) {
    const models = [...new Set(rows(data.alerts?.metrics).map((row) => row.model))];
    const selectedModel = models.includes(model) ? model : models.includes("causal_dynamic_ensemble") ? "causal_dynamic_ensemble" : models[0];
    const records = [...new Map([...rows(data.alerts?.history), ...rows(data.alerts?.latest)].map((row) => [`${key(row)}|${row.policy}`, row])).values()];
    const order = (a, b) => Number(b.policy === "bounded_episode_budget") - Number(a.policy === "bounded_episode_budget") || a.policy.localeCompare(b.policy);
    const metrics = rows(data.alerts?.metrics).filter((row) => row.model === selectedModel && row.evaluation_split === "retrospective_diagnostic").map((row) => {
      const origins = records.filter((r) => r.model === selectedModel && r.policy === row.policy && r.evaluation_split === row.evaluation_split && r.actual != null).map((r) => date(r.origin_date)).sort();
      return { ...row, evaluation_start: date(row.evaluation_start) || origins[0], evaluation_end: date(row.evaluation_end) || origins.at(-1) };
    }).sort(order);
    return { models, model: selectedModel, metrics, forecasts: records.filter((row) => row.model === selectedModel && date(row.origin_date) === date(week)).sort(order) };
  }

  function alertExpiry(row) {
    if (row.policy === "episode_hysteresis_cooldown") return "고정 만료 없음";
    return date(row.policy === "bounded_episode_budget" ? row.valid_until : row.target_date) || "—";
  }

  function el(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = String(text);
    return element;
  }
  function metrics(parent, items, className = "enhancement-metrics") {
    const list = el("dl", className);
    for (const [name, value, detail] of items) {
      const item = el("div"); item.append(el("dt", null, name), el("dd", null, value));
      if (detail) item.append(el("small", null, detail));
      list.append(item);
    }
    parent.append(list);
  }
  function table(parent, headings, values, caption) {
    if (!values.length) return;
    const wrap = el("div", "table-scroll enhancement-table"); wrap.tabIndex = 0; wrap.setAttribute("aria-label", caption);
    const grid = el("table"); grid.append(el("caption", "sr-only", caption));
    const head = el("thead"), tr = el("tr");
    for (const text of headings) { const cell = el("th", null, text); cell.scope = "col"; tr.append(cell); }
    head.append(tr); grid.append(head);
    const body = el("tbody");
    for (const valuesRow of values) { const row = el("tr"); for (const value of valuesRow) row.append(el("td", null, value)); body.append(row); }
    grid.append(body); wrap.append(grid); parent.append(wrap);
  }
  function disclosure(parent, title, render) {
    const details = el("details", "enhancement-disclosure"); details.append(el("summary", null, title));
    const body = el("div", "enhancement-detail-body"); render(body); details.append(body);
    parent.append(details); return details;
  }
  function selectControl(id, title, options, selected, onChange) {
    const field = el("label", "enhancement-field", title); field.htmlFor = id;
    const select = el("select"); select.id = id;
    for (const [value, text] of options) { const option = el("option", null, text); option.value = value; select.append(option); }
    select.value = String(selected); select.addEventListener("change", () => onChange(select.value));
    field.append(select); return field;
  }
  function readSelections(search) {
    const params = new URLSearchParams(search);
    return { model: params.get("forecast_model") || params.get("model"),
      horizon: [1, 4, 13].includes(Number(params.get("forecast_horizon"))) ? Number(params.get("forecast_horizon")) : 1,
      alertModel: params.get("forecast_alert_model"),
      healthScope: ["current", "all", ...STATES].includes(params.get("forecast_health_scope")) ? params.get("forecast_health_scope") : "current",
      healthTarget: ["worsening", "recovery", ...STATES.map((code) => `endpoint_${code}`)].includes(params.get("forecast_health_target")) ? params.get("forecast_health_target") : "worsening",
      labelHorizon: [1, 4, 13].includes(Number(params.get("forecast_label_horizon"))) ? Number(params.get("forecast_label_horizon")) : null,
      evaluationScope: params.get("forecast_evaluation") === "all" ? "all" : "selected" };
  }
  function selectionUrl(href, selection) {
    const url = new URL(href);
    for (const [key, value] of Object.entries({ forecast_model: selection.model, forecast_horizon: selection.horizon,
      forecast_alert_model: selection.alertModel, forecast_health_scope: selection.healthScope,
      forecast_health_target: selection.healthTarget, forecast_label_horizon: selection.labelHorizon })) {
      if (value != null) url.searchParams.set(key, String(value)); else url.searchParams.delete(key);
    }
    return url;
  }
  function syncUrl() {
    window.history.replaceState(null, "", selectionUrl(window.location.href, state));
  }
  function changeComparison(change) {
    if (state.context.onComparisonChange) state.context.onComparisonChange(change);
    else { if (change.model) state.model = change.model; if (change.horizon) state.horizon = change.horizon; render(); }
    syncUrl();
  }
  function renderApplied(container) {
    const week = state.context.week, next = state.context.officialForecast || week.next_week;
    const applied = el("section", "enhancement-applied");
    const heading = el("div", "enhancement-applied-heading");
    const historical = week.date !== date(state.data.data_as_of);
    heading.append(el("h3", null, "공식 예측"), el("span", "enhancement-status", historical ? "과거 OOS" : "운영"));
    applied.append(heading);
    if (next?.probabilities) {
      applied.append(el("p", "section-caption", `1주 국면 · ${date(next.date || next.target_date || week.next_week?.date)} 대상 · ${label(next.model)}`));
      metrics(applied, STATES.map((code) => [STATE_LABELS[code], pct(next.probabilities[code])]), "enhancement-events");
    }
    const comparisons = appliedComparison(state.data, week);
    if (comparisons.some((row) => finite(row.applied))) {
      applied.append(el("h4", null, "최초 이탈 확률"));
      table(applied, ["기간 · 대상", "공식", "연구 보정", "차이", "과거 KM · 95% 구간"], comparisons.map((row) => [
        `${row.horizon}주 · ${row.target || "—"}`, pct(row.applied), pct(row.candidate), finite(row.delta) ? `${signed(row.delta * 100, 1)}%p` : "—",
        `${pct(row.km)}${finite(row.interval?.lower) && finite(row.interval?.upper) ? ` · ${pct(row.interval.lower)}–${pct(row.interval.upper)}` : ""}`,
      ]), "같은 최초 이탈 사건의 공식 적용값과 보정 연구안 비교");
    }
    container.append(applied);
  }
  function renderAlerts(container) {
    const data = state.data;
    if (!rows(data.alerts?.metrics).length) return;
    const selected = alertView(data, { week: state.context.week.date, model: state.alertModel }); state.alertModel = selected.model;
    const section = el("section", "enhancement-alerts");
    const heading = el("div", "enhancement-applied-heading"); heading.append(el("h3", null, "1주 악화 경보"), el("span", "enhancement-status", "연구")); section.append(heading);
    section.append(selectControl("enhancement-alert-model", "경보 모델", selected.models.map((model) => [model, label(model)]), selected.model, (model) => { state.alertModel = model; render(); syncUrl(); }));
    const policy = data.alerts.policy || {};
    section.append(el("p", "section-caption", `신규 정책 · 유효 ${count(policy.validity_weeks || 1)}주 · 오경보 예산 연 ${count(policy.budget ?? policy.annual_false_week_budget ?? 4)}주 · 평가 창 최대 ${count(policy.calibration_window_weeks || policy.window_weeks || 156)}주`));
    const statuses = { issued: "경보", watching: "관찰", insufficient_history: "표본 대기", budget_limited: "예산 대기", cooldown: "재경보 대기", awaiting_reset: "신호 해제 대기", no_supported_policy: "유효 정책 대기", off: "경보 중단" };
    table(section, ["정책", "상태", "확률 / 기준", "유효 종료", "잔여 예산 / 평가 창"], selected.forecasts.map((row) => [label(row.policy), statuses[row.policy_status] || (row.alert ? "경보" : "관찰"), `${pct(row.probability)} / ${row.threshold > 1 ? "중단" : pct(row.threshold)}`, alertExpiry(row), finite(row.budget_capacity_remaining) ? `${count(row.budget_capacity_remaining)}주 / ${count(row.calibration_rows)}주` : "—"]), "선택 주의 경보 상태와 만료일");
    disclosure(section, "경보 평가 · 전체 기간", (body) => {
      const criteria = { bounded_episode_budget: "다음 주 1:1 · 단일 경보 선행", weekly_threshold: "다음 주 임계값 · 구간 선행", episode_hysteresis_cooldown: "활성 구간 내 악화 · 구간 선행" };
      table(body, ["정책 · 성공/선행 기준", "평가 기준일", "평가 주 / 악화", "포착 / 악화", "포착률", "오경보 주 / 년", "오경보 구간 / 년", "평균 선행"], selected.metrics.map((row) => [
        `${label(row.policy)} · ${criteria[row.policy] || "—"}`,
        `${row.evaluation_start || "—"}–${row.evaluation_end || "—"}`, `${count(row.n_predictions)} / ${count(row.event_count)}`, `${count(row.hit_count)} / ${count(row.event_count)}`, pct(row.recall),
        `${count(row.false_alert_weeks)} / ${n(row.false_alerts_per_year, 2)}${row.false_week_budget_exceeded ? " 초과" : ""}`,
        `${count(row.false_alert_episodes)} / ${n(row.false_alert_episodes_per_year, 2)}`, finite(row.mean_lead_weeks_detected_events) ? `${n(row.mean_lead_weeks_detected_events, 1)}주` : "—",
      ]), "정책별 유효기간과 악화 매칭 기준을 구분한 전체기간 평가");
    });
    container.append(section);
  }
  function renderWeeklyCandidates(container) {
    const value = state.data.weekly_candidates;
    if (!value || value.schema_version !== "regime-weekly-candidates-summary/1") return;
    disclosure(container, "후보 주간 기록", (body) => {
      const status = { deadline_missed: "발행 마감 경과", issued: "발행 완료", issued_local_preview: "로컬 발행 완료", already_issued: "발행 기록 유지", pending: "결과 대기", ready: "발행 준비" }[value.status || value.latest?.status] || "기록 확인";
      body.append(el("p", "section-caption", `${status} · 기준 ${date(value.latest_origin_at || value.latest?.origin_at) || "—"} · 갱신 ${date(value.as_of) || "—"}`));
      metrics(body, [["발행", `${count(value.issued_packets)}회`], ["결과 대기", `${count(value.pending_predictions)}건`], ["평가 완료", `${count(value.matured_predictions)}건`]], "enhancement-events");
      const completed = rows(value.models).filter((row) => row.n > 0);
      table(body, ["후보", "버전", "기간", "실제 평가", "Log loss", "Brier", "정확도"], completed.map((row) => [label(row.model), row.recipe_sha256?.slice(0, 8) || "기존 기록", `${row.horizon_weeks}주`, count(row.n), n(row.log_loss, 4), n(row.brier, 4), pct(row.accuracy)]), "실제 발행된 후보 예측만 평가한 주간 기록");
    });
  }
  function renderPrediction(container, selected) {
    const forecast = selected.forecast;
    const result = el("div", "enhancement-prediction");
    if (!forecast) { result.append(el("p", "section-caption", "선택 모델·기준일의 예측 없음")); container.append(result); return; }
    const predicted = STATES.reduce((best, code) => forecast.probabilities[code] > forecast.probabilities[best] ? code : best, STATES[0]);
    const heading = el("div", "enhancement-prediction-heading");
    heading.append(el("span", null, `${date(forecast.target_date).replaceAll("-", ".")} 시점`), el("strong", `state-text-${predicted}`, STATE_LABELS[predicted]));
    result.append(heading);
    const reference = selected.horizon === 1 ? (state.context.officialForecast || state.context.week.next_week)
      : forecastRows(state.data).find((row) => row.model === "markov_endpoint" && row.horizon_weeks === selected.horizon && date(row.origin_date) === state.context.week.date);
    const referenceName = selected.horizon === 1 ? "공식 1주 예측" : `Markov ${selected.horizon}주 기준선`;
    if (reference?.probabilities) result.append(el("p", "section-caption", `차이 · ${referenceName} 대비`));
    const probabilities = el("div", "enhancement-probabilities");
    for (const code of STATES) {
      const row = el("div", "enhancement-probability");
      const bar = el("span", "enhancement-probability-track"), fill = el("span", `probability-fill ${code}`);
      fill.style.width = `${forecast.probabilities[code] * 100}%`; bar.append(fill);
      row.append(el("span", null, STATE_LABELS[code]), bar, el("strong", null, pct(forecast.probabilities[code])));
      if (reference?.probabilities) { row.classList.add("has-comparison"); row.append(el("small", null, `${signed((forecast.probabilities[code] - reference.probabilities[code]) * 100, 1)}%p`)); }
      probabilities.append(row);
    }
    result.append(probabilities);
    const events = [["기간 말 악화", pct(forecast.worsening_probability)], ["기간 말 회복", pct(forecast.recovery_probability)]];
    if (finite(forecast.first_departure_probability)) events.push(["기간 내 최초 이탈", pct(forecast.first_departure_probability)]);
    if (finite(forecast.risk_off_entry_probability)) events.push([forecast.current_state === "risk_off" ? "위험회피 재진입" : "위험회피 진입", pct(forecast.risk_off_entry_probability)]);
    metrics(result, events, "enhancement-events");
    container.append(result);
  }
  function renderScorecard(container, selected) {
    const metric = selected.evaluation.find((row) => row.model === selected.model);
    const card = el("div", "enhancement-scorecard"); card.append(el("h3", null, "기간 말 평가"));
    if (state.context.modelEvaluation) card.append(selectControl("enhancement-evaluation-window", "평가 기간 · 선택 주까지", [[26, "26주"], [52, "52주"], [104, "104주"], ["all", "전체"]], state.context.evaluationWindow, (window) => changeComparison({ window })));
    if (!metric || !metric.n_predictions) { card.append(el("p", "section-caption", "확정된 공통 비교 표본 없음")); container.append(card); return; }
    card.append(el("p", "section-caption", `확정 ${metric.evaluation_start}–${metric.evaluation_end} · ${selected.horizon}주`));
    const baseline = metric.baseline_model ? label(metric.baseline_model) : "기준선";
    metrics(card, [["Log loss", n(metric.log_loss, 4)], ["기준선 대비", signed(metric.delta_log_loss), baseline], ["평가 표본", `${count(metric.n_predictions)}주`], ["Brier", n(metric.brier, 4)]]);
    const link = el("a", "enhancement-evaluation-link", "모델 비교·이력 →"); link.href = "#history";
    card.append(link); container.append(card);
  }
  function renderDetails(container, selected) {
    const data = state.data;
    const diagnosticHeading = el("p", "enhancement-evaluation-heading", `전체 기간 진단 · ${date(data.data_as_of)} 기준`); container.append(diagnosticHeading);
    if (selected.calibration.length) disclosure(container, "이탈 확률 보정", (body) => {
      table(body, ["보정", "기간", "평가", "표본", "Log loss ↓", "Brier ↓"], selected.calibration.map((row) => [label(row.model), `${row.horizon_weeks}주`, split(row.evaluation_split), count(row.n_predictions), n(row.log_loss, 4), n(row.brier, 4)]), "운영 이탈 확률의 보정 방법별 평가");
    });
    if (selected.economics.length) disclosure(container, `악화 신호 후 ${selected.economicHorizon}주 시장`, (body) => {
      const current = state.context.week.current.state;
      const conditioned = selected.economics.filter((row) => row.stratum === current);
      const values = conditioned.flatMap((row) => rows(row.risk_bins).filter((bin) => bin.weeks > 0).map((bin) => [
        split(row.split), `${pct(bin.lower_inclusive)}–${pct(bin.upper)}`, count(bin.weeks),
        pct(bin.downside_event_rate), pct(bin.mean_forward_return), pct(bin.mean_minimum_cumulative_return), pct(bin.mean_realized_volatility),
      ]));
      body.append(el("p", "section-caption", `${STATE_LABELS[current]} 국면${current === "risk_off" ? " · 악화확률 0 (최하위 국면)" : ""}`));
      table(body, ["평가", "1주 악화확률", "표본", "5% 이상 하락", "평균 수익", "평균 최저 수익", "연율 변동성"], values, "같은 현재 국면에서 악화 신호와 이후 시장 위험 비교");
    });
    if (selected.economicIncremental.length) disclosure(container, `${selected.economicHorizon}주 하락 예측 비교`, (body) => {
      body.append(el("p", "section-caption", "주간 종가 −5% 이하 (기준일 대비) · 국면 단독 예측 대비"));
      table(body, ["입력", "평가", "표본 / 사건", "Log loss", "차이", "차이 95% 구간"], selected.economicIncremental.map((row) => [label(row.model), split(row.evaluation_split), `${count(row.n_predictions)}/${count(row.event_count)}`, n(row.log_loss, 4), signed(row.delta_log_loss), `${signed(row.ci_low)}–${signed(row.ci_high)}`]), "국면 예측값이 실제 하락 위험 예측에 더하는 정보");
    });
    if (selected.eventMetrics.length) disclosure(container, "사건별 예측 평가", (body) => {
      table(body, ["사건", "평가", "표본 / 사건", "Log loss", "Brier", "Markov 대비"], selected.eventMetrics.map((row) => [label(row.target), split(row.evaluation_split), `${count(row.n_predictions)}/${count(row.event_count)}`, n(row.log_loss, 4), n(row.brier, 4), signed(row.delta_log_loss)]), "같은 사건과 기간의 예측 평가");
    });
    renderReliability(container, selected);
    renderRobustness(container, selected);
    renderInformation(container);
    disclosure(container, "평가 파일", (body) => {
      const nav = el("nav", "source-links");
      const link = el("a", null, "전체 예측·평가 JSON"); link.href = "./data/forecast-enhancements.json"; nav.append(link);
      for (const artifact of rows(data.provenance?.artifacts)) {
        if (typeof artifact.url === "string" && /^(\.\/data\/|https:\/\/)/.test(artifact.url)) { const a = el("a", null, artifact.label || "근거 파일"); a.href = artifact.url; nav.append(a); }
      }
      body.append(nav);
    });
  }
  function renderReliability(container, selected) {
    const pool = selected.calibrationHealth.filter((row) => row.evaluation_split === "retrospective_diagnostic" && row.window === "recent_52_mature_origins");
    if (!pool.length) return;
    const scopes = [...new Set(pool.map((row) => row.current_state))];
    const current = state.context.week.current.state;
    const scope = state.healthScope === "current" ? current : state.healthScope;
    const chosenScope = scopes.includes(scope) ? scope : "all";
    const available = pool.filter((row) => row.current_state === chosenScope);
    const targets = [...new Set(available.map((row) => row.target))];
    const chosenTarget = targets.includes(state.healthTarget) ? state.healthTarget : targets.includes("endpoint_risk_off") ? "endpoint_risk_off" : targets[0];
    const result = available.find((row) => row.target === chosenTarget);
    disclosure(container, "최근 확률 신뢰도", (body) => {
      body.append(el("p", "section-caption", "최근 확정 52주"));
      const controls = el("div", "enhancement-controls");
      controls.append(selectControl("enhancement-health-scope", "기준 국면", scopes.map((value) => [value, value === "all" ? "전체 상태" : STATE_LABELS[value]]), chosenScope, (value) => { state.healthScope = value; render(); syncUrl(); }),
        selectControl("enhancement-health-target", "예측 사건", targets.map((value) => [value, label(value)]), chosenTarget, (value) => { state.healthTarget = value; render(); syncUrl(); }));
      body.append(controls);
      if (!result) return;
      const status = { insufficient_events: "사건 표본 부족", review_due: "보정 점검", within_descriptive_thresholds: "기준 이내" }[result.status] || "평가";
      metrics(body, [["예측 / 실제 빈도", `${pct(result.mean_probability)} / ${pct(result.observed_rate)}`], ["표본 / 사건", `${count(result.n_predictions)} / ${count(result.event_count)}`], ["보정 오차", pct(result.binary_ece), status]], "enhancement-events");
      const chart = reliabilityChart(rows(result.reliability_bins));
      if (chart) body.append(chart);
      table(body, ["예측확률", "실제 빈도", "표본 / 사건"], rows(result.reliability_bins).map((bin) => [pct(bin.mean_probability), pct(bin.observed_rate), `${count(bin.n)}/${count(bin.events)}`]), "최근 확정 52주 안의 선택 국면·사건 확률 보정");
    });
  }
  function reliabilityChart(bins) {
    if (!bins.length) return null;
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 340 225"); svg.classList.add("enhancement-reliability-chart"); svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "예측확률과 실제 빈도. 대각선에 가까울수록 보정이 잘 맞습니다.");
    const add = (tag, attrs, text) => { const item = document.createElementNS(ns, tag); for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, String(value)); if (text) item.textContent = text; svg.append(item); return item; };
    const x = (p) => 48 + p * 260, y = (p) => 176 - p * 150;
    for (const tick of [0, .5, 1]) {
      add("line", { x1: x(0), x2: x(1), y1: y(tick), y2: y(tick), class: "reliability-grid" });
      add("text", { x: 40, y: y(tick) + 4, "text-anchor": "end" }, `${tick * 100}%`);
      add("text", { x: x(tick), y: 195, "text-anchor": "middle" }, `${tick * 100}%`);
    }
    add("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), class: "reliability-reference" });
    for (const bin of bins) {
      const dot = add("circle", { cx: x(bin.mean_probability), cy: y(bin.observed_rate), r: 5, class: "reliability-point" });
      const title = document.createElementNS(ns, "title"); title.textContent = `예측 ${pct(bin.mean_probability)}, 실제 ${pct(bin.observed_rate)}, ${bin.n}주`; dot.append(title);
    }
    add("text", { x: 178, y: 220, "text-anchor": "middle" }, "평균 예측확률");
    add("text", { x: 48, y: 13 }, "실제 빈도");
    return svg;
  }

  function labelSensitivityRows(robustness, horizon) {
    return rows(robustness.label_sensitivity?.rows || robustness.label_sensitivity).filter((row) => row.horizon_weeks === horizon);
  }
  function renderRobustness(container, selected) {
    const robustness = state.data.robustness || {};
    const applicable = (value) => rows(value).filter((row) => (row.model == null || row.model === selected.model) && (row.horizon_weeks == null || row.horizon_weeks === selected.horizon));
    const events = applicable(robustness.leave_one_event_out), blocks = applicable(robustness.block_sensitivity);
    const labelHorizon = state.labelHorizon || selected.horizon;
    const labels = labelSensitivityRows(robustness, labelHorizon);
    const competitive = applicable(robustness.label_sensitivity?.competitive_baselines?.rows || robustness.competitive_baselines?.rows);
    if (!events.length && !blocks.length && !labels.length && !competitive.length) return;
    disclosure(container, "조건별 안정성", (body) => {
      table(body, ["제외 사건", "평가", "제외 표본", "남은 표본", "기준선 대비"], events.map((row) => [date(row.episode_id) || row.event_label || row.event || row.excluded_event, split(row.evaluation_split), count(row.removed_origins), count(row.remaining_origins ?? row.n_predictions ?? row.n), signed(row.delta_log_loss)]), "사건 하나씩 제외한 모델 점수");
      table(body, ["평가", "블록 길이", "확률 오차 차이", "95% 하한", "95% 상한"], blocks.map((row) => [split(row.evaluation_split), `${count(row.block_weeks ?? row.block_length)}주`, signed(row.delta_log_loss ?? row.mean_difference), signed(row.ci_low ?? row.ci_lower ?? row.lower), signed(row.ci_high ?? row.ci_upper ?? row.upper)]), "시계열 블록 길이별 점수 차이");
      if (competitive.length) {
        body.append(el("h4", null, `${selected.horizon}주 기준선 비교`));
        table(body, ["모델 / 기준선", "기간", "평가", "블록", "표본", "Δ Log loss", "차이 95% 구간"], competitive.map((row) => [`${label(row.model)} / ${label(row.baseline_model)}`, `${row.horizon_weeks}주`, split(row.evaluation_split), `${row.block_weeks}주`, count(row.n_predictions), signed(row.delta_log_loss), `${signed(row.ci_low)}–${signed(row.ci_high)}`]), "공식 국면 정의에서 동일 표본의 기존 기준선 비교");
      }
      if (labels.length) {
        body.append(el("h4", null, "국면 정의별 비교"));
        body.append(selectControl("enhancement-label-horizon", "국면 정의 평가기간", [[1, "1주"], [4, "4주"], [13, "13주"]], labelHorizon, (value) => { state.labelHorizon = Number(value); render(); syncUrl(); }));
      }
      table(body, ["국면 정의", "모델", "기간", "평가", "표본", "Log loss", "기준선 대비"], labels.map((row) => [row.spec_label || label(row.spec_id), label(row.model), `${row.horizon_weeks}주`, split(row.evaluation_split), count(row.n_predictions ?? row.n), n(row.log_loss, 4), signed(row.delta_log_loss)]), `국면 정의별 ${labelHorizon}주 예측 안정성`);
      table(body, ["국면 정의", "평가", "표본", "기본 정의와 일치", "전환 시점 일치"], rows(robustness.label_sensitivity?.stability).map((row) => [label(row.spec_id), split(row.evaluation_split), count(row.n_predictions), pct(row.state_agreement), pct(row.transition_jaccard)]), "국면 정의를 바꿨을 때 판정과 전환 시점 일치");
    });
  }
  function renderInformation(container) {
    const info = state.data.additional_information;
    const sources = rows(info?.sources);
    if (!sources.length) return;
    disclosure(container, "추가 정보", (body) => {
      const grid = el("div", "enhancement-source-grid");
      const features = rows(info.features);
      for (const source of sources) {
        const card = el("article", "enhancement-source-card");
        const heading = el("h4", null, source.label || source.id);
        if (typeof source.url === "string" && /^https:\/\//.test(source.url)) {
          const link = el("a", null, source.label || source.id); link.href = source.url; heading.replaceChildren(link);
        }
        card.append(heading);
        if (source.unit) card.append(el("p", "enhancement-source-unit", source.unit));
        const entries = Object.entries(source.latest || {}).filter(([, value]) => finite(value));
        for (const [id, value] of entries.slice(0, 4)) {
          const feature = features.find((item) => item.id === id);
          const unit = feature?.unit || source.unit;
          const display = ["ratio", "fraction", "probability"].includes(unit) ? pct(value) : `${n(value, 2)}${unit && unit !== source.unit ? ` ${unit}` : ""}`;
          const line = el("div", "enhancement-source-value"); line.append(el("span", null, feature?.label || id.replaceAll("_", " ")), el("strong", null, display)); card.append(line);
        }
        const matched = rows(info.evaluation?.rows).filter((row) => row.source === source.id && row.n > 0);
        const diagnostic = matched.filter((row) => row.evaluation_split !== "selection");
        const comparisonRows = diagnostic.length ? diagnostic : matched;
        card.append(el("p", "section-caption", `관측 ${date(source.observed_at) || "—"} · 확보 ${date(source.available_at) || "—"}`));
        card.append(el("small", null, comparisonRows.length ? `${diagnostic.length ? "과거 진단" : "선정"} ${count(Math.max(...comparisonRows.map((row) => row.n)))}주${comparisonRows.some((row) => row.status === "current_vintage_sensitivity") ? " · 현재 빈티지" : ""}` : "축적 중"));
        grid.append(card);
      }
      body.append(grid);
      const comparisons = rows(info.evaluation?.rows).filter((row) => row.n > 0);
      table(body, ["정보", "모델", "기간", "평가", "비교 표본", "Δ Log loss"], comparisons.map((row) => [sources.find((source) => source.id === row.source)?.label || row.source, label(row.model), `${row.horizon_weeks}주`, `${split(row.evaluation_split)}${row.status === "current_vintage_sensitivity" ? " · 현재 빈티지" : ""}`, count(row.n), signed(row.delta_log_loss)]), "추가 정보의 동일 표본 비교");
    });
  }

  function renderBoundary() {
    const container = document.getElementById("label-boundary-brief");
    if (!container) return;
    const week = state.context.week;
    const supplied = week.label_explanation || week.label_context || week.boundary_context;
    const extra = rows(state.data?.label_explanations).find((row) => date(row.origin_date ?? row.date) === week.date);
    const value = supplied || extra;
    container.replaceChildren(); container.hidden = !value;
    if (!value) return;
    const items = [];
    if (finite(value.risk_score)) items.push(["국면 점수", n(value.risk_score)]);
    if (finite(value.distance_to_boundary)) items.push(["이탈 경계까지 거리", n(value.distance_to_boundary)]);
    if (finite(value.weekly_score_change ?? value.score_change)) items.push(["전주 대비", signed(value.weekly_score_change ?? value.score_change, 3)]);
    if (finite(value.active_boundary)) items.push(["현재 이탈 경계", n(value.active_boundary)]);
    else if (finite(value.lower_threshold) && finite(value.upper_threshold)) items.push(["판정 경계", `${n(value.lower_threshold)} / ${n(value.upper_threshold)}`]);
    if (!items.length) { container.hidden = true; return; }
    container.append(el("small", "section-caption", "SPY 국면 점수 = 추세 − 스트레스"));
    metrics(container, items, "enhancement-events");
    if (finite(value.known_zero_return_rolloff_change)) container.append(el("p", "section-caption", `다음 주 점수 변화 ${signed(value.known_zero_return_rolloff_change, 3)} · 가격 불변 가정`));
  }
  function render() {
    if (typeof document === "undefined" || !state.context) return;
    renderBoundary();
    const container = document.getElementById("forecast-enhancements");
    if (!container) return;
    const expanded = new Set([...container.querySelectorAll("details[open]")].map((item) => item.querySelector("summary")?.textContent));
    container.replaceChildren(); container.hidden = !state.data && !state.error;
    if (state.error) { container.append(el("p", "section-caption", state.error)); return; }
    if (!state.data) return;
    document.documentElement.dataset.forecastComparison = "ready";
    const selected = view(state.data, { week: state.context.week.date, model: state.model, horizon: state.horizon, modelOptions: state.context.modelOptions });
    if (state.context.selectedForecast && selected.model === state.context.comparisonModel && selected.horizon === state.context.modelHorizon) {
      const official = state.context.selectedForecast;
      const ranks = STATES.indexOf(state.context.week.current.state);
      selected.forecast = selected.forecast || { ...official, origin_date: state.context.week.date, target_date: official.target_date || official.date,
        current_state: state.context.week.current.state,
        worsening_probability: STATES.reduce((total, code, index) => total + (index > ranks ? official.probabilities[code] : 0), 0),
        recovery_probability: STATES.reduce((total, code, index) => total + (index < ranks ? official.probabilities[code] : 0), 0) };
    }
    const common = state.context.modelEvaluation;
    selected.evaluation = common?.evaluationScope ? common.leaderboard.map((row) => ({ ...row, model: row.name,
      evaluation_start: common.evaluationScope.completedStart, evaluation_end: common.evaluationScope.completedEnd,
      baseline_model: selected.horizon === 1 ? state.context.officialForecast?.model : "markov_endpoint",
      delta_log_loss: common.comparisons?.[row.name]?.logLossDifference }))
      : evaluate(state.data, { week: state.context.week.date, horizon: selected.horizon, scope: state.evaluationScope });
    state.model = selected.model; state.horizon = selected.horizon;
    const heading = el("div", "enhancement-heading");
    const copy = el("div"); copy.append(el("h2", null, "예측 비교"), el("p", "section-caption", `${state.context.week.date.replaceAll("-", ".")} 기준`));
    heading.append(copy); container.append(heading);
    renderApplied(container);
    const researchHeading = el("div", "enhancement-heading enhancement-research-heading");
    researchHeading.append(el("h3", null, `${selected.horizon}주 말 국면 · ${selected.model === (state.context.officialForecast || state.context.week.next_week)?.model ? "공식" : "연구"}`));
    const controls = el("div", "enhancement-controls");
    controls.append(selectControl("enhancement-model", "예측 모델", selected.models.map((model) => [model, label(model)]), selected.model, (model) => changeComparison({ model })),
      selectControl("enhancement-horizon", "예측 기간", (state.context.modelOptions ? [1, 4, 13] : selected.horizons).map((h) => [h, `${h}주`]), selected.horizon, (h) => changeComparison({ horizon: Number(h) })));
    researchHeading.append(controls); container.append(researchHeading);
    const primary = el("div", "enhancement-primary"); renderPrediction(primary, selected); renderScorecard(primary, selected); container.append(primary);
    renderAlerts(container);
    renderWeeklyCandidates(container);
    const details = el("div", "enhancement-details"); renderDetails(details, selected); container.append(details);
    for (const item of container.querySelectorAll("details")) if (expanded.has(item.querySelector("summary")?.textContent)) item.open = true;
    container.dataset.model = selected.model; container.dataset.horizon = String(selected.horizon); container.dataset.origin = state.context.week.date;
  }
  async function update(context) {
    state.context = context;
    const generation = context.payload?.meta?.generation_id;
    if (!generation) return;
    if (state.generation !== generation) {
      state.generation = generation; state.data = null; state.error = null; state.loading = null;
      if (typeof document !== "undefined") delete document.documentElement.dataset.forecastComparison;
      if (typeof window !== "undefined") Object.assign(state, readSelections(window.location.search));
      const requestedGeneration = generation;
      state.loading = (async () => {
        try {
          const response = await fetch("./data/forecast-enhancements.json", { cache: "no-store" });
          if (response.status === 404) return;
          if (!response.ok) throw new Error("예측 비교를 불러오지 못했습니다.");
          const data = await response.json();
          if (state.generation !== requestedGeneration) return;
          const errors = validate(data, state.context.payload);
          if (errors.length) throw new Error(errors[0]);
          state.data = data;
          state.context.onDataReady?.();
        } catch (error) { if (state.generation === requestedGeneration) state.error = error.message; }
        finally { if (state.generation === requestedGeneration) { state.loading = null; render(); } }
      })();
    }
    if (context.comparisonModel) state.model = context.comparisonModel;
    if ([1, 4, 13].includes(context.modelHorizon)) state.horizon = context.modelHorizon;
    render();
  }
  function getData(payload) {
    return state.data?.source_generation_id === payload?.meta?.generation_id && Date.parse(state.data?.data_as_of) === Date.parse(payload?.meta?.data_as_of) ? state.data : null;
  }
  return Object.freeze({ validate, forecastRows, view, evaluate, appliedComparison, alertView, alertExpiry, readSelections, selectionUrl, labelSensitivityRows, getData, update });
});
