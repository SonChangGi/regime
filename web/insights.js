/* Derived presentation logic. No network, DOM globals, or model fitting. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.REGIME_INSIGHTS = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const number = (value) => typeof value === "number" && Number.isFinite(value) ? value : null;
  function conditionalMetrics(row, weighting = "episode") {
    const value = row || {};
    const episode = weighting === "episode";
    return Object.freeze({
      mean: number(episode ? value.episode_equal_mean_return : value.mean_return),
      lower: number(episode ? value.episode_equal_mean_return_ci95_lower : value.mean_return_ci95_lower),
      upper: number(episode ? value.episode_equal_mean_return_ci95_upper : value.mean_return_ci95_upper),
      excess: number(episode ? value.episode_equal_excess_return : value.excess_mean_return),
      benchmark: number(episode ? value.episode_equal_unconditional_benchmark_mean_return : value.unconditional_benchmark_mean_return),
      weeklyMean: number(value.mean_return), weeklyExcess: number(value.excess_mean_return),
      sample: number(value.n), episodes: number(value.unique_episodes), nonOverlapping: number(value.non_overlapping_n),
    });
  }
  function portfolioAdjustment(holdings, target, value = 0) {
    if (!holdings || !target) return { error: "비중을 입력해 주세요." };
    const assets = [...new Set([...Object.keys(holdings), ...Object.keys(target)])];
    if (assets.some((asset) => number(holdings[asset] ?? 0) === null || (holdings[asset] ?? 0) < 0 || number(target[asset] ?? 0) === null || (target[asset] ?? 0) < 0)) return { error: "비중은 0 이상의 숫자로 입력해 주세요." };
    const sum = (map) => Object.values(map).reduce((total, weight) => total + weight, 0);
    if (Math.abs(sum(holdings) - 1) > 0.0001) return { error: "현재 비중 합계를 100%로 맞춰 주세요." };
    if (Math.abs(sum(target) - 1) > 0.0001) return { error: "목표 비중을 확인할 수 없습니다." };
    const rows = assets.map((asset) => ({ asset, current: holdings[asset] || 0, target: target[asset] || 0, delta: (target[asset] || 0) - (holdings[asset] || 0) })).map((row) => ({ ...row, amount: row.delta * value }));
    const buys = rows.filter((row) => row.asset !== "CASH").reduce((total, row) => total + Math.max(0, row.delta), 0);
    const sells = rows.filter((row) => row.asset !== "CASH").reduce((total, row) => total + Math.max(0, -row.delta), 0);
    return { error: null, rows, oneWay: Math.max(buys, sells), fullL1: buys + sells };
  }
  function forecastTiming(forecast, signal = {}, calendarEntry = null, now = Date.now()) {
    const target = Date.parse(forecast?.target_at);
    return Object.freeze({
      elapsed: !Number.isFinite(target) || now >= target || forecast?.status !== "active",
      nextIssueAt: signal.next_issue_at || signal.next_publication_at || null,
      nextEntryAt: signal.next_entry_at || signal.next_scheduled_entry_at || calendarEntry,
      confirmedEntry: Boolean(signal.next_entry_at || signal.next_scheduled_entry_at),
    });
  }
  // View contracts: dates belong to the selected origin; publication metadata
  // belongs to the latest release. A historical reconstruction has no issue time.
  function selectedForecastTiming(payload = {}, week = {}, now = Date.now()) {
    const latest = payload.forecast || {};
    const historical = week.date !== latest.origin_at?.slice(0, 10);
    const origin = historical ? (week.data_as_of || week.date) : latest.origin_at;
    const target = historical ? week.next_week?.date : latest.target_at;
    return Object.freeze({ origin, target, historical,
      issuedAt: historical ? null : (latest.issued_at || latest.published_at || latest.decision_at || null),
      remainingSeconds: historical ? null : Math.max(0, (Date.parse(target) - now) / 1000),
      latestOrigin: latest.origin_at, latestTarget: latest.target_at,
      latestIssuedAt: latest.issued_at || latest.published_at || latest.decision_at || null,
    });
  }
  function forecastDirections(current, probabilities) {
    const index = FORECAST_STATES.indexOf(current);
    if (index < 0 || !probabilities || FORECAST_STATES.some((key) => number(probabilities[key]) === null
      || probabilities[key] < 0 || probabilities[key] > 1)
      || Math.abs(FORECAST_STATES.reduce((sum, key) => sum + probabilities[key], 0) - 1) > 1e-5) return null;
    return Object.freeze({ stay: probabilities[current],
      worsening: FORECAST_STATES.slice(index + 1).reduce((sum, key) => sum + probabilities[key], 0),
      recovery: FORECAST_STATES.slice(0, index).reduce((sum, key) => sum + probabilities[key], 0) });
  }
  function transitionOutlook(week = {}) {
    return [4, 13].map((horizon) => {
      const key = `${horizon}w`, model = week.transition_risk?.[key], direction = week.directional_risk?.[key];
      return Object.freeze({ horizon, target: model?.target_end, model: model?.model,
        probability: number(model?.probability), fallback: model?.fallback === true,
        fallbackReason: model?.fallback_reason || "", noDeparture: number(direction?.no_departure),
        firstDestination: direction?.first_destination || null, directionModel: direction?.model,
        baseline: number(week.duration_context?.departure_probability?.[key]),
        baselineSupport: number(week.duration_context?.support?.horizon_at_risk?.[key]),
        baselineLower: number(week.duration_context?.ci95?.departure_probability?.[key]?.lower),
        baselineUpper: number(week.duration_context?.ci95?.departure_probability?.[key]?.upper),
      });
    });
  }
  function durationEstimate(duration = {}) {
    const median = number(duration.median_remaining_weeks), restricted = median === null;
    const field = restricted ? "restricted_mean_remaining_weeks" : "median_remaining_weeks";
    const ci = duration.ci95?.[field];
    return Object.freeze({ value: number(duration[field]), restricted,
      restriction: number(duration.restriction_weeks),
      lower: number(ci?.lower), upper: number(ci?.upper),
      completed: number(duration.completed_spells), censored: number(duration.censored_spells),
      supportedCompleted: number(duration.support?.completed_at_current_age),
      supportedAtRisk: number(duration.support?.at_risk_at_current_age) });
  }
  // Optional research extension. Absent evidence never borrows the first-
  // destination probability and is never extrapolated to a different origin.
  function validateForecastResearch(payload = {}) {
    const block = payload.research?.forecast_research;
    if (block === undefined) return [];
    const errors = [], prefix = "research.forecast_research";
    const validDate = (value) => typeof value === "string" && /^\d{4}-\d{2}-\d{2}T.*(Z|[+-]\d{2}:\d{2})$/.test(value) && Number.isFinite(Date.parse(value));
    const prob = (value) => number(value) !== null && value >= 0 && value <= 1;
    const vector = (value) => value && Object.keys(value).length === 3 && FORECAST_STATES.every((key) => prob(value[key]));
    const sum = (value) => FORECAST_STATES.reduce((total, key) => total + value[key], 0);
    if (block?.schema_version !== "regime-forecast-research/1" || !validDate(block.data_as_of)
      || Date.parse(block.data_as_of) !== Date.parse(payload.meta?.data_as_of)
      || !Array.isArray(block.models) || !block.models.length) return [`${prefix}: schema, cutoff or models invalid`];
    const ids = new Set(), actuals = new Map((payload.weekly || []).map((week) => [week.date, week.current?.state]));
    let referenceDates = null;
    for (const model of block.models) {
      if (typeof model?.id !== "string" || !model.id || ids.has(model.id) || !Array.isArray(model.history)) {
        errors.push(`${prefix}: duplicate or invalid model`); continue;
      }
      ids.add(model.id);
      const historyDates = model.history.map((row) => row?.origin_date);
      if (historyDates.some((date, index) => index > 0 && Date.parse(date) <= Date.parse(historyDates[index - 1]))
        || (referenceDates && JSON.stringify(historyDates) !== JSON.stringify(referenceDates))) errors.push(`${prefix}.${model.id}: history origins must be chronological and matched`);
      referenceDates = historyDates;
      const dates = new Set();
      for (const row of [...model.history, ...(model.latest ? [model.latest] : [])]) {
        if (!validDate(row?.origin_date) || Date.parse(row.origin_date) > Date.parse(block.data_as_of)
          || (dates.has(row.origin_date?.slice(0, 10)) && row !== model.latest) || !FORECAST_STATES.includes(row?.current_state)
          || (actuals.has(row.origin_date?.slice(0, 10)) && actuals.get(row.origin_date.slice(0, 10)) !== row.current_state)) {
          errors.push(`${prefix}.${model.id}: origin or state mismatch`); continue;
        }
        dates.add(row.origin_date.slice(0, 10));
        if (row.next_state && (!vector(row.next_state) || Math.abs(sum(row.next_state) - 1) > 1e-8)) errors.push(`${prefix}.${model.id}: invalid next state`);
        if (row.horizons && Object.keys(row.horizons).length === 0) continue; // one-week-only model
        if (!row.horizons || Object.keys(row.horizons).length !== 3) errors.push(`${prefix}.${model.id}: incomplete horizons`);
        let previousHit = 0, previousEntry = 0, previousDeparture = 0, previousFirst = null;
        for (const horizon of [1, 4, 13]) {
          const result = row.horizons?.[`${horizon}w`];
          if (!result || result.horizon_weeks !== horizon || !validDate(result.target_date)
            || Math.abs(Date.parse(result.target_date) - Date.parse(row.origin_date) - horizon * 7 * 86400000) > 3600000
            || Date.parse(result.target_date.slice(0, 10)) - Date.parse(row.origin_date.slice(0, 10)) !== horizon * 7 * 86400000
            || !vector(result.endpoint) || Math.abs(sum(result.endpoint) - 1) > 1e-5
            || !result.first_departure || Object.keys(result.first_departure).length !== 4
            || !FORECAST_STATES.every((key) => prob(result.first_departure[key])) || !prob(result.first_departure.no_departure)
            || Math.abs(sum(result.first_departure) + result.first_departure.no_departure - 1) > 1e-5
            || result.first_departure[row.current_state] !== 0 || !prob(result.any_risk_off_occupancy) || !prob(result.any_risk_off_entry)) {
            errors.push(`${prefix}.${model.id}.${horizon}w: invalid path probabilities`); continue;
          }
          const hit = result.any_risk_off_occupancy, entry = result.any_risk_off_entry, departure = 1 - result.first_departure.no_departure;
          if (hit + 1e-5 < previousHit || entry + 1e-5 < previousEntry || departure + 1e-5 < previousDeparture
            || result.first_departure.no_departure > result.endpoint[row.current_state] + 1e-5
            || (previousFirst && FORECAST_STATES.some((key) => result.first_departure[key] + 1e-5 < previousFirst[key]))
            || (horizon === 1 && row.next_state && FORECAST_STATES.some((key) => Math.abs(row.next_state[key] - result.endpoint[key]) > 1e-5))
            || entry > hit + 1e-5 || entry > departure + 1e-5
            || (row.current_state !== "risk_off" && Math.abs(entry - hit) > 1e-5)
            || (row.current_state === "risk_off" && horizon === 1 && entry > 1e-5)
            || hit + 1e-5 < result.endpoint.risk_off || hit + 1e-5 < result.first_departure.risk_off
            || (row.current_state !== "risk_off" && hit > departure + 1e-5)
            || (horizon === 1 && (Math.abs(hit - result.endpoint.risk_off) > 1e-5
              || FORECAST_STATES.some((key) => Math.abs(result.endpoint[key]
                - (key === row.current_state ? result.first_departure.no_departure : result.first_departure[key])) > 1e-5)))) {
            errors.push(`${prefix}.${model.id}.${horizon}w: inconsistent path events`);
          }
          previousHit = hit; previousEntry = entry; previousDeparture = departure;
          previousFirst = result.first_departure;
        }
      }
      if (model.latest && model.latest.origin_date !== block.data_as_of) errors.push(`${prefix}.${model.id}: latest cutoff mismatch`);
    }
    if (!ids.has(block.selected_model)) errors.push(`${prefix}: selected_model missing`);
    return errors;
  }
  function researchForecastRow(payload = {}, date, requestedModel = null) {
    const block = payload.research?.forecast_research;
    if (!block || validateForecastResearch(payload).length) return null;
    const model = block.models.find((item) => item.id === (requestedModel || block.selected_model));
    if (!model) return null;
    const row = [...(model.latest ? [model.latest] : []), ...model.history].find((item) => item.origin_date.slice(0, 10) === date);
    return row ? { ...row, model: model.id, label: model.label || model.id } : null;
  }
  function multistateForecastForWeek(payload = {}, date, requestedModel = null) {
    const row = researchForecastRow(payload, date, requestedModel);
    return row && [1, 4, 13].every((horizon) => row.horizons?.[`${horizon}w`])
      ? row : null;
  }
  function modelQuality(model = {}, selectedModel = null) {
    const nameOf = (value) => typeof value === "string" ? value : value?.name ?? value?.model ?? value?.id ?? null;
    const selected = selectedModel ?? nameOf(model.champion);
    const matches = (model.leaderboard || []).filter((row) => nameOf(row) === selected
      && [undefined, "holdout", "retrospective_diagnostic"].includes(row.evaluation_split));
    // Health summaries belong to the champion; never borrow them for another model.
    const row = selected && matches.length === 1 ? matches[0] : {};
    const metric = (key) => number(Object.hasOwn(row, key) ? row[key] : row.metrics?.[key]);
    const calibration = metric("calibration_error"), selectionCalibration = metric("selection_calibration_error");
    return Object.freeze({
      model: selected, available: Boolean(selected) && matches.length === 1,
      logLoss: metric("log_loss"), brier: metric("brier"), calibration, selectionCalibration,
      calibrationDrift: calibration !== null && selectionCalibration !== null ? calibration - selectionCalibration : null,
      recall: metric("transition_recall"), precision: metric("transition_precision"),
      captured: metric("on_time_departure_count"), events: metric("transition_event_count"),
      falseAlarms: metric("false_alarms_per_year"), falseAlarmCount: metric("false_alarm_count"),
      delay: metric("mean_detection_delay_forecast_weeks"), detected: metric("detected_event_count"), weeks: metric("n_predictions"),
      fallbackCount: metric("fallback_count"),
      worseningEvents: metric("worsening_event_count"), worseningCaptured: metric("on_time_worsening_count"),
      recoveryEvents: metric("recovery_event_count"), recoveryCaptured: metric("on_time_recovery_count"),
      researchCandidate: row.research_candidate === true,
    });
  }
  const FORECAST_RESEARCH_IDS = Object.freeze(["boundary_filtered_history", "boundary_student_t"]);
  const ENHANCEMENT_MODEL_IDS = Object.freeze(["boundary_filtered_history", "causal_dynamic_ensemble", "direct_endpoint_ridge", "direct_endpoint_xgboost", "directional_duration_hazard", "evolving_boundary_ewma", "evolving_boundary_gjr_skewt", "markov_endpoint"]);
  const FORECAST_STATES = Object.freeze(["risk_on", "transition", "risk_off"]);
  const enhancementIndexes = new WeakMap();
  function enhancementIndex(payload = {}) {
    const data = payload?.forecast_enhancements;
    if (!data || data.source_generation_id !== payload.meta?.generation_id || Date.parse(data.data_as_of) !== Date.parse(payload.meta?.data_as_of)) return null;
    if (!enhancementIndexes.has(data)) {
      const byKey = new Map([...(data.history || []), ...(data.latest || [])].map((row) => [`${row.origin_date.slice(0, 10)}|${row.model}|${row.horizon_weeks}`, row]));
      enhancementIndexes.set(data, { rows: [...byKey.values()], byKey });
    }
    return enhancementIndexes.get(data);
  }
  function enhancementForecastRows(payload = {}) { return enhancementIndex(payload)?.rows || []; }
  function forecastModelIds(payload = {}, horizon = 1) {
    const original = horizon === 1 ? [...(payload.model?.forecast_comparison?.models || []), ...forecastImprovementModels(payload).map((row) => row.id)] : [];
    return [...new Set([...original, ...enhancementForecastRows(payload).filter((row) => row.horizon_weeks === horizon).map((row) => row.model)])];
  }
  function enhancedForecastForWeek(payload, model, origin, horizon = 1) {
    const row = enhancementIndex(payload)?.byKey.get(`${origin}|${model}|${horizon}`);
    if (!row?.probabilities) return null;
    const probabilities = row.probabilities;
    const predicted = FORECAST_STATES.reduce((best, code) => probabilities[code] > probabilities[best] ? code : best, FORECAST_STATES[0]);
    return { model, date: row.target_date.slice(0,10), target_date: row.target_date, state: predicted, probabilities,
      confidence: probabilities[predicted], fallback: row.fallback === true, actual: row.actual, horizon_weeks: horizon,
      entropy: -FORECAST_STATES.reduce((sum, code) => sum + (probabilities[code] > 0 ? probabilities[code] * Math.log(probabilities[code]) : 0), 0) / Math.log(3), research_candidate: true };
  }
  function forecastImprovementModels(payload = {}) {
    const block = payload?.research?.forecast_improvement;
    return block?.schema_version === "regime-forecast-improvement/1" && Array.isArray(block.models)
      ? block.models.filter((model) => FORECAST_RESEARCH_IDS.includes(model?.id)) : [];
  }
  function forecastComparisonModel(payload = {}, horizon = 1) {
    const official = payload.model || {};
    const research = forecastImprovementModels(payload).map((model) => ({
      ...(model.metrics?.holdout || {}), name: model.id, evaluation_split: "holdout",
      selection_log_loss: model.metrics?.selection?.log_loss ?? null,
      selection_calibration_error: model.metrics?.selection?.calibration_error ?? null,
      research_candidate: true,
    }));
    const existing = horizon === 1 ? [...(official.leaderboard || []), ...research] : [];
    const names = new Set(existing.map((row) => row.name || row.model));
    const extra = (enhancementIndex(payload) ? payload.forecast_enhancements.model_metrics || [] : []).filter((row) => row.horizon_weeks === horizon && row.evaluation_split === "retrospective_diagnostic" && !names.has(row.model))
      .map((row) => ({ ...row, name: row.model, research_candidate: true }));
    return { ...official, leaderboard: [...existing, ...extra] };
  }
  function forecastEvaluation(payload = {}, { asOf, window = 52, horizon = 1 } = {}) {
    const nameOf = (value) => typeof value === "string" ? value : value?.name ?? value?.model ?? value?.id ?? null;
    const dateOf = (value) => typeof value === "string" && /^\d{4}-\d{2}-\d{2}/.test(value)
      && Number.isFinite(Date.parse(value)) ? value.slice(0, 10) : null;
    const weekly = [...(payload.weekly || [])].filter((row) => dateOf(row.date)).sort((a, b) => a.date.localeCompare(b.date));
    const cutoff = dateOf(asOf) || weekly.at(-1)?.date || null;
    const available = weekly.filter((row) => row.date <= cutoff);
    const count = [26, 52, 104].includes(Number(window)) ? Number(window) : 52;
    const origins = window === "all" ? available : available.slice(-count);
    const comparisonModel = forecastComparisonModel(payload, horizon);
    const research = forecastImprovementModels(payload);
    const officialNames = payload.model?.forecast_comparison?.models
      || (payload.model?.leaderboard || []).map(nameOf);
    const extra = enhancementForecastRows(payload).filter((row) => row.horizon_weeks === horizon);
    const names = [...new Set([...(horizon === 1 ? [...officialNames, ...research.map((row) => row.id)] : []), ...extra.map((row) => row.model)])].filter(Boolean);
    const extraByModel = new Map(names.map((model) => [model, new Map(extra.filter((row) => row.model === model).map((row) => [row.origin_date.slice(0,10), row]))]));
    const actualByDate = new Map(weekly.map((row) => [row.date, row.current?.state]));
    const researchByModel = new Map(research.map((model) => [model.id,
      new Map([...(model.history || []), model.latest].filter(Boolean).map((row) => [dateOf(row.origin_date), row]))]));
    const rowsByModel = new Map(names.map((name) => [name, []]));
    const scope = { start: origins[0]?.date || null, end: origins.at(-1)?.date || null,
      asOf: cutoff, originCount: origins.length, completedCount: 0, pendingCount: 0,
      excludedCount: 0, completedStart: null, completedEnd: null };
    for (const week of origins) {
      const current = FORECAST_STATES.indexOf(week.current?.state);
      const forecasts = names.map((name) => (horizon === 1 ? researchByModel.get(name)?.get(week.date)
        || (week.model_forecasts || []).find((item) => item.model === name) : null) || extraByModel.get(name)?.get(week.date));
      const targets = forecasts.map((row) => dateOf(row?.target_date ?? row?.date));
      const knownTargets = targets.filter(Boolean);
      if (knownTargets.length && knownTargets.every((target) => target > cutoff)) { scope.pendingCount += 1; continue; }
      const candidates = forecasts.map((row) => {
        const target = dateOf(row?.target_date ?? row?.date);
        const raw = FORECAST_STATES.map((state) => number(row?.probabilities?.[state]));
        if (!target || target <= week.date || raw.some((p) => p === null || p < 0 || p > 1)
          || Math.abs(raw.reduce((total, p) => total + p, 0) - 1) > 1e-6) return null;
        // Match analysis.validation.evaluate_predictions, including hard 0/1 baselines.
        const clipped = raw.map((p) => Math.max(1e-9, Math.min(1, p)));
        const total = clipped.reduce((sum, p) => sum + p, 0);
        const probability = clipped.map((p) => p / total);
        const predicted = probability.reduce((best, p, index) => p > probability[best] ? index : best, 0);
        return { target, raw, probability, predicted, current, fallback: row.fallback === true };
      });
      // Every displayed model is evaluated on exactly the same completed origins.
      if (!names.length || current < 0 || candidates.some((row) => !row)
        || candidates.some((row) => row.target !== candidates[0].target)) { scope.excludedCount += 1; continue; }
      const actual = FORECAST_STATES.indexOf(actualByDate.get(candidates[0].target));
      if (actual < 0) { scope.excludedCount += 1; continue; }
      scope.completedCount += 1;
      scope.completedOriginStart ??= week.date;
      scope.completedOriginEnd = week.date;
      scope.completedStart ??= candidates[0].target;
      scope.completedEnd = candidates[0].target;
      candidates.forEach((row, index) => rowsByModel.get(names[index]).push({ ...row, actual, origin: week.date }));
    }
    const leaderboard = names.map((name) => {
      const original = (comparisonModel.leaderboard || []).find((row) => nameOf(row) === name) || {};
      return {
        selection_log_loss: null, selection_calibration_error: null, ...original,
        name, research_candidate: original.research_candidate === true,
        horizon_weeks: horizon, target: "endpoint", evaluation_split: "holdout",
        ...forecastEvaluationMetrics(rowsByModel.get(name)), n: rowsByModel.get(name).length, scope_rank: null,
      };
    });
    if (horizon !== 1) for (const row of leaderboard) { row.mean_detection_delay_forecast_weeks = null; row.detected_event_count = null; }
    [...leaderboard].filter((row) => row.n_predictions > 0).sort((a, b) =>
      a.log_loss - b.log_loss || a.calibration_error - b.calibration_error || a.name.localeCompare(b.name))
      .forEach((row, index) => { row.scope_rank = index + 1; });
    const referenceName = horizon === 1 ? nameOf(payload.selection?.operating_champion) || nameOf(payload.model?.champion) : "markov_endpoint";
    const reference = rowsByModel.get(referenceName) || [];
    const referenceLoss = leaderboard.find((row) => row.name === referenceName)?.log_loss;
    const comparisons = Object.fromEntries(leaderboard.map((metric) => {
      const rows = rowsByModel.get(metric.name);
      const n = reference.length === rows.length ? reference.length : 0;
      return [metric.name, { samePredictions: n ? rows.filter((row, index) => row.predicted === reference[index].predicted).length : 0,
        comparedWeeks: n,
        meanProbabilityDifference: n ? rows.reduce((total, row, index) => total
          + row.raw.reduce((sum, p, state) => sum + Math.abs(p - reference[index].raw[state]), 0), 0) / (3 * n) : null,
        logLossDifference: n && number(referenceLoss) !== null ? metric.log_loss - referenceLoss : null }];
    }));
    return { leaderboard, scope, comparisons };
  }
  function forecastEvaluationMetrics(rows) {
    const n = rows.length, safeRatio = (a, b) => b ? a / b : 0;
    const metrics = { log_loss: null, brier: null, accuracy: null, balanced_accuracy: null, macro_f1: null,
      transition_precision: null, transition_recall: null, transition_event_count: 0, on_time_departure_count: 0,
      false_alarm_count: 0, exposure_years: n / 52.1775, false_alarms_per_year: null, detected_event_count: 0,
      mean_detection_delay_forecast_weeks: null, transition_state_precision: null, transition_state_recall: null,
      calibration_error: null, n_predictions: n, fallback_count: 0, worsening_event_count: 0,
      on_time_worsening_count: 0, recovery_event_count: 0, on_time_recovery_count: 0,
      period_start: rows[0]?.origin || null, period_end: rows.at(-1)?.target || null };
    if (!n) return metrics;
    const confusion = FORECAST_STATES.map(() => [0, 0, 0]), events = [], bins = Array.from({ length: 10 }, () => []);
    let loss = 0, brier = 0, correct = 0;
    rows.forEach((row, index) => {
      const { actual, current, predicted, probability } = row;
      confusion[actual][predicted] += 1;
      correct += Number(actual === predicted);
      loss -= Math.log(probability[actual]);
      brier += probability.reduce((total, p, state) => total + (p - Number(state === actual)) ** 2, 0);
      const confidence = probability[predicted];
      const bin = Array.from({ length: 10 }, (_, i) => (i + 1) * 0.1).findIndex((upper) => confidence <= upper);
      bins[bin < 0 ? 9 : bin].push({ confidence, correct: Number(actual === predicted) });
      const event = actual !== current, alert = predicted !== current;
      if (event) { events.push(index); metrics.transition_event_count += 1; }
      metrics.on_time_departure_count += Number(event && alert);
      metrics.false_alarm_count += Number(!event && alert);
      metrics.fallback_count += Number(row.fallback);
      metrics.worsening_event_count += Number(actual > current);
      metrics.on_time_worsening_count += Number(actual > current && predicted > current);
      metrics.recovery_event_count += Number(actual < current);
      metrics.on_time_recovery_count += Number(actual < current && predicted < current);
    });
    const actualCounts = confusion.map((row) => row.reduce((sum, value) => sum + value, 0));
    const predictedCounts = FORECAST_STATES.map((_, state) => confusion.reduce((sum, row) => sum + row[state], 0));
    const represented = actualCounts.filter((value) => value > 0).length;
    const delays = [];
    events.forEach((position, index) => {
      const end = events[index + 1] ?? n;
      for (let next = position; next < end; next += 1) {
        if (rows[next].predicted === rows[position].actual) { delays.push(next - position); break; }
      }
    });
    return { ...metrics, log_loss: loss / n, brier: brier / n, accuracy: correct / n,
      balanced_accuracy: actualCounts.reduce((total, value, state) => total + safeRatio(confusion[state][state], value), 0) / represented,
      macro_f1: actualCounts.reduce((total, value, state) => total + safeRatio(2 * confusion[state][state], value + predictedCounts[state]), 0) / 3,
      transition_precision: safeRatio(metrics.on_time_departure_count, metrics.on_time_departure_count + metrics.false_alarm_count),
      transition_recall: safeRatio(metrics.on_time_departure_count, metrics.transition_event_count),
      false_alarms_per_year: metrics.false_alarm_count / metrics.exposure_years,
      detected_event_count: delays.length,
      mean_detection_delay_forecast_weeks: delays.length ? delays.reduce((sum, delay) => sum + delay, 0) / delays.length : null,
      transition_state_precision: safeRatio(confusion[1][1], predictedCounts[1]),
      transition_state_recall: safeRatio(confusion[1][1], actualCounts[1]),
      calibration_error: bins.reduce((total, bin) => total + Math.abs(bin.reduce((sum, row) => sum + row.correct - row.confidence, 0)) / n, 0) };
  }
  function researchForecastForWeek(payload, modelName, date) {
    const model = forecastImprovementModels(payload).find((row) => row.id === modelName);
    if (!model) return null;
    const row = [model.latest, ...(model.history || [])].find((row) => row?.origin_date?.slice(0, 10) === date);
    if (!row?.probabilities) return null;
    const probabilities = { ...row.probabilities };
    const predicted = FORECAST_STATES.reduce((best, state) => probabilities[state] > probabilities[best] ? state : best, FORECAST_STATES[0]);
    return { model: model.id, date: row.target_date.slice(0, 10), state: predicted,
      probabilities, confidence: probabilities[predicted], fallback: false,
      entropy: -FORECAST_STATES.reduce((sum, state) => sum + (probabilities[state] > 0 ? probabilities[state] * Math.log(probabilities[state]) : 0), 0) / Math.log(3),
      research_candidate: true, actual: row.actual, evidence_track: "reconstructed_market" };
  }
  function validateForecastImprovement(payload = {}) {
    const block = payload?.research?.forecast_improvement;
    if (block === undefined) return [];
    const errors = [], fail = (message) => errors.push(`forecast_improvement ${message}`);
    const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
    const timestamp = (value) => typeof value === "string" && Number.isFinite(Date.parse(value));
    const zonedTimestamp = (value) => timestamp(value) && /(?:Z|[+-]\d{2}:\d{2})$/i.test(value);
    const sha = (value) => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
    const simplex = (value) => object(value) && Object.keys(value).length === 3
      && FORECAST_STATES.every((state) => number(value[state]) !== null && value[state] >= 0 && value[state] <= 1)
      && Math.abs(FORECAST_STATES.reduce((sum, state) => sum + value[state], 0) - 1) < 1e-8;
    if (!object(block) || block.schema_version !== "regime-forecast-improvement/1"
      || block.evidence_track !== "reconstructed_market"
      || block.selected_model !== "boundary_filtered_history" || !zonedTimestamp(block.data_as_of)) {
      fail("메타데이터가 올바르지 않습니다."); return errors;
    }
    if (payload.meta?.data_as_of && Date.parse(block.data_as_of) !== Date.parse(payload.meta.data_as_of)) fail("기준 시각이 다릅니다.");
    if (block.selection?.frozen_model !== block.selected_model || !timestamp(block.selection?.selection_end)
      || Date.parse(block.selection.selection_end) > Date.parse("2023-01-01")) fail("선정기간이 올바르지 않습니다.");
    const models = block.models;
    if (!Array.isArray(models) || models.length !== 2 || new Set(models.map((model) => model?.id)).size !== 2
      || models.some((model) => !FORECAST_RESEARCH_IDS.includes(model?.id))) { fail("모델 목록이 올바르지 않습니다."); return errors; }
    let expectedOrigins = null;
    for (const model of models) {
      if (typeof model.label !== "string" || !model.label || !object(model.metrics)) { fail("모델 정보가 없습니다."); continue; }
      for (const split of ["selection", "holdout"]) {
        const metrics = model.metrics[split];
        if (!object(metrics) || !Number.isInteger(metrics.n_predictions) || metrics.n_predictions <= 0
          || ["log_loss", "brier", "transition_recall", "false_alarms_per_year"].some((key) => number(metrics[key]) === null || metrics[key] < 0)
          || metrics.transition_recall > 1
          || ["on_time_departure_count", "transition_event_count", "false_alarm_count"].some((key) => !Number.isInteger(metrics[key]) || metrics[key] < 0)
          || metrics.on_time_departure_count > metrics.transition_event_count) fail(`${model.id} 지표가 올바르지 않습니다.`);
      }
      if (!Array.isArray(model.history) || ["selection", "holdout"].some((split) => model.history.filter((row) => row?.evaluation_split === split).length !== model.metrics[split]?.n_predictions)
        || model.history.length !== model.metrics.selection?.n_predictions + model.metrics.holdout?.n_predictions) { fail(`${model.id} 평가 표본이 다릅니다.`); continue; }
      const origins = model.history.map((row) => row?.origin_date);
      if (new Set(origins).size !== origins.length || origins.some((date, index) => index > 0 && Date.parse(date) <= Date.parse(origins[index - 1]))) fail(`${model.id} 이력이 중복되거나 정렬되지 않았습니다.`);
      const identities = model.history.map((row) => [Date.parse(row?.origin_date), Date.parse(row?.target_date), row?.current_state, row?.actual, row?.evaluation_split]);
      if (expectedOrigins && JSON.stringify(expectedOrigins) !== JSON.stringify(identities)) fail("모델별 비교 시점이나 관측이 다릅니다.");
      expectedOrigins = identities;
      const scored = { selection: [], holdout: [] };
      for (const [row, latest] of [...model.history.map((row) => [row, false]), [model.latest, true]]) {
        const horizonHours = object(row) ? (Date.parse(row.target_date) - Date.parse(row.origin_date)) / 3600000 : NaN;
        if (!object(row) || !zonedTimestamp(row.origin_date) || !zonedTimestamp(row.target_date)
          || !Number.isFinite(horizonHours) || horizonHours < 167 || horizonHours > 169
          || !FORECAST_STATES.includes(row.current_state) || !simplex(row.probabilities) || !simplex(row.raw_probabilities)) { fail(`${model.id} 예측이 올바르지 않습니다.`); continue; }
        if (latest ? row.actual !== null || row.evaluation_split !== "unobserved" || Date.parse(row.origin_date) !== Date.parse(block.data_as_of)
          : !FORECAST_STATES.includes(row.actual) || row.evaluation_split !== (Date.parse(row.origin_date) < Date.parse("2023-01-01") ? "selection" : "holdout")
            || (row.evaluation_split === "selection" && Date.parse(row.target_date) >= Date.parse(block.selection.selection_end))
            || Date.parse(row.target_date) > Date.parse(model.latest?.origin_date)) fail(`${model.id} 예측 목표가 관측과 다릅니다.`);
        const predicted = FORECAST_STATES.reduce((best, state) => row.probabilities[state] > row.probabilities[best] ? state : best, FORECAST_STATES[0]);
        if (row.predicted !== predicted) fail(`${model.id} 최빈 국면이 다릅니다.`);
        const calibration = row.calibration;
        if (!object(calibration) || number(calibration.temperature) === null || calibration.temperature <= 0
          || !Number.isInteger(calibration.rows) || calibration.rows < 0 || calibration.rows > 156
          || !zonedTimestamp(calibration.last_train_target) || Date.parse(calibration.last_train_target) >= Date.parse(row.origin_date)) fail(`${model.id} 교정 시점이 올바르지 않습니다.`);
        const observed = payload.weekly?.find((week) => week.date === row.origin_date.slice(0, 10));
        const target = payload.weekly?.find((week) => week.date === row.target_date.slice(0, 10));
        if (observed?.current?.state && observed.current.state !== row.current_state) fail(`${model.id} 현재 국면이 다릅니다.`);
        if (!latest && target?.current?.state && target.current.state !== row.actual) fail(`${model.id} 실제 국면이 다릅니다.`);
        if (!latest && FORECAST_STATES.includes(row.actual) && Object.hasOwn(scored, row.evaluation_split)) {
          const actualIndex = FORECAST_STATES.indexOf(row.actual), currentIndex = FORECAST_STATES.indexOf(row.current_state), predictedIndex = FORECAST_STATES.indexOf(predicted);
          const clipped = FORECAST_STATES.map((state) => Math.max(1e-9, row.probabilities[state]));
          const total = clipped.reduce((sum, value) => sum + value, 0), p = clipped.map((value) => value / total);
          const event = actualIndex !== currentIndex, alert = predictedIndex !== currentIndex;
          scored[row.evaluation_split].push({ loss: -Math.log(p[actualIndex]), brier: p.reduce((sum, value, index) => sum + (value - (index === actualIndex ? 1 : 0)) ** 2, 0),
            event, hit: event && alert, falseAlarm: !event && alert,
            worsening: actualIndex > currentIndex, worseningHit: actualIndex > currentIndex && predictedIndex > currentIndex,
            recovery: actualIndex < currentIndex, recoveryHit: actualIndex < currentIndex && predictedIndex < currentIndex });
        }
      }
      const last = model.history.at(-1);
      if (Date.parse(last?.target_date) !== Date.parse(block.data_as_of) || last?.actual !== model.latest?.current_state) fail(`${model.id} 최신 관측 연결이 다릅니다.`);
      for (const split of ["selection", "holdout"]) {
        const rows = scored[split], count = rows.length, sum = (key) => rows.reduce((total, row) => total + Number(row[key]), 0);
        const events = sum("event"), hits = sum("hit"), falseAlarms = sum("falseAlarm");
        const expected = { n_predictions: count, log_loss: sum("loss") / count, brier: sum("brier") / count,
          transition_event_count: events, on_time_departure_count: hits, false_alarm_count: falseAlarms,
          transition_recall: events ? hits / events : 0, false_alarms_per_year: falseAlarms / count * 52.1775,
          worsening_event_count: sum("worsening"), on_time_worsening_count: sum("worseningHit"),
          recovery_event_count: sum("recovery"), on_time_recovery_count: sum("recoveryHit") };
        const metric = model.metrics[split];
        for (const [key, value] of Object.entries(expected)) {
          if (!Number.isFinite(value) || number(metric?.[key]) === null || Math.abs(metric[key] - value) > 1e-8) fail(`${model.id} ${split}.${key}가 예측 이력과 다릅니다.`);
        }
      }
    }
    if (!object(block.provenance) || ["input_sha256", "code_sha256", "baseline_oos_sha256", "cache_key"].some((key) => !sha(block.provenance[key]))) fail("재현 근거가 올바르지 않습니다.");
    return errors;
  }
  function contextPosition(scores = {}) {
    const read = (key) => number(typeof scores[key] === "object" ? scores[key]?.value ?? scores[key]?.score : scores[key]);
    const trend = read("trend"), rawStability = read("stress");
    const stress = rawStability === null ? null : -rawStability;
    const scale = Math.max(1, Math.abs(trend || 0), Math.abs(stress || 0));
    return { trend, stress, scale, x: trend === null ? null : 50 + trend / scale * 42, y: stress === null ? null : 50 - stress / scale * 42 };
  }
  function researchScope(payload = {}) {
    const downside = payload.research?.extensions?.downside || {};
    return { asOf: payload.meta?.data_as_of || null,
      downside: (downside.latest || []).map((row) => ({ ...row, origin: row.origin_date || downside.as_of || null })) };
  }
  function sensitivityRanges(summary = {}) {
    const range = (rows, field) => {
      const values = (rows || []).filter((row) => row.spec_id !== "operating_control").map((row) => number(row[field])).filter((value) => value !== null);
      return { control: number((rows || []).find((row) => row.spec_id === "operating_control")?.[field]), lower: values.length ? Math.min(...values) : null, upper: values.length ? Math.max(...values) : null };
    };
    return [
      { label: "주간 국면 전환율", ...range(summary.weekly_flip_rate, "value") },
      { label: "전환 시점 일치도", ...range(summary.transition_jaccard, "value") },
      ...[["위험 선호 비중", "risk_on"], ["전환 비중", "transition"], ["위험 회피 비중", "risk_off"]].map(([label, state]) => ({ label, ...range(summary.state_occupancy, `occupancy_${state}`) })),
    ];
  }
  function createPerformanceRenderers(dependencies) {
    const { createElement, createSvg, finiteNumber, firstValue, isObject, formatNumber, formatSignedPercent, formatDate, performanceRowValue, performanceRowDate, strategySummaryRow, svgLinePath } = dependencies;
  function renderPerformanceLineChart(rows, strategyKeys, labels, evaluationRange, primaryKey) {
    const card = createElement("section", "performance-visual performance-wealth-card");
    const heading = createElement("div", "performance-visual-heading");
    heading.append(
      createElement("strong", null, "누적 자산"),
      createElement("span", null, "지수 · 시작 1.0"),
    );
    card.append(heading);
    const series = strategyKeys.map((key) => ({
      key,
      label: labels[key],
      values: rows.map((row) => performanceRowValue(row, key, "wealth")),
    })).filter((item) => item.values.some((value) => finiteNumber(value) !== null));
    if (rows.length < 8 || !series.length) return null;
    const width = 960;
    const height = 286;
    const margin = { top: 18, right: 170, bottom: 34, left: 54 };
    const values = series.flatMap((item) => item.values).filter((value) => finiteNumber(value) !== null);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const padding = Math.max(0.03, (high - low) * 0.08);
    const yMin = low - padding;
    const yMax = high + padding;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const xForIndex = (index) => margin.left + innerWidth * index / Math.max(1, rows.length - 1);
    const yForValue = (value) => margin.top + innerHeight * (yMax - value) / Math.max(0.000001, yMax - yMin);
    const svg = createSvg("svg", {
      class: "performance-chart",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `${evaluationRange} 누적 자산 지수. ${series.map((item) => item.label).join(", ")} 비교`,
    });
    for (let index = 0; index < 4; index += 1) {
      const value = yMin + (yMax - yMin) * index / 3;
      const y = yForValue(value);
      svg.append(createSvg("line", { class: "performance-grid-line", x1: margin.left, x2: width - margin.right, y1: y, y2: y }));
      const label = createSvg("text", { class: "performance-axis-label", x: margin.left - 8, y: y + 4, "text-anchor": "end" });
      label.textContent = formatNumber(value, 2);
      svg.append(label);
    }
    for (const item of series) {
      svg.append(createSvg("path", {
        class: `performance-series performance-series-${item.key}${item.key === primaryKey ? " is-primary" : ""}`,
        d: svgLinePath(item.values, xForIndex, yForValue),
        fill: "none",
      }));
    }
    const endpoints = series.map((item) => {
      const index = item.values.findLastIndex((value) => finiteNumber(value) !== null);
      return { item, value: item.values[index], x: xForIndex(index), y: yForValue(item.values[index]) };
    }).sort((left, right) => left.y - right.y);
    let previousY = margin.top - 18;
    for (const point of endpoints) {
      const labelY = Math.min(height - margin.bottom, Math.max(point.y, previousY + 17));
      previousY = labelY;
      svg.append(createSvg("line", { class: `performance-endpoint-guide performance-series-${point.item.key}`, x1: point.x, y1: point.y, x2: width - margin.right + 8, y2: labelY }));
      const text = createSvg("text", { class: `performance-endpoint-label performance-label-${point.item.key}`, x: width - margin.right + 12, y: labelY + 4 });
      text.textContent = `${point.item.label.replace("실행 기준 · ", "").replace("관찰 후보 · ", "")} ${formatNumber(point.value, 2)}`;
      svg.append(text);
    }
    const firstLabel = createSvg("text", { class: "performance-axis-label", x: margin.left, y: height - 8, "text-anchor": "start" });
    firstLabel.textContent = formatDate(performanceRowDate(rows[0]));
    const lastLabel = createSvg("text", { class: "performance-axis-label", x: width - margin.right, y: height - 8, "text-anchor": "end" });
    lastLabel.textContent = formatDate(performanceRowDate(rows[rows.length - 1]));
    svg.append(firstLabel, lastLabel);
    const legend = createElement("div", "performance-legend");
    for (const item of series) {
      const legendItem = createElement("span", `performance-legend-${item.key}${item.key === primaryKey ? " is-primary" : ""}`);
      legendItem.append(createElement("i"), createElement("span", null, item.label));
      legend.append(legendItem);
    }
    card.append(svg, legend);
    return card;
  }

  function drawdownSeries(rows, strategyKey) {
    let peak = null;
    return rows.map((row) => {
      const supplied = performanceRowValue(row, strategyKey, "drawdown");
      if (supplied !== null) return supplied;
      const wealth = performanceRowValue(row, strategyKey, "wealth");
      if (wealth === null) return null;
      peak = peak === null ? wealth : Math.max(peak, wealth);
      return peak > 0 ? wealth / peak - 1 : null;
    });
  }

  function renderPerformanceDrawdownChart(rows, strategyKeys, labels, evaluationRange, primaryKey) {
    const series = strategyKeys.map((key) => ({ key, label: labels[key], values: drawdownSeries(rows, key) }))
      .filter((item) => item.values.some((value) => finiteNumber(value) !== null));
    if (rows.length < 8 || !series.length) return null;
    const card = createElement("section", "performance-visual performance-drawdown-card");
    const heading = createElement("div", "performance-visual-heading");
    heading.append(
      createElement("strong", null, "낙폭"),
      createElement("span", null, "고점 대비"),
    );
    const width = 960;
    const height = 190;
    const margin = { top: 16, right: 18, bottom: 28, left: 54 };
    const values = series.flatMap((item) => item.values).filter((value) => finiteNumber(value) !== null);
    const yMin = Math.min(-0.01, ...values);
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const xForIndex = (index) => margin.left + innerWidth * index / Math.max(1, rows.length - 1);
    const yForValue = (value) => margin.top + innerHeight * (0 - value) / Math.max(0.000001, -yMin);
    const zeroY = yForValue(0);
    const svg = createSvg("svg", {
      class: "performance-chart performance-drawdown-chart",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `${evaluationRange} 낙폭. 0선 포함`,
    });
    svg.append(createSvg("line", { class: "performance-zero-line", x1: margin.left, x2: width - margin.right, y1: zeroY, y2: zeroY }));
    for (const item of [...series].reverse()) {
      const line = svgLinePath(item.values, xForIndex, yForValue);
      if (!line) continue;
      if (item.key === primaryKey) {
        const area = `${line} L${xForIndex(rows.length - 1).toFixed(2)},${zeroY.toFixed(2)} L${xForIndex(0).toFixed(2)},${zeroY.toFixed(2)} Z`;
        svg.append(createSvg("path", { class: "performance-drawdown-area", d: area }));
      }
      svg.append(createSvg("path", { class: `performance-series performance-series-${item.key}${item.key === primaryKey ? " is-primary" : ""}`, d: line, fill: "none" }));
    }
    const minLabel = createSvg("text", { class: "performance-axis-label", x: margin.left - 8, y: yForValue(yMin) + 4, "text-anchor": "end" });
    minLabel.textContent = formatSignedPercent(yMin, 0);
    const zeroLabel = createSvg("text", { class: "performance-axis-label", x: margin.left - 8, y: zeroY + 4, "text-anchor": "end" });
    zeroLabel.textContent = "0%";
    svg.append(minLabel, zeroLabel);
    card.append(heading, svg);
    return card;
  }

  function renderPerformanceFallback(strategies, strategyKeys, labels, evaluationRange, evaluationWeeks, primaryKey) {
    const card = createElement("section", "performance-visual performance-fallback-card");
    const heading = createElement("div", "performance-visual-heading");
    heading.append(
      createElement("strong", null, "누적 성과"),
      createElement("span", null, "누적 수익률"),
    );
    const rows = strategyKeys.map((key) => ({
      key,
      label: labels[key],
      value: finiteNumber(strategySummaryRow(strategies, key).cumulative_return),
    })).filter((item) => item.value !== null);
    const scale = Math.max(0.01, ...rows.map((row) => Math.abs(row.value)));
    const bars = createElement("div", "performance-summary-bars");
    for (const row of rows) {
      const item = createElement("div", `performance-summary-bar ${row.key === primaryKey ? "is-primary" : ""}`);
      const track = createElement("span", "performance-summary-track");
      const fill = createElement("i");
      fill.style.width = `${Math.max(2, Math.abs(row.value) / scale * 100)}%`;
      track.append(fill);
      item.append(createElement("span", null, row.label), track, createElement("strong", null, formatSignedPercent(row.value)));
      bars.append(item);
    }
    card.append(heading, bars);
    return card;
  }

  function renderPerformanceBridge(strategyMetrics, evaluationRange, evaluationWeeks) {
    const gross = finiteNumber(strategyMetrics.gross_cumulative_return);
    const net = finiteNumber(strategyMetrics.cumulative_return);
    if (gross === null || net === null) return null;
    const costDrag = net - gross;
    const card = createElement("section", "performance-visual performance-bridge-card");
    const heading = createElement("div", "performance-visual-heading");
    heading.append(
      createElement("strong", null, "Gross → Cost → Net"),
      createElement("span", null, "누적 수익률"),
    );
    const bridge = createElement("div", "performance-bridge");
    for (const [label, value, operator] of [
      ["Gross", gross, ""],
      ["Cost drag", costDrag, "+"],
      ["Net", net, "="],
    ]) {
      const step = createElement("div", `performance-bridge-step is-${label.toLowerCase().replace(/\s+/g, "-")}`);
      if (operator) step.append(createElement("span", "performance-bridge-operator", operator));
      step.append(createElement("small", null, label), createElement("strong", null, formatSignedPercent(value, 2)));
      bridge.append(step);
    }
    card.append(heading, bridge);
    return card;
  }

  function turnoverValues(strategyMetrics) {
    const fullL1 = finiteNumber(firstValue(strategyMetrics, [
      "annualized_full_l1_turnover", "full_l1_turnover", "annualized_turnover",
    ]));
    const suppliedOneWay = finiteNumber(firstValue(strategyMetrics, [
      "annualized_one_way_turnover", "one_way_turnover", "investor_turnover",
    ]));
    return Object.freeze({
      oneWay: suppliedOneWay !== null ? suppliedOneWay : fullL1 === null ? null : fullL1 / 2,
      fullL1,
      inferred: suppliedOneWay === null && fullL1 !== null,
    });
  }

  function renderPerformanceTurnover(strategies, strategyKeys, labels, evaluationRange, evaluationWeeks, primaryKey) {
    const rows = strategyKeys.map((key) => ({
      key,
      label: labels[key],
      ...turnoverValues(strategySummaryRow(strategies, key)),
    })).filter((item) => item.oneWay !== null);
    if (!rows.length) return null;
    const scale = Math.max(0.01, ...rows.map((row) => row.oneWay));
    const card = createElement("section", "performance-visual performance-turnover-card");
    const heading = createElement("div", "performance-visual-heading");
    heading.append(
      createElement("strong", null, "회전율"),
      createElement("span", null, "투자자 one-way · 연환산"),
    );
    const bars = createElement("div", "performance-turnover-bars");
    for (const row of rows) {
      const item = createElement("div", `performance-turnover-row ${row.key === primaryKey ? "is-primary" : ""}`);
      const track = createElement("span", "performance-turnover-track");
      const fill = createElement("i");
      fill.style.width = `${Math.max(2, row.oneWay / scale * 100)}%`;
      track.append(fill);
      item.append(createElement("span", null, row.label), track, createElement("strong", null, formatSignedPercent(row.oneWay)));
      item.title = `투자자 one-way ${formatSignedPercent(row.oneWay)} · full-L1 ${formatSignedPercent(row.fullL1)}${row.inferred ? " · one-way는 full-L1의 1/2 환산" : ""}`;
      bars.append(item);
    }
    card.append(heading, bars);
    return card;
  }

  function renderPerformanceDetailTable(strategies, strategyKeys, labels, evaluationComplete, primaryKey) {
    const metrics = [
      ["연수익", "annualized_return", (value) => formatSignedPercent(value), true],
      ["Sharpe", "sharpe", (value) => formatNumber(value, 2), true],
      ["MDD", "maximum_drawdown", (value) => formatSignedPercent(value), false],
      ["연간 매수+매도", "annualized_turnover", (value) => formatSignedPercent(value), false],
    ];
    const details = createElement("details", "performance-detail compact-table-details");
    details.append(createElement("summary", null, "상세 지표 · full-L1 회전율"));
    const scroll = createElement("div", "table-scroll performance-detail-scroll");
    scroll.tabIndex = 0;
    const strategyTable = createElement("table", "decision-shadow-table");
    strategyTable.append(createElement("caption", "sr-only", "위험 국면 기반 주간 리밸런싱 전략 성과 비교"));
    const strategyHead = createElement("thead");
    const strategyHeadRow = createElement("tr");
    strategyHeadRow.append(createElement("th", null, "지표"));
    for (const key of strategyKeys) {
      const heading = createElement("th", key === primaryKey ? "is-shadow" : null, labels[key]);
      heading.setAttribute("scope", "col");
      strategyHeadRow.append(heading);
    }
    strategyHead.append(strategyHeadRow);
    const strategyBody = createElement("tbody");
    for (const [label, field, formatter, requiresCompleteSample] of metrics) {
      const row = createElement("tr");
      const rowHeading = createElement("th", null, label);
      rowHeading.setAttribute("scope", "row");
      if (field === "annualized_turnover") {
        rowHeading.title = "매수와 매도 비중을 모두 합산한 연환산 full-L1 값";
        rowHeading.setAttribute("aria-label", "연간 매수와 매도 비중 합계 full-L1");
      }
      row.append(rowHeading);
      for (const key of strategyKeys) {
        const strategyMetrics = strategySummaryRow(strategies, key);
        const rawValue = field === "annualized_turnover"
          ? firstValue(strategyMetrics, ["annualized_full_l1_turnover", "annualized_turnover"])
          : strategyMetrics[field];
        const value = requiresCompleteSample && !evaluationComplete ? "—" : formatter(rawValue);
        const cell = createElement("td", key === primaryKey ? "is-shadow" : null, value);
        cell.setAttribute("data-label", labels[key]);
        row.append(cell);
      }
      strategyBody.append(row);
    }
    strategyTable.append(strategyHead, strategyBody);
    scroll.append(strategyTable);
    details.append(scroll);
    return details;
  }

    return { renderPerformanceLineChart, renderPerformanceDrawdownChart, renderPerformanceFallback, renderPerformanceBridge, renderPerformanceTurnover, renderPerformanceDetailTable, turnoverValues };
  }
  return Object.freeze({ conditionalMetrics, portfolioAdjustment, forecastTiming, selectedForecastTiming, forecastDirections, transitionOutlook, durationEstimate, validateForecastResearch, researchForecastRow, multistateForecastForWeek, modelQuality, forecastImprovementModels, forecastComparisonModel, forecastEvaluation, forecastModelIds, enhancedForecastForWeek, enhancementForecastRows, ENHANCEMENT_MODEL_IDS, researchForecastForWeek, validateForecastImprovement, FORECAST_RESEARCH_IDS, contextPosition, researchScope, sensitivityRanges, createPerformanceRenderers });
});
