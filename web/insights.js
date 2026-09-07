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
      delay: metric("mean_detection_delay_forecast_weeks"), weeks: metric("n_predictions"),
      fallbackCount: metric("fallback_count"),
    });
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
  return Object.freeze({ conditionalMetrics, portfolioAdjustment, forecastTiming, modelQuality, contextPosition, researchScope, sensitivityRanges, createPerformanceRenderers });
});
