"""Accessibility and responsive-layout checks that do not require a browser."""

import re
import json
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "web" / "styles.css").read_text(encoding="utf-8") + (ROOT / "web" / "insights.css").read_text(encoding="utf-8")
JS = (ROOT / "web" / "app.js").read_text(encoding="utf-8") + (ROOT / "web" / "insights.js").read_text(encoding="utf-8")


def test_methodology_is_one_closed_disclosure_after_results_with_named_controls_retained():
    class Structure(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.elements = {}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            element_id = attrs.get("id")
            if element_id:
                self.elements[element_id] = (tag, attrs, [item[1] for item in self.stack])
            if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
                self.stack.append((tag, element_id))

        def handle_endtag(self, tag):
            for index in range(len(self.stack) - 1, -1, -1):
                if self.stack[index][0] == tag:
                    self.stack = self.stack[:index]
                    break

    structure = Structure()
    structure.feed(HTML)
    tag, attrs, parents = structure.elements["interpretation-methods"]
    assert tag == "details" and "open" not in attrs and "dashboard" in parents
    assert HTML.count("<summary>해석·계산 기준</summary>") == 1
    assert HTML.index('id="interpretation-methods"') > HTML.index('id="data-health"')
    for element_id in ("membership-definition", "duration-estimate-note", "forecast-method-notes"):
        assert "interpretation-methods" in structure.elements[element_id][2]
    for element_id in ("forecast-origin-at", "forecast-target-at", "latest-week", "transition-horizon-bars", "multistate-forecast", "duration-context"):
        assert "interpretation-methods" not in structure.elements[element_id][2]
    assert "종료시점 예측구간" not in HTML[:HTML.index('id="interpretation-methods"')]
    assert "공식 모델은 별도 유지" not in JS


def test_document_has_language_landmarks_and_skip_link() -> None:
    assert '<html lang="ko"' in HTML
    assert 'class="skip-link" href="#main-content"' in HTML
    assert '<nav class="site-nav" aria-label="연결 프로젝트 바로가기">' in HTML
    assert '<header class="page-header" aria-labelledby="page-title">' in HTML
    assert '<nav class="section-nav" aria-label="페이지 섹션 바로가기" id="dashboard-view-nav">' in HTML
    assert '<main id="main-content" tabindex="-1">' in HTML
    assert 'id="data-health"' in HTML


def test_favicon_is_inline_and_cannot_generate_a_local_404() -> None:
    assert '<link rel="icon" href="data:image/svg+xml,' in HTML
    assert 'href="./favicon' not in HTML


def test_interactive_controls_have_accessible_names() -> None:
    assert '<label for="analysis-date">' in HTML
    assert '<label for="week-select">' in HTML
    assert '<label for="history-window">' in HTML
    assert '<label for="transition-horizon-select">' in HTML
    assert '<label for="model-forecast-select">비교 모델</label>' in HTML
    assert 'id="model-forecast-select"' in HTML
    assert (
        'aria-controls="history model-quality-brief model-health-strip model-forecast-explorer model-loss-chart leaderboard-body conditional-stats"'
    ) in HTML
    assert 'aria-describedby="model-forecast-scope"' in HTML
    assert 'id="model-forecast-scope" class="sr-only"' in HTML
    assert "선택 모델·예측 기간을 요약, 평가, 예측 이력에 공통 적용" in HTML
    assert 'id="history-series-select"' not in HTML
    assert 'id="chart-readout-actual"' in HTML
    assert 'id="chart-readout-entropy"' in HTML
    assert 'id="chart-readout-observed-label"' in HTML
    assert 'id="history-observed-group-label"' in HTML
    assert '<label for="conditional-basis-select">기준</label>' in HTML
    assert 'id="conditional-basis-select"' in HTML
    assert 'aria-controls="conditional-stat-grid conditional-stat-body"' in HTML
    assert '<label for="conditional-asset-select">' in HTML
    assert '<label for="conditional-horizon-select">' in HTML
    assert 'id="theme-toggle"' in HTML and 'aria-pressed="false"' in HTML
    assert 'id="previous-week"' in HTML and 'aria-label="이전 관측 주"' in HTML
    assert 'id="next-week"' in HTML and 'aria-label="다음 관측 주"' in HTML
    assert 'id="latest-week"' in HTML
    assert 'id="screen-reader-status"' in HTML and 'aria-live="polite"' in HTML


def test_visuals_have_semantic_fallbacks_and_non_color_encoding() -> None:
    assert 'id="probability-chart"' in HTML
    assert 'role="img"' in HTML
    assert '<summary>수치 표</summary>' in HTML
    assert '<tbody id="history-data-body">' in HTML
    for code, line_class in (("risk_on", "risk-on-line"), ("transition", "transition-line"), ("risk_off", "risk-off-line")):
        assert f'class="legend-line {line_class}" aria-hidden="true"' in HTML
        assert f'data-state-label="{code}"' in HTML
    assert 'const lineStyles = { risk_on: "실선", transition: "파선", risk_off: "점선" }' in JS
    assert 'setAttribute("aria-label", historyMeta.legendLabel)' in JS
    assert "실제 t+1 결과" in HTML
    assert 'id="chart-selection-readout"' in HTML and 'aria-live="polite"' in HTML
    assert 'id="probability-chart-wrap"' in HTML and 'tabindex="0"' in HTML
    assert "방향키로 날짜 이동" in HTML
    assert 'data-state-symbol="risk_on"' in HTML
    assert 'data-state-symbol="transition"' in HTML
    assert 'data-state-symbol="risk_off"' in HTML
    assert "function stateMeta(" in JS
    assert "현재 t" not in HTML and "예측 t+1" not in HTML
    assert 'setText(dom["probability-chart-title"], historyMeta.title)' in JS
    assert "function actualNextWeekForWeek(" in JS
    assert "function forecastEntropyForWeek(" in JS
    assert 'tableCaption: `${model} ${membership ? "관측 소속도" : "관측 확률"}·${state.modelForecastHorizon}주 예측확률·실제 ${state.modelForecastHorizon}주 후 결과·정규화 예측 엔트로피`' in JS
    assert 'setText(dom["history-observed-group-label"], `${historyMeta.observedMeasure} · t`)' in JS
    assert 'setText(dom["chart-readout-observed-label"], historyMeta.observedMeasure)' in JS
    assert 'isCurrent && isV5Payload() ? "소속도"' in JS
    assert '"예측확률"' in JS
    assert "52주 극단값은 시장 맥락이며 예측 기여도와는 별도" not in HTML
    assert 'id="next-model-context"' not in HTML
    assert 'id="next-model-context-detail"' in HTML
    assert "function displayFreshness" in JS
    assert "공개 스냅샷" in JS
    assert 'currentFreshness.status === "stale"' in JS
    assert 'id="conditional-stat-grid" class="conditional-stat-grid" role="region"' in HTML
    assert 'setAttribute("aria-label", `${basisLabel} 자산별 평균 수익률`)' in JS
    assert '`${basisLabel} · ${horizonLabel} 자산 성과 표`' in JS
    assert '`${modelForecastLabel(state.comparisonModel)} 예측 국면`' in JS
    assert '"관측 국면"' in JS
    assert "표본 부족" in JS
    assert 'createElement("td", null, formatSignedPercent(row.median_return))' in JS
    assert 'createElement("td", null, formatPercent(row.positive_rate))' in JS
    assert "선정 구간" in HTML and "2023년 이후 진단" in HTML
    assert 'setText(dom["transition-value-label"], "1주 이탈")' in JS
    assert 'createElement("span", null, `${horizon}주 내 이탈 · ${formatDate(result.target, false)}까지`)' in JS
    assert '`${horizon}주 국면 이탈 확률 ${formatPercent(value)}`' in JS
    assert 'class="diagnostic-label"' not in HTML
    assert "diagnostic-label" not in JS
    assert "stroke-dasharray" in CSS
    assert "border-style: dashed" in CSS
    assert "border-style: double" in CSS


def test_loading_error_and_empty_states_remain_visible_contracts() -> None:
    for state_id in ("loading-state", "error-state", "empty-state"):
        assert f'id="{state_id}"' in HTML
    assert "DataContractError" in JS
    assert "renderAlerts" not in JS
    assert "선택 주 결과가 저하 상태입니다" not in JS
    assert "소스 상태 확인 필요" not in JS


def test_keyboard_focus_reduced_motion_and_mobile_rules_exist() -> None:
    assert ":focus-visible" in CSS
    assert "prefers-reduced-motion: reduce" in CSS
    breakpoints = re.findall(r"@media \(max-width: (\d+)px\)", CSS)
    assert len(breakpoints) >= 3
    assert "function handleChartKeydown" in JS
    assert '"Enter", " "' in JS
    assert "event.currentTarget.click()" in JS
    assert 'event.key === "ArrowLeft"' in JS
    assert 'event.key === "ArrowRight"' in JS
    assert 'event.key === "Home"' in JS
    assert 'event.key === "End"' in JS
    assert 'event.key === "Escape"' in JS
    assert "aria-current" in JS
    table_scrolls = re.findall(r'class="[^"]*\btable-scroll\b[^"]*" tabindex="0"', HTML)
    assert len(table_scrolls) == 5
    assert 'aria-label="선택 자산과 보유 기간의 국면별 조건부 성과 표"' in HTML
    assert '`${basisLabel}, ${stateMeta(code).label}, ${asset}, 수익률 ${formatted}, 전체 평균 대비' in JS
    assert '.table-scroll[tabindex="0"]:focus-visible' in CSS
    assert ".transition-horizon-field select" in CSS
    assert ".model-forecast-field select" in CSS
    assert "@media (max-width: 760px)" in CSS
    assert (
        ".model-heading-controls,\n"
        "  .model-forecast-field,\n"
        "  .model-forecast-field select {\n"
        "    width: 100%;"
    ) in CSS
    assert (
        ".model-forecast-body {\n"
        "    grid-template-columns: 1fr;"
    ) in CSS


def test_mobile_timeline_groups_fit_24px_targets_without_widening_the_marks() -> None:
    assert "WCAG-sized keyboard/touch target on a 390px viewport" in CSS
    assert "touch-action: manipulation;" in CSS
    assert ".timeline-cell::before" in CSS
    assert "width: 8px;" in CSS
    # Inspect the final, more specific rules: an earlier 24px declaration can
    # exist while a year-wrapper rule silently reduces the actual touch target.
    groups = list(re.finditer(r"\.timeline-year-group\s*\{([^}]+)\}", CSS))
    cells = list(re.finditer(r"\.timeline-year-weeks \.timeline-cell\s*\{([^}]+)\}", CSS))
    assert len(groups) == len(cells) == 2
    assert "min-width: calc(var(--timeline-week-count) * var(--timeline-week-width))" in groups[0][1]
    mobile_start = CSS.rfind("@media", 0, groups[-1].start())
    assert CSS[mobile_start:].startswith("@media (max-width: 760px)")

    def pixels(rule: str, property_name: str) -> float:
        match = re.search(rf"{re.escape(property_name)}:\s*([0-9.]+)px", rule)
        assert match is not None
        return float(match[1])

    target = pixels(cells[-1][1], "min-width")
    height = pixels(cells[-1][1], "min-height")
    week_width = pixels(groups[-1][1], "--timeline-week-width")
    gap = pixels(re.search(r"\.timeline-year-weeks\s*\{([^}]+)\}", CSS)[1], "gap")
    assert target >= 24 and height >= 24
    for count in (1, 2, 52, 53):
        assert count * week_width >= count * target + (count - 1) * gap


def test_small_status_chips_use_high_contrast_text_tokens() -> None:
    assert "--status-ok-text: #006b52;" in CSS
    assert "--status-review-text: #805000;" in CSS
    assert "color: var(--status-ok-text);" in CSS
    assert "color: var(--status-review-text);" in CSS


def _relative_luminance(color: str) -> float:
    components = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in components
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(left: str, right: str) -> float:
    lighter, darker = sorted((_relative_luminance(left), _relative_luminance(right)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _css_variables(selector: str) -> dict[str, str]:
    match = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", CSS)
    assert match is not None
    return dict(re.findall(r"(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{6});", match.group(1)))


def test_actual_payload_marks_are_separate_from_wcag_text_tokens_in_both_themes() -> None:
    payload = json.loads((ROOT / "publication" / "live" / "regime-results.json").read_text(encoding="utf-8"))
    payload_colors = {state["id"].replace("_", "-"): state["color"].lower() for state in payload["states"]}
    light = _css_variables(":root")
    dark = _css_variables('html[data-theme="dark"]')
    for name, payload_color in payload_colors.items():
        assert light[f"--{name}-mark"].lower() == payload_color
        assert dark[f"--{name}-mark"].lower() == payload_color
        assert _contrast(light[f"--{name}-text"], light["--surface"]) >= 4.5
        assert _contrast(dark[f"--{name}-text"], dark["--surface"]) >= 4.5
        assert _contrast(payload_color, light["--surface"]) >= 3.0
        assert _contrast(payload_color, dark["--surface"]) >= 3.0
    assert 'setProperty(`--${code.replaceAll("_", "-")}-mark`, color)' in JS
    assert 'setProperty(`--${code.replaceAll("_", "-")}`, color)' not in JS


def test_context_and_performance_units_are_explicit() -> None:
    assert re.search(r'id="factor-caption"[^>]*>52주 표준화[^<]*점수</p>', HTML)
    assert "52주 표준화 기반 합성점수" in JS
    assert "평균 95% CI" in HTML
    assert "연율 하방 변동성" in HTML
    assert 'observation_week' in JS
    assert 'observation_age_days' in JS


def test_mobile_navigation_reveals_active_project() -> None:
    assert "function revealActiveProjectLink()" in JS
    assert "links.querySelector('[aria-current=\"page\"]')" in JS
    assert "links.scrollLeft += activeRect.right - linksRect.right + edgePadding" in JS
    assert 'window.addEventListener("load", revealActiveProjectLink, { once: true })' in JS
    assert "document.fonts.ready.then(revealActiveProjectLink)" in JS
    assert 'window.addEventListener("resize", revealActiveProjectLink)' in JS


def test_collapsed_detail_cards_omit_general_warning_surfaces() -> None:
    assert 'id="header-health"' not in HTML
    assert 'id="header-mode"' not in HTML
    assert 'id="model-diagnostic"' not in HTML
    assert 'id="shadow-nowcast-summary"' not in HTML
    assert 'id="header-result-identity"' not in HTML
    assert 'id="forecast-contract-status"' not in HTML
    assert 'id="forecast-expired-notice"' not in HTML
    assert 'id="research-evidence-details"' in HTML
    assert 'id="research-methods" class="research-methods card"' in HTML
    assert 'id="fx-ablation-status"' in HTML
    assert 'id="model-evidence-summary"' in HTML
    assert '"전향적 shadow"' in JS
    assert '"core 비승격"' in JS
    assert "실제 OOS ${formatNumber(evaluationOrigins, 0)}개" in JS
    assert '"검토 필요"' not in JS
    assert "개인·비상업 파생 결과" not in HTML
    assert '<details id="research-methods"' in HTML
    assert "데이터 · 운영" in HTML
    assert "This product uses the FRED® API" not in HTML
    assert "진단 주의" not in JS
    assert "투자 조언 아님" not in HTML
    assert 'id="model-loss-chart"' in HTML
    assert 'id="leaderboard-body"' in HTML
    assert ".header-status-strip" in CSS


def test_probability_chart_declares_honest_fixed_axis() -> None:
    assert "for (const tick of [0, 0.25, 0.5, 0.75, 1])" in JS
    assert '`${Math.round(tick * 100)}%`' in JS
    assert 'aria-valuemin="0"' in HTML and 'aria-valuemax="100"' in HTML
    assert "`${range} · ${history.length}주`" in JS
