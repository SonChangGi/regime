"""Accessibility and responsive-layout checks that do not require a browser."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")
JS = (ROOT / "web" / "app.js").read_text(encoding="utf-8")


def test_document_has_language_landmarks_and_skip_link() -> None:
    assert '<html lang="ko"' in HTML
    assert 'class="skip-link" href="#main-content"' in HTML
    assert '<nav class="site-nav" aria-label="연결 프로젝트 바로가기">' in HTML
    assert '<header class="page-header" aria-labelledby="page-title">' in HTML
    assert '<nav class="section-nav" aria-label="페이지 섹션 바로가기">' in HTML
    assert '<main id="main-content" tabindex="-1">' in HTML
    assert 'id="data-health"' in HTML


def test_interactive_controls_have_accessible_names() -> None:
    assert '<label for="analysis-date">' in HTML
    assert '<label for="week-select">' in HTML
    assert '<label for="history-window">' in HTML
    assert '<label for="transition-horizon-select">' in HTML
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
    assert "차트 값을 표로 보기" in HTML
    assert '<tbody id="history-data-body">' in HTML
    assert "Risk-on · 실선" in HTML
    assert "Transition · 파선" in HTML
    assert "Risk-off · 점선" in HTML
    assert 'id="chart-selection-readout"' in HTML and 'aria-live="polite"' in HTML
    assert 'id="probability-chart-wrap"' in HTML and 'tabindex="0"' in HTML
    assert "방향키로 날짜 이동" in HTML
    assert "↗" in HTML and "◆" in HTML and "↘" in HTML
    assert "현재 t" not in HTML and "예측 t+1" not in HTML
    assert 'membership ? "국면 소속도 히스토리" : "국면 확률 히스토리"' in JS
    assert 'isCurrent && isV5Payload() ? "소속도"' in JS
    assert '"예측확률"' in JS
    assert "최초 이탈 방향" in JS
    assert "52주 극단값은 시장 맥락이며 예측 기여도와는 별도" in HTML
    assert 'id="next-model-context"' not in HTML
    assert 'id="next-model-context-detail"' in HTML
    assert "입력 현재 국면·과거 전이" in JS
    assert "입력 완료 OOS 예측 풀 26·52·104주" in JS
    assert "function displayFreshness" in JS
    assert "과거 조회" in JS
    assert "공개 스냅샷" in JS
    assert 'currentFreshness.status === "stale"' in JS
    assert 'id="conditional-stat-grid" class="conditional-stat-grid" role="region"' in HTML
    assert "표본 부족" in JS
    assert 'createElement("td", null, formatSignedPercent(row.median_return))' in JS
    assert 'createElement("td", null, formatPercent(row.positive_rate))' in JS
    assert "선정 구간" in HTML and "2023+ 진단" in HTML
    assert "향후 1주 안에 한 번 이상 현재 국면에서 이탈할 확률" in JS
    assert "향후 ${horizon}주 안에 한 번 이상 현재 국면에서 이탈할 확률" in JS
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
    assert 'aria-label="선택 자산과 보유 기간의 국면별 조건부 성과 표 · 가로 스크롤 가능"' in HTML
    assert "${STATE_META[code].label}, ${asset} ${OUTCOME_ASSET_LABELS[asset]}" in JS
    assert '.table-scroll[tabindex="0"]:focus-visible' in CSS
    assert ".transition-horizon-field select" in CSS


def test_small_status_chips_use_high_contrast_text_tokens() -> None:
    assert "--status-ok-text: #006b52;" in CSS
    assert "--status-review-text: #805000;" in CSS
    assert "color: var(--status-ok-text);" in CSS
    assert "color: var(--status-review-text);" in CSS


def test_context_and_performance_units_are_explicit() -> None:
    assert "52주 표준화 기반 합성점수" in HTML
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


def test_result_identity_is_compact_without_general_warning_surface() -> None:
    assert 'id="header-health"' not in HTML
    assert 'id="header-mode"' not in HTML
    assert 'id="model-diagnostic"' not in HTML
    assert 'id="shadow-nowcast-summary"' not in HTML
    assert 'id="header-result-identity" class="result-identity-chip"' in HTML
    assert 'role="status" aria-live="polite"' in HTML
    assert HTML.index('id="header-result-identity"') > HTML.index('id="research-evidence"')
    assert HTML.index('id="header-result-identity"') > HTML.index('id="models"')
    assert 'id="research-evidence-details"' in HTML
    assert '["모의자료", profile, "파이프라인 검증"]' in JS
    assert '["실데이터", profile]' in JS
    assert 'id="fx-ablation-status"' in HTML
    assert 'id="model-evidence-summary"' in HTML
    assert '"전향적 shadow"' in JS
    assert '"core 비승격"' in JS
    assert "실제 OOS ${formatNumber(evaluationOrigins, 0)}개" in JS
    assert '"검토 필요"' in JS
    assert "개인·비상업 파생 결과" not in HTML
    assert '<details class="research-notice-details operations-details">' in HTML
    assert "데이터 · 출처 · 운영" in HTML
    assert "This product uses the FRED® API" not in HTML
    assert "진단 주의" not in JS
    assert "투자 조언 아님" not in HTML
    assert 'id="model-loss-chart"' in HTML
    assert 'id="leaderboard-body"' in HTML
    assert ".header-status-strip" in CSS


def test_probability_chart_declares_honest_fixed_axis() -> None:
    assert "for (const tick of [0, 0.25, 0.5, 0.75, 1])" in JS
    assert '`${Math.round(tick * 100)}%`' in JS
    assert '현재 국면 ${isV5Payload() ? "소속도" : "확률"} · 0–100% 축' in JS
