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
    assert '<footer class="dashboard-footer">' in HTML


def test_interactive_controls_have_accessible_names() -> None:
    assert '<label for="analysis-date">' in HTML
    assert '<label for="week-select">' in HTML
    assert '<label for="history-window">' in HTML
    assert '<label for="transition-horizon-select">' in HTML
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
    assert "현재 t" in HTML and "예측 t+1" in HTML
    assert "선정 구간" in HTML and "2023+ 진단" in HTML
    assert "향후 1주 안에 한 번 이상 현재 국면에서 이탈할 확률" in JS
    assert "향후 ${horizon}주 안에 한 번 이상 현재 국면에서 이탈할 확률" in JS
    assert 'class="diagnostic-label"' not in HTML
    assert "diagnostic-label" not in JS
    assert "stroke-dasharray" in CSS
    assert "border-style: dashed" in CSS
    assert "border-style: double" in CSS


def test_loading_error_empty_and_degraded_states_are_visible_contracts() -> None:
    for state_id in ("loading-state", "error-state", "empty-state", "data-alerts"):
        assert f'id="{state_id}"' in HTML
    assert "renderAlerts" in JS
    assert "선택 주 결과가 저하 상태입니다" in JS
    assert "소스 상태 확인 필요" in JS


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
    assert HTML.count('class="table-scroll" tabindex="0"') == 4
    assert '.table-scroll[tabindex="0"]:focus-visible' in CSS
    assert ".transition-horizon-field select" in CSS


def test_mobile_navigation_reveals_active_project() -> None:
    assert "function revealActiveProjectLink()" in JS
    assert "links.querySelector('[aria-current=\"page\"]')" in JS
    assert "links.scrollLeft += activeRect.right - linksRect.right" in JS
    assert 'window.addEventListener("resize", revealActiveProjectLink)' in JS


def test_health_and_research_limitations_remain_visible_on_mobile() -> None:
    assert 'id="header-health"' in HTML
    assert 'id="header-mode"' in HTML
    assert 'id="model-diagnostic"' in HTML
    assert 'id="shadow-nowcast-summary"' in HTML
    assert "개인·비상업 파생 결과" in HTML
    assert '<details class="research-notice-details operations-details">' in HTML
    assert "데이터 · 출처 · 운영" in HTML
    assert "This product uses the FRED® API" in HTML
    assert ".header-status-strip" in CSS


def test_probability_chart_declares_honest_fixed_axis() -> None:
    assert "for (const tick of [0, 0.25, 0.5, 0.75, 1])" in JS
    assert '`${Math.round(tick * 100)}%`' in JS
    assert "현재 국면 확률 · 0–100% 축" in JS
