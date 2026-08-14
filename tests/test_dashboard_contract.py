"""Static contract checks for the dependency-free regime dashboard."""

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
HTML_PATH = WEB / "index.html"
CSS_PATH = WEB / "styles.css"
JS_PATH = WEB / "app.js"


def _valid_v3_browser_payload() -> dict:
    current = {
        "state": "transition",
        "probabilities": {"risk_on": 0.25, "transition": 0.5, "risk_off": 0.25},
        "confidence": 0.5,
        "entropy": 0.95,
    }
    next_week = {
        "state": "transition",
        "date": "2026-08-14",
        "probabilities": {"risk_on": 0.1, "transition": 0.8, "risk_off": 0.1},
        "confidence": 0.8,
        "entropy": 0.5,
    }
    model = {
        "champion": "markov",
        "selection_status": "provisional_predeployment",
        "leaderboard": [],
        "version": "weekly-nondl-structural-v3",
        "label_version": "market-causal-3state-v1",
        "feature_set_version": "weekly-pit-market-internals-v3",
        "primary_horizon_weeks": 1,
        "transition_selection_end": "2023-01-01",
        "transition_horizons_weeks": [1, 4, 13],
        "baseline_v2": {
            "result_version": "weekly-regime-result-v2",
            "label_version": "market-causal-3state-v1",
            "model_version": "weekly-nondl-walkforward-v2",
            "champion": "markov",
            "payload_sha256": "a" * 64,
            "artifacts_inventory_sha256": "b" * 64,
        },
        "transition_champions": {"1w": "hazard", "4w": "hazard", "13w": "duration"},
        "transition_leaderboard": [
            {
                "horizon_weeks": horizon,
                "model": "hazard",
                "selected": True,
                "evaluation_split": "selection",
                "binary_log_loss": 1.2,
                "brier": 0.2,
                "average_precision": None,
                "precision": 0.0,
                "recall": 0.0,
                "false_alarms_per_year": 0.0,
                "n_predictions": 10,
                "event_count": 0,
                "non_event_count": 10,
                "fallback_count": 0,
                "calibration_fallback_count": 0,
            }
            for horizon in (1, 4, 13)
        ],
        "shadow_nowcast": {"status": "shadow_only", "canonical_target": False},
    }
    transition_risk = {
        "1w": {
            "probability": 0.2,
            "target_end": "2026-08-14",
            "model": "hazard",
            "threshold": 0.5,
            "fallback": False,
            "fallback_reason": "",
        },
        "4w": {
            "probability": 0.3,
            "target_end": "2026-09-04",
            "model": "hazard",
            "threshold": 0.5,
            "fallback": False,
            "fallback_reason": "",
        },
        "13w": {
            "probability": 0.5,
            "target_end": "2026-11-06",
            "model": "duration",
            "threshold": 0.5,
            "fallback": False,
            "fallback_reason": "",
        },
    }
    return {
        "meta": {
            "schema_version": "1.0.0",
            "result_version": "weekly-regime-result-v3",
            "generation_id": "20260812T000000Z-example",
            "generated_at": "2026-08-12T00:00:00Z",
            "data_as_of": "2026-08-07",
            "mode": "demo",
            "timezone": "America/New_York",
        },
        "states": [
            {"id": "risk_on"},
            {"id": "transition"},
            {"id": "risk_off"},
        ],
        "model": model,
        "weekly": [{
            "date": "2026-08-07",
            "current": current,
            "next_week": next_week,
            "transition_probability": 0.2,
            "transition_risk": transition_risk,
            "scores": {"trend": 0.1, "stress": -0.1, "macro": 0.0, "financial_conditions": 0.0},
        }],
        "sources": [{"id": "fixture", "status": "degraded"}],
        "feature_catalog": [{"id": "fixture"}],
    }


def _valid_v4_browser_payload() -> dict:
    payload = deepcopy(_valid_v3_browser_payload())
    payload["meta"]["result_version"] = "weekly-regime-result-v4"
    payload["model"].update(
        {
            "version": "weekly-nondl-structural-v4",
            "feature_set_version": "weekly-pit-structural-v4",
            "baseline_v3": {
                "result_version": "weekly-regime-result-v3",
                "label_version": "market-causal-3state-v1",
                "model_version": "weekly-nondl-structural-v3",
                "champion": "markov",
                "payload_sha256": "de93c585117b2784750f586a4f84ad99964c63081b252ad7affd7a75bd797095",
                "artifacts_inventory_sha256": "8ef3778cc8c36faff0c80e2bf094f1f11bd6966ab3b7b2d6edb84ba292aff6b9",
                "captured_at": "2026-08-13",
            },
            "structural_preregistration": {
                "path": "config/structural_v4.json",
                "sha256": "2f53ada564efca770261f16ce6eb16ec9c9782bde014de7a7d85b7b24dbe407b",
            },
            "feature_manifest_sha256": "f" * 64,
            "evidence_artifacts": {
                "state_label_history": {
                    "path": "state-label-history.csv",
                    "row_count": 700,
                    "sha256": "b" * 64,
                    "label_fit_weeks": 520,
                    "label_fit_end": "2021-12-17T00:00:00",
                    "initial_state": "transition",
                },
                "weekly_state_forecasts": {
                    "path": "weekly-state-forecasts.csv",
                    "row_count": len(payload["weekly"]),
                    "sha256": "c" * 64,
                },
            },
            "structural_models": {
                "xgb_hazard_destination": {
                    "hazard_model": "binary_xgboost",
                    "destination_model": "xgboost",
                    "direct_jump_floor": 0.000001,
                },
                "causal_dynamic_ensemble": {
                    "experts": ["markov", "xgboost", "xgb_hazard_destination"],
                    "half_life_weeks": 52,
                    "minimum_history_rows": 26,
                    "eligible_loss_rule": "target_date_strictly_before_origin",
                },
                "joint_survival_hazard": {
                    "base_target_weeks": 1,
                    "horizons_weeks": [1, 4, 13],
                    "future_covariates": "origin_values_frozen",
                    "identity": "one_minus_product_one_minus_weekly_hazard",
                },
            },
            "ablation": {
                "anchor_model": "xgboost",
                "reference_variant": "legacy_v3",
                "published_variant": "all_structural",
                "primary_period": "pre_2023_selection_oos",
                "post_2023_role": "retrospective_diagnostic_only",
                "may_change_published_variant": False,
                "manifest_sha256": "a" * 64,
            },
        }
    )
    return payload


def _browser_validation_errors(payload: dict) -> list[str]:
    program = """
const api = require(process.argv[1]);
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => { input += chunk; });
process.stdin.on("end", () => {
  process.stdout.write(JSON.stringify(api.validatePayload(JSON.parse(input)).errors));
});
"""
    completed = subprocess.run(
        ["node", "-e", program, str(JS_PATH)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _browser_history_window_cases() -> list[object]:
    program = """
const api = require(process.argv[1]);
process.stdout.write(JSON.stringify([
  api.resolveHistoryWindow(11, 52),
  api.resolveHistoryWindow(52, 52),
  api.resolveHistoryWindow(51, 52),
  api.resolveHistoryWindow(120, 104),
  api.resolveHistoryWindow(11, "all"),
  api.resolveHistoryWindow(0, 26),
]));
"""
    completed = subprocess.run(
        ["node", "-e", program, str(JS_PATH)],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


class DashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.scripts: list[dict[str, str | None]] = []
        self.links: list[dict[str, str | None]] = []
        self.anchors: list[dict[str, str | None]] = []
        self.inputs: list[dict[str, str | None]] = []
        self.selects: list[dict[str, str | None]] = []
        self.landmarks: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.ids.add(element_id)
        if tag == "script":
            self.scripts.append(attributes)
        elif tag == "link":
            self.links.append(attributes)
        elif tag == "a":
            self.anchors.append(attributes)
        elif tag == "input":
            self.inputs.append(attributes)
        elif tag == "select":
            self.selects.append(attributes)
        if tag in {"header", "main", "footer", "nav", "section", "article"}:
            self.landmarks.add(tag)


def parsed_html() -> DashboardParser:
    parser = DashboardParser()
    parser.feed(HTML_PATH.read_text(encoding="utf-8"))
    return parser


def test_dashboard_assets_are_local_and_present() -> None:
    parser = parsed_html()
    assert HTML_PATH.is_file()
    assert CSS_PATH.is_file()
    assert JS_PATH.is_file()
    assert any(str(script.get("src", "")).startswith("./app.js?") and "defer" in script for script in parser.scripts)
    assert any(str(link.get("href", "")).startswith("./styles.css?") for link in parser.links)

    assert all(not str(script.get("src", "")).startswith(("http://", "https://", "//")) for script in parser.scripts)
    assert all(not str(link.get("href", "")).startswith(("http://", "https://", "//")) for link in parser.links)

    allowed_external_links = {
        "https://sonchanggi.github.io/quant-dashboard/",
        "https://sonchanggi.github.io/fearNgreed/",
        "https://sonchanggi.github.io/momentum-factor-lab/",
        "https://sonchanggi.github.io/dram-price/",
        "https://sonchanggi.github.io/best-factor/",
        "https://sonchanggi.github.io/etf-tracking/",
        "https://sonchanggi.github.io/sox/",
        "https://sonchanggi.github.io/port/",
        "https://sonchanggi.github.io/regime/",
        "https://fred.stlouisfed.org/docs/api/terms_of_use.html",
        "https://www.alphavantage.co/terms_of_service/",
    }
    external_links = {
        href
        for anchor in parser.anchors
        if (href := anchor.get("href")) and href.startswith(("http://", "https://", "//"))
    }
    assert external_links == allowed_external_links

    document = HTML_PATH.read_text(encoding="utf-8").lower()
    assert "//cdn" not in document


def test_required_result_surfaces_exist() -> None:
    required_ids = {
        "app-state",
        "loading-state",
        "error-state",
        "empty-state",
        "dashboard",
        "analysis-date",
        "week-select",
        "latest-week",
        "current-regime-card",
        "next-regime-card",
        "current-probabilities",
        "next-probabilities",
        "probability-shifts",
        "transition-card",
        "probability-chart",
        "probability-chart-wrap",
        "chart-selection-readout",
        "chart-readout-date",
        "history-data-body",
        "factor-scores",
        "regime-timeline",
        "top-drivers",
        "market-context",
        "leaderboard-body",
        "source-health-body",
        "feature-catalog",
        "header-data-as-of",
        "header-analysis-date",
        "header-mode",
        "model-diagnostic",
        "model-loss-chart",
        "transition-horizon-bars",
        "transition-model-section",
        "transition-horizon-select",
        "transition-model-summary",
        "transition-leaderboard-body",
        "shadow-nowcast-summary",
    }
    assert required_ids <= parsed_html().ids


def test_date_controls_support_arbitrary_date_and_exact_week_selection() -> None:
    parser = parsed_html()
    assert any(item.get("id") == "analysis-date" and item.get("type") == "date" for item in parser.inputs)
    assert any(item.get("id") == "week-select" for item in parser.selects)

    script = JS_PATH.read_text(encoding="utf-8")
    assert 'const DATA_URL = "./data/regime-results.json"' in script
    assert "function snapToPriorDate" in script
    assert "dates[middle] <= targetDate" in script
    assert "preserveSnapNote" in script
    assert "window.__REGIME_DASHBOARD__" in script


def test_date_controls_share_one_control_row_and_one_helper_row() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    form_start = document.index('id="date-form"')
    form_end = document.index("</form>", form_start)
    form = document[form_start:form_end]
    first_group_end = form.index("</div>")
    assert 'id="snap-note"' not in form[:first_group_end]
    assert 'id="analysis-date"' in form and 'aria-describedby="snap-note"' in form
    assert 'id="snap-note" class="control-note sr-only"' in form
    assert 'role="status" aria-live="polite"' in form
    assert '"date week steps latest"' in styles
    assert '"note note note note"' in styles
    assert ".date-controls :is(input, select, button)" in styles
    assert "min-height: 44px" in styles


def test_shared_navigation_and_theme_contract_are_explicit() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'class="site-nav"' in document
    assert (
        'href="https://sonchanggi.github.io/regime/" '
        'aria-current="page">Regime</a>'
    ) in document
    assert 'class="section-nav"' in document
    for anchor in ("#overview", "#history", "#evidence", "#models", "#data-health"):
        assert f'href="{anchor}"' in document
    assert 'const THEME_STORAGE_KEY = "quant-research-theme"' in script


def test_chart_exploration_is_single_focus_and_state_isolated() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'id="probability-chart-wrap"' in document
    assert 'tabindex="0"' in document
    assert 'id="chart-selection-readout"' in document
    assert "function handleChartKeydown" in script
    assert "function previewChartDateFromPointer" in script
    assert "chartPinnedDate" in script and "chartPreviewDate" in script
    assert 'circle.setAttribute("tabindex"' not in script


def test_model_selection_and_holdout_diagnostic_are_distinct() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'id="model-diagnostic"' in document
    assert "holdout_diagnostic" in script
    assert "선정 구간" in script
    assert "진단 결과는 선정에 미사용" in script
    assert "is-holdout-best" in script
    assert "2023+ 1위" in script


def test_comparison_visuals_use_existing_probability_and_model_values() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'id="probability-shifts"' in document
    assert 'id="model-loss-chart"' in document
    assert 'id="model-loss-axis"' in document
    assert "function renderProbabilityShifts" in script
    assert "function renderModelLossChart" in script
    assert "getProbability(week.current, code)" in script
    assert "getProbability(week.next_week, code)" in script
    assert 'metricValue(row, ["selection_log_loss"])' in script
    assert 'metricValue(row, ["log_loss", "multiclass_log_loss"])' in script
    assert 'dom["model-loss-axis"].replaceChildren' in script


def test_results_first_layout_keeps_equal_cards_and_wide_history() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    assert 'class="probability-shift-card card" aria-labelledby="probability-shift-title" hidden' in document
    assert 'viewBox="0 0 1200 300"' in document
    assert "width: 1200" in script
    assert "const desiredTicks = Math.min(7, history.length)" in script
    assert "function scrollChartDateIntoView" in script
    assert "requestAnimationFrame(() => scrollChartDateIntoView(state.chartPinnedDate))" in script
    assert ".hero-grid {\n  grid-template-columns: repeat(3, minmax(0, 1fr));" in styles
    assert ".hero-grid > .transition-card {\n    grid-column: auto;" in styles
    assert "#history.analysis-grid" in styles
    assert ".factor-list {\n  margin-top: 16px;\n  grid-template-columns: repeat(4" in styles
    assert ".chart-point.is-active" in styles


def test_full_model_tables_are_collapsed_by_default() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    assert document.count('class="compact-table-details"') == 2
    assert "<summary>전체 모델 표</summary>" in document
    assert "<summary>전체 이탈 모델 표</summary>" in document
    assert 'class="compact-table-details" open' not in document


def test_browser_contract_rejects_probability_keys_beyond_three_states() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "probabilityKeys.length !== STATE_ORDER.length" in script
    assert "확률 키는 표준 세 상태와 정확히 일치" in script


def test_v3_transition_contract_is_additive_and_fail_closed() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'const V3_RESULT_VERSION = "weekly-regime-result-v3"' in script
    assert "TRANSITION_HORIZONS" in script
    assert 'const expectedKeys = ["1w", "4w", "13w"]' in script
    assert 'const exactRiskKeys = ["probability", "target_end", "model", "threshold", "fallback", "fallback_reason"]' in script
    assert "transition_probability와 transition_risk.1w.probability가 일치하지 않습니다" in script
    assert "1주 이탈 확률과 next_week 현재 국면 잔류 확률이 일치하지 않습니다" in script
    assert "payload.model.primary_horizon_weeks !== 1" in script
    assert "payload.model.transition_leaderboard" in script
    assert "payload.model.shadow_nowcast" in script
    assert 'id="transition-horizon-bars"' in document


def test_v4_structural_contract_is_additive_and_fail_closed() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'const V4_RESULT_VERSION = "weekly-regime-result-v4"' in script
    assert 'const V4_MODEL_VERSION = "weekly-nondl-structural-v4"' in script
    assert 'const V4_FEATURE_SET_VERSION = "weekly-pit-structural-v4"' in script
    assert "FROZEN_V4_BASELINE_V3" in script
    assert "FROZEN_V4_STRUCTURAL_PREREGISTRATION" in script
    assert "baseline_v3" in script
    assert "structural_preregistration" in script
    assert "feature_manifest_sha256" in script
    assert "evidence_artifacts" in script
    assert "state-label-history.csv" in script
    assert "weekly-state-forecasts.csv" in script
    assert "xgb_hazard_destination" in script
    assert "causal_dynamic_ensemble" in script
    assert "joint_survival_hazard" in script
    assert "retrospective_diagnostic_only" in script


def test_declared_result_versions_and_no_event_average_precision_fail_closed() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "declaredResultVersion !== V3_RESULT_VERSION" in script
    assert "declaredResultVersion !== V4_RESULT_VERSION" in script
    assert "지원하지 않는 meta.result_version입니다" in script
    assert "row.average_precision === null && eventCount === 0" in script
    assert "무이벤트 구간의 null이어야 합니다" in script
    assert "const binaryLogLoss = strictFiniteNumber(row.binary_log_loss)" in script
    assert 'for (const metric of ["brier", "precision", "recall"])' in script
    assert "function strictFiniteNumber" in script
    assert "function strictProbability" in script
    assert "strictProbability(row.average_precision)" in script


def test_v3_transition_metric_ranges_counts_and_selection_cutoff_are_explicit() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "isIsoDate(payload.model.transition_selection_end)" in script
    assert "model.transition_selection_end는 YYYY-MM-DD 형식의 실제 날짜" in script
    assert "binary_log_loss가 0 이상의 유한한 숫자" in script
    assert '"non_event_count", "fallback_count", "calibration_fallback_count"' in script
    assert "!Number.isInteger(value)" in script
    assert "predictionCount !== eventCount + nonEventCount" in script
    assert "n_predictions는 event_count와 non_event_count의 합" in script


def test_browser_validator_executes_valid_v3_semantic_contract() -> None:
    assert _browser_validation_errors(_valid_v3_browser_payload()) == []


def test_browser_validator_executes_valid_v4_semantic_contract() -> None:
    assert _browser_validation_errors(_valid_v4_browser_payload()) == []


def test_v3_browser_contract_requires_nonempty_generation_id() -> None:
    payload = _valid_v3_browser_payload()
    payload["meta"]["generation_id"] = None
    errors = _browser_validation_errors(payload)
    assert any("generation_id" in error for error in errors)


def test_browser_validator_executes_python_v3_semantic_rejections() -> None:
    cases = [
        ("generation id", lambda payload: payload["meta"].__setitem__("generation_id", "")),
        ("model.version", lambda payload: payload["model"].__setitem__("version", "wrong-v3")),
        ("label version", lambda payload: payload["model"].__setitem__("label_version", "wrong-label")),
        ("feature version", lambda payload: payload["model"].__setitem__("feature_set_version", "wrong-features")),
        (
            "baseline hash",
            lambda payload: payload["model"]["baseline_v2"].__setitem__("payload_sha256", "A" * 64),
        ),
        (
            "baseline required field",
            lambda payload: payload["model"]["baseline_v2"].pop("champion"),
        ),
        (
            "shadow canonical",
            lambda payload: payload["model"]["shadow_nowcast"].__setitem__("canonical_target", True),
        ),
        (
            "horizon target",
            lambda payload: payload["weekly"][0]["transition_risk"]["13w"].__setitem__("target_end", "2026-11-13"),
        ),
        (
            "next-week alias",
            lambda payload: payload["weekly"][0]["next_week"].__setitem__("date", "2026-08-21"),
        ),
        (
            "fallback reason",
            lambda payload: payload["weekly"][0]["transition_risk"]["1w"].__setitem__("fallback_reason", None),
        ),
        (
            "champion keys",
            lambda payload: payload["model"]["transition_champions"].__setitem__("26w", "hazard"),
        ),
    ]
    for label, mutate in cases:
        payload = deepcopy(_valid_v3_browser_payload())
        mutate(payload)
        assert _browser_validation_errors(payload), f"browser validator accepted invalid {label}"


def test_browser_validator_executes_python_v4_semantic_rejections() -> None:
    cases = [
        ("v4 model", lambda payload: payload["model"].__setitem__("version", "wrong-v4")),
        ("v4 feature", lambda payload: payload["model"].__setitem__("feature_set_version", "wrong")),
        ("v3 baseline hash", lambda payload: payload["model"]["baseline_v3"].__setitem__("payload_sha256", "A" * 64)),
        ("v3 baseline valid wrong hash", lambda payload: payload["model"]["baseline_v3"].__setitem__("payload_sha256", "0" * 64)),
        ("v3 baseline inventory valid wrong hash", lambda payload: payload["model"]["baseline_v3"].__setitem__("artifacts_inventory_sha256", "1" * 64)),
        ("v3 baseline champion", lambda payload: payload["model"]["baseline_v3"].__setitem__("champion", "xgboost")),
        ("v3 baseline captured at", lambda payload: payload["model"]["baseline_v3"].__setitem__("captured_at", "2026-08-14")),
        ("v3 baseline shape", lambda payload: payload["model"]["baseline_v3"].__setitem__("extra", True)),
        ("prereg path", lambda payload: payload["model"]["structural_preregistration"].__setitem__("path", "config/structural_v5.json")),
        ("prereg valid wrong hash", lambda payload: payload["model"]["structural_preregistration"].__setitem__("sha256", "0" * 64)),
        ("feature manifest", lambda payload: payload["model"].__setitem__("feature_manifest_sha256", "bad")),
        ("label evidence path", lambda payload: payload["model"]["evidence_artifacts"]["state_label_history"].__setitem__("path", "../state-label-history.csv")),
        ("label evidence hash", lambda payload: payload["model"]["evidence_artifacts"]["state_label_history"].__setitem__("sha256", "bad")),
        ("weekly evidence count", lambda payload: payload["model"]["evidence_artifacts"]["weekly_state_forecasts"].__setitem__("row_count", 2)),
        ("expert order", lambda payload: payload["model"]["structural_models"]["causal_dynamic_ensemble"].__setitem__("experts", ["xgboost", "markov", "xgb_hazard_destination"])),
        ("hazard floor", lambda payload: payload["model"]["structural_models"]["xgb_hazard_destination"].__setitem__("direct_jump_floor", 0.0)),
        ("survival horizon", lambda payload: payload["model"]["structural_models"]["joint_survival_hazard"].__setitem__("horizons_weeks", [1, 4])),
        ("ablation authority", lambda payload: payload["model"]["ablation"].__setitem__("may_change_published_variant", True)),
        ("ablation manifest hash", lambda payload: payload["model"]["ablation"].__setitem__("manifest_sha256", "bad")),
    ]
    for label, mutate in cases:
        payload = _valid_v4_browser_payload()
        mutate(payload)
        assert _browser_validation_errors(payload), f"browser validator accepted invalid {label}"


def test_v3_transition_models_have_horizon_specific_diagnostic_surface() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    assert 'id="transition-model-section"' in document
    assert 'for="transition-horizon-select"' in document
    for value in ("1", "4", "13"):
        assert f'<option value="{value}"' in document
    for label in ("AP ↑", "Precision ↑", "Recall ↑", "False alarms / 연 ↓"):
        assert label in document
    assert "선정 구간 · 2023+ 진단" in document
    assert "2023+ 진단(선정 미사용)" in script
    assert "function renderTransitionModels" in script
    assert "function renderTransitionHorizons" in script
    assert "function renderShadowNowcast" in script
    assert "section.hidden = true" in script
    assert "min-height: 44px" in styles
    assert "#transition-leaderboard-table" in styles


def test_browser_contract_requires_provisional_predeployment_selection_status() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'payload.model.selection_status !== "provisional_predeployment"' in script
    assert "model.selection_status는 provisional_predeployment여야 합니다" in script
    assert 'createElement("span", null, "선정 모델")' in script


def test_research_notices_are_preserved_but_collapsed_below_results() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    overview_end = document.index("</section>", document.index('id="overview"'))
    alerts_position = document.index('id="data-alerts"')
    diagnostic_position = document.index('id="model-diagnostic"')
    assert alerts_position > overview_end
    assert diagnostic_position > overview_end
    assert '<details class="model-review-details">' in document
    assert '<details class="research-notice-details operations-details">' in document
    assert 'id="research-notice-summary"' in document
    assert '<details class="research-notice-details operations-details" open' not in document
    assert "개인·비상업 파생 결과" in document
    assert "renderMethodNotices" in script
    assert "alertCount" in script and "알림 ${alertCount}" in script


def test_default_canvas_uses_compact_copy_and_one_operations_disclosure() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert document.count('class="eyebrow"') == 1
    assert 'class="table-scroll-hint"' not in document
    assert 'class="method-note"' not in document
    assert document.count('class="research-notice-details operations-details"') == 1
    assert 'id="method-notices"' in document
    assert "공개 배포 전 권리 확인 필요:" not in script
    assert "사후 진단 일반화" not in document
    assert "This product uses the FRED® API" in document


def test_dashboard_uses_only_real_payload_values() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "fetch(DATA_URL" in script
    assert "validatePayload(payload)" in script
    assert "DataContractError" in script
    assert "Math.random" not in script
    assert "mockData" not in script
    assert "sampleData" not in script
    assert ".innerHTML" not in script
    assert "DEMO · 모의자료" in script


def test_history_window_never_claims_more_weeks_than_are_available() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert _browser_history_window_cases() == ["all", 52, "all", 104, "all", "all"]
    assert "function syncHistoryWindowControl()" in script
    assert "preferredHistoryWindow: 52" in script
    assert "const requested = state.preferredHistoryWindow" in script
    assert 'option.textContent = available ? `전체 · ${available}주` : "전체"' in script
    assert "option.disabled = weeks > available" in script
    assert "`${range} · ${history.length}주 관측" in script


def test_three_state_and_health_contracts_are_explicit() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    for state_code in ("risk_on", "transition", "risk_off"):
        assert state_code in script
    for health_code in (
        "ok",
        "stale",
        "degraded",
        "quota_exhausted",
        "schema_changed",
        "revision_gap",
        "rights_unconfirmed",
        "license_blocked",
    ):
        assert health_code in script


def test_signed_market_percentages_do_not_use_probability_validation() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "function formatSignedPercent" in script
    assert 'if (format === "percent") return formatSignedPercent(number)' in script
    assert 'if (format === "probability") return formatPercent(number)' in script
    assert "return formatSignedPercent(number);" in script


def test_styles_have_no_remote_assets_or_gradients() -> None:
    styles = CSS_PATH.read_text(encoding="utf-8").lower()
    assert "url(" not in styles
    assert "gradient" not in styles
    assert "@import" not in styles


def test_v4_uses_the_same_transition_dashboard_surfaces_as_v3() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "[V3_RESULT_VERSION, V4_RESULT_VERSION].includes(resultVersion)" in script
    assert "if (!hasTransitionContract || !allRows.length)" in script
