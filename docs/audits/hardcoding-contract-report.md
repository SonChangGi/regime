# Regime 하드코딩·계약 감사 보고서

- 감사 기준일: 2026-08-27
- 범위: 분석 상수, 실행 계약, 감사 대상 선택, lifecycle, generation hash, Python·JavaScript 중복 계약
- 판정 기준: **구현됨**, **synthetic/계약 검증됨**, **repo-local 현행 산출물 검증**, **미통합·미실행**을 구분한다.

## 결론

정답지·운영정책·generation 무결성은 단일화가 크게 진전됐다. label 상수는 [`config/label-spec.json`](../../config/label-spec.json), 실제 운영정책은 [`config/operating-contract.json`](../../config/operating-contract.json), 의미 기반 JSON hash와 lifecycle은 [`src/regime_lab/integrity.py`](../../src/regime_lab/integrity.py)가 담당한다. 웹도 현재 payload의 상태명·색·심벌을 우선 사용하고, JavaScript 상수는 frozen V3/V4 fallback으로 제한했다.

그러나 “하드코딩 제거 완료”라고 판정할 단계는 아니다. 3상태 순서, 일부 상태명, V5 비교모델군과 `0.01` 정책 값은 여러 Python·JavaScript 검증 계층에 남아 있다. 일부는 fail-closed schema 검증에 필요한 의도적 중복이지만, 실행 source of truth와 독립적으로 변하면 drift가 재발할 수 있다. 다만 현재 repo-local reviewed generation `20260826T184946.198911Z`는 generic `selection-family-audit/v2`를 `regime-generation-manifest/2`에 결속했다. 구형 `regime-v5-v4-matched-comparison/1`은 frozen V4 parity 전용 sidecar로 함께 보존되며 generic 전체 후보 감사를 대신하지 않는다.

이번 감사에서 발견된 불리한 결함도 보존한다. 당시 [`scripts/audit_outputs.py`](../../scripts/audit_outputs.py)의 active V5 경로가 schema `2.0.0`, 과거 모델군 및 복잡도 표를 자체 보유하고 있었다. 현재 작업트리에서는 V5 schema·roster·복잡도를 typed 계약에서 읽도록 수정됐고 관련 targeted tests가 통과했다. repo-local `publication/live`와 최종 local package는 V5 dynamic·manifest/2·generic sidecar로 마이그레이션됐고, 개발용 [`web/data/regime-results.json`](../../web/data/regime-results.json)은 frozen V4 Markov fallback으로 남아 있다. 최종 live-derived package는 원격 배포·공개 byte readback과 실제 브라우저 QA까지 완료했다([`release-evidence.md`](release-evidence.md)).

2026-08-27의 private research 실행은 artifact 상태 계약이 실제 산출물에서도 fail-closed임을 추가 확인했다. 최종 source-bound composite label generation은 `complete`이지만 `reconstructed_oos`, `automatic_promotion_eligible=false`, `operating_pipeline_mutated=false`이고, exact-split generation은 `blocked_input_contract`다. shadow audit와 555-origin standard mechanism ablation도 `automatic_promotion_eligible=false`, `public_release_eligible=false`다. 이들 private build를 `publication/live` 세대나 operational evidence로 간주하지 않는다.

## 1. 계약별 source of truth

| 계약 | source of truth | 소비자·검증기 | 현재 판정 |
|---|---|---|---|
| 정답지 | [`config/label-spec.json`](../../config/label-spec.json) | [`label_spec.py`](../../src/regime_lab/analysis/label_spec.py), [`labels.py`](../../src/regime_lab/analysis/labels.py) | 구현·hash lock·v1 회귀 검증 |
| 운영정책 | [`config/operating-contract.json`](../../config/operating-contract.json) | [`operating_contract.py`](../../src/regime_lab/operating_contract.py) | typed·hash-bound, 실행 중 fail-closed |
| immutable preregistration | [`config/structural_v5.json`](../../config/structural_v5.json)와 operating-contract의 SHA-256 | operating-contract loader | 파일 byte hash 불일치 시 실패 |
| V5 public schema | [`src/regime_lab/contract_v5.py`](../../src/regime_lab/contract_v5.py) | Python composer·audit·packager | schema `2.1.0`으로 단일 참조 |
| lifecycle·semantic hash | [`src/regime_lab/integrity.py`](../../src/regime_lab/integrity.py) | promotion, packaging, audit | canonical JSON·허용 상태조합 검증 |
| run lifecycle | [`src/regime_lab/run_registry.py`](../../src/regime_lab/run_registry.py) | local run orchestration | append-only JSONL 상태전이 구현·synthetic 검증 |
| publication generation | [`publication/live/generation-manifest.json`](../../publication/live/generation-manifest.json) | integrity validator·audit CLI | repo-local 현행 generation 결속 검증 |
| selection-family 보조 감사 | [`publication/live/selection-family-audit.json`](../../publication/live/selection-family-audit.json) | generation·promotion·packaging·audit 재구성 검증 | manifest/2 현행 generation 결속, 11후보·365 matched selection origin·`operational_oos`; 공개 readback 완료 |
| private research generation | [`label-bakeoff generation manifest`](../../build/label-bakeoff-final-20260827-r3/runs/20260826T160035.092517Z-80388cd7/generation-manifest.json), [`strict generation manifest`](../../build/label-bakeoff-strict-final-20260827-r3/runs/20260826T155922.233456Z-92120dc9/generation-manifest.json), [`standard mechanism report`](../../build/mechanism-ablation-standard-final-20260827-r2/runs/20260826T163113.864460Z-223b1830/mechanism-ablation-report.json) | label/PIT·mechanism research builder | complete와 blocked 상태 모두 보존; derived-only·private·자동 승격 불가 |
| 웹 상태 표현 | payload `states[]` | [`web/app.js`](../../web/app.js) | payload 우선, frozen legacy fallback만 유지 |

`operating-contract.json`은 현재 공식 champion을 `causal_dynamic_ensemble`, frozen regression baseline을 `markov`로 분리한다. 이는 “공식 V5가 Markov”라는 표현을 허용하지 않는다. Markov는 현재 비교·회귀 기준선이며 공식 운영 모델은 dynamic ensemble이다.

## 2. active audit가 stale V4/V5를 현행으로 오인하던 문제

### 발견 당시 문제

기존 기본 audit 명령은 ignored local payload/artifact 경로를 암묵적으로 사용했고, active publication과 frozen V4의 역할을 구분하지 못했다. 또한 active V5 audit 내부에 다음 값이 독립 복제돼 있었다.

- V5 schema `2.0.0`
- 과거 5개 forecast-comparison model 목록
- V4 기반 standard/full 모델 집합
- 자체 `COMPLEXITY` 순위표

이 상태에서는 운영 contract를 고쳐도 audit가 다른 의미를 검사하거나, stale 파일을 `ok`로 판정할 위험이 있었다.

### 현재 작업트리의 수정

[`scripts/audit_outputs.py`](../../scripts/audit_outputs.py)는 이제 `--target`을 필수로 요구한다.

- `--target publication-live`: `publication/live`의 V5 payload·sidecar·generation manifest와 `operating + reviewed_publication`을 검증한다. V4는 통과할 수 없다.
- `--target local-generation --manifest PATH`: manifest가 지정한 단일 generation의 payload·artifact를 감사한다.
- `--target frozen-v4`: [`src/regime_lab/frozen_v4.py`](../../src/regime_lab/frozen_v4.py)의 immutable baseline만 재현 감사한다.

인자 없는 실행과 `local-generation`의 manifest 누락은 실패한다. active V5 schema와 model roster는 [`contract_v5.py`](../../src/regime_lab/contract_v5.py), active complexity rank는 operating contract에서 읽는다. manifest hash가 알려진 2026-08-25 reviewed generation은 `historical_reviewed_rosters`로 별도 보존해 새 운영군과 섞지 않는다.

파일에 남아 있는 `provisional_predeployment` 검사는 frozen V2–V4 legacy audit 경로의 역사 계약이다. active V5 publication 경로의 lifecycle로 사용되지 않는다.

2026-08-26에 다음 targeted suite를 재실행해 수정 상태를 확인했다.

- `tests/test_audit_outputs.py`
- `tests/test_generation_integrity.py`
- `tests/test_operating_contract.py`

결과는 통과였고 기존 환경 의존 skip 1건이 있었다. 이는 active audit drift의 코드·계약 회귀를 확인한 것이며 full CI, 배포 또는 공개 사이트 검증은 아니다.

## 3. hash와 generation 결속

`canonical_json_sha256_v1`은 UTF-8, key sort, compact separator, `allow_nan=false`를 사용한다. 따라서 공백·들여쓰기·객체 key 순서가 달라도 같은 JSON 의미는 같은 digest를 낸다. [`tests/test_generation_integrity.py`](../../tests/test_generation_integrity.py)는 이 특성을 직접 검증한다.

generation manifest는 다음 edge를 한 세대로 묶는다.

- payload contract hash: manifest back-reference만 제거한 semantic hash
- comparison sidecar contract hash: raw payload back-reference만 제거한 semantic hash
- selection-family sidecar contract hash: generation·policy·후보군·matched-origin·source CSV 재구성 결과
- artifact inventory hash와 file count
- input snapshot as-of와 hash
- label registry/spec hash
- execution spec hash

promotion의 reviewed-candidate hash도 publication 필드를 deterministic pre-publication 상태로 정규화한 뒤 같은 canonical serializer를 사용한다. packaging은 최종 public candidate와 sidecar·manifest를 다시 검증하고, asset cache key는 [`scripts/package_public_demo.py`](../../scripts/package_public_demo.py)가 실제 packaged `app.js`·`styles.css` bytes의 전체 SHA-256으로 생성한다.

현재 repo-local [`publication/live/regime-results.json`](../../publication/live/regime-results.json)은 schema `2.1.0`, generation `20260826T184946.198911Z`, dynamic champion, `selected_by_gate + operating + reviewed_publication`, `evidence_track=operational_oos`다. 명시적 `publication-live` audit와 manifest/2 결속이 통과했고, [`build/pages-workflow-package-final-20260827`](../../build/pages-workflow-package-final-20260827)의 네 data file은 `publication/live`의 payload·generation manifest·두 sidecar와 byte-identical하다. 반면 source-tree 개발용 [`web/data/regime-results.json`](../../web/data/regime-results.json)은 schema `1.0.0`의 frozen V4 Markov fallback이다. [Pages workflow](../../.github/workflows/pages.yml)은 CI·allowlist 검증·배포를 통과했고 공개 8파일 readback도 로컬 승인 패키지와 일치했다. 따라서 **live-derived remote migration까지 확인됐다**([`release-evidence.md`](release-evidence.md)).

## 4. lifecycle 모순 제거

허용 조합은 다음 세 개뿐이다.

1. `selected_by_gate + candidate + unpublished`
2. `selected_by_gate + reviewed + unpublished`
3. `selected_by_gate + operating + reviewed_publication`

`reviewed_publication + provisional_predeployment`, `reviewed_publication + candidate`, meta/model alias 불일치는 Python integrity validator와 browser contract 양쪽에서 거부된다. 이 중복 검사는 보안 경계상 의도적이지만, 허용 조합의 기준 데이터는 operating contract와 Python validator가 같이 읽도록 유지해야 한다.

immutable preregistration, 운영정책, 실행 상태도 분리됐다.

- preregistration: 원본 파일과 byte hash
- operating contract: 현재 코드가 읽는 typed policy
- run registry: `started → collecting → analyzing → completed → publication_reviewed → published` 또는 명시적 중단 상태의 append-only 이벤트

[`build/weekly-automation/generation-v5/run-registry.jsonl`](../../build/weekly-automation/generation-v5/run-registry.jsonl)에는 run `20260826T163631.770407Z-live-build`의 `started → collecting → analyzing → completed` 전이가 실제로 기록됐다. 이는 단일 local generation의 운영 이력 확인이며, 장기 누적이나 원격 publication 완료 증거는 아니다. 이전 중단 run도 별도 registry에 보존된다.

## 5. Python·JavaScript 중복과 남은 하드코딩

### 의도적으로 유지할 중복

- `risk_on`, `transition`, `risk_off` 순서: wire schema를 fail-closed로 검증하는 최소 상수다.
- V3/V4 model·artifact 상수: frozen 결과의 byte-level 재현과 regression audit에 필요하다.
- browser의 `FROZEN_LEGACY_STATE_META`: 과거 payload를 읽는 fallback이며 active V5 상태 메타의 source가 아니다.
- schema별 required-field 목록: 서로 다른 실행환경에서 손상된 payload를 독립 거부하기 위해 필요하다.

### 통합이 더 필요한 부분

1. `STATE_ORDER`는 [`schema.py`](../../src/regime_lab/schema.py), [`contract_v5.py`](../../src/regime_lab/contract_v5.py), 분석 모듈, audit script, JavaScript에 반복된다. wire-level 상수 자체는 유지하되 build 시 operating contract와 parity를 자동 생성·검사하는 편이 안전하다.
2. [`src/regime_lab/v5.py`](../../src/regime_lab/v5.py)의 `STATE_LABELS_KO`는 operating contract의 `label_ko`와 중복된다. composer가 typed state definitions만 사용하도록 통합해야 한다.
3. `V5_FORECAST_COMPARISON_MODELS`는 현재 [`contract_v5.py`](../../src/regime_lab/contract_v5.py)에 11개 모델로 명시돼 있다. weekly base/core/ensemble component를 합성하는 deterministic function과 exact-order contract로 대체하면 roster drift 면적이 줄어든다.
4. `0.01`은 materiality, Brier tolerance, simplicity tolerance라는 서로 다른 의미로 사용된다. 현재는 operating contract에 이름을 분리했지만 [`web/app.js`](../../web/app.js), promotion, audit의 일부 validator에도 숫자가 반복된다. browser에는 policy hash와 named fields를 전달하고, Python은 typed policy에서 읽되 frozen legacy만 별도 상수로 남겨야 한다.
5. [`scripts/audit_outputs.py`](../../scripts/audit_outputs.py)의 큰 V2–V4 상수군은 재현에는 필요하지만 active 코드와 한 파일에 공존한다. `frozen_audit/v2_v4`와 active V5 감사 모듈을 분리하면 잘못된 분기 진입 가능성과 리뷰 비용이 낮아진다.
6. [`src/regime_lab/selection_family_audit.py`](../../src/regime_lab/selection_family_audit.py)의 generic `selection-family-audit/v2`는 generation·manifest·promotion·packaging·audit에 연결됐고, 현재 [`publication/live/selection-family-audit.json`](../../publication/live/selection-family-audit.json)에 실제 생성됐다. [`publication/live/v5-vs-v4-comparison.json`](../../publication/live/v5-vs-v4-comparison.json)은 V5–frozen-V4 byte parity라는 별도 목적의 전용 schema로 함께 남는다. 두 sidecar가 manifest/2의 서로 다른 edge로 결속되므로 어느 하나를 다른 하나로 오인하면 안 된다.

## 6. 웹 계약 상태

현재 [`web/app.js`](../../web/app.js)의 `stateMeta(code, payload)`는 `payload.states[]`를 먼저 사용한다. `applyPayloadStateTheme`가 CSS custom property, visible label, symbol을 payload에서 적용한다. [`web/index.html`](../../web/index.html)에는 다음 카드가 추가돼 있다.

- canonical label의 SPY 추세·변동성·낙폭 정의
- broad market·macro가 predictor/challenger라는 구분
- 관측 소속도와 다음 주 예측확률의 의미 차이
- origin, 실제 decision 시각, target, 남은 horizon
- target이 지난 forecast의 만료 안내

DOM·계약·접근성 테스트와 JavaScript syntax 검사에 더해, 최종 [`build/pages-workflow-package-final-20260827`](../../build/pages-workflow-package-final-20260827)을 실제 로컬 브라우저에서 확인했다. 이 package는 manifest/2와 `selection-family-audit/v2`를 포함해 optional fallback 없이 current generation을 읽으며, payload·generation manifest·selection-family sidecar·V5–V4 comparison이 `publication/live`와 byte-identical하다. 기존 viewport·dark mode·keyboard·예측 시계·소속도/확률 분리·console error 검사는 최종 live-derived package에서 통과했다. 현재 target은 유효하므로 만료 forecast 숨김은 DOM 계약 테스트 증거다. source `web/`를 직접 띄우면 frozen V4 fallback을 읽지만 최종 package의 data path와는 다르다. 같은 검사를 공개 URL에서 1280·1024·390px, light/dark, console error 0으로 다시 통과했다.

## 7. 완료 게이트

하드코딩·계약 작업을 완료로 승격하려면 다음을 모두 만족해야 한다.

1. operating contract와 Python·JavaScript의 state order, state meta, model roster, selection policy parity를 독립 재계산한다.
2. 이후 reviewed generation에서도 generic selection-family sidecar가 payload·policy·origin hash 및 source CSV와 결속되는지 회귀 확인한다. 첫 manifest/2 현행 generation 생성은 완료됐다.
3. payload와 다른 generation의 sidecar·artifact를 섞은 모든 조합이 audit·packaging에서 실패하는지 확인한다.
4. formatting·key order만 바꾼 JSON은 같은 canonical hash, semantic value를 바꾼 JSON은 다른 hash가 되는지 유지한다.
5. frozen V4 audit은 성공하되 `publication-live`로는 절대 성공하지 않는지 CI에서 확인한다.
6. full test suite, lint/type checks, public package verification, browser viewport·dark·keyboard QA를 완료한다.
7. 별도 승인된 경우에만 commit·push·deploy 후 원격 파일 hash와 DOM을 readback한다.

## 8. 현재 판정

- label·operating contract 단일화: **구현·계약 검증됨**.
- canonical promotion/packaging hash: **구현·synthetic 및 repo-local generation 검증됨**.
- stale active audit 문제: **발견 내용 보존, 현재 작업트리에서 수정·targeted test 검증됨**.
- lifecycle 모순: **Python·browser fail-closed 계약으로 수정됨**.
- 웹 상태 메타: **payload 우선으로 수정, frozen fallback 유지**.
- migration 상태: **repo-local `publication/live`와 최종 local package는 V5 dynamic·operational_oos·manifest/2로 결속; `web/data`는 frozen V4 fallback; live-derived remote 배포·readback 검증 완료**.
- generic selection-family sidecar: **builder·validator·generation/promotion/package/audit 결속 구현, current reviewed generation과 최종 package에 통합; 11후보·365-origin 보조평가가 champion을 바꾸지 않음**.
- private label·shadow·standard-ablation artifact: **source-bound reconstructed/private 계약으로 생성됐으며 publication generation과 분리됨; 555-origin standard ablation과 strict exact-split run의 차단 상태를 모두 보존**.
- Python·JavaScript 하드코딩: **감소했으나 일부 중복 남음**.
- 원격 배포·public readback: **CI·Pages·8파일 byte parity·실제 브라우저 QA 완료**.
