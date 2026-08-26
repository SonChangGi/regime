# Regime PIT·운영 OOS 감사 보고서

- 감사 기준일: 2026-08-27
- 범위: `t → t+1` 시간 인과성, 시장자료 first-seen, revision replay, forecast ledger, generation binding
- 핵심 구분: **operational OOS**는 당시 시스템이 실제로 본 입력과 발행 기록이고, **reconstructed OOS**는 공식 release 시각 또는 최신 archive로 재구성한 역사 연구다.

## 결론

모형 배열 자체는 `S[t+1]`을 target으로 사용하며 공식 `gap=1`에서 모든 학습 target을 예측 origin보다 엄격히 과거로 제한한다. 현재 보존된 실데이터 split 554개를 다시 확인한 결과 전부 `last_train_target < origin < target`이고 `gap=1`이다. fold별 estimator·imputer·scaler도 해당 train prefix에만 적합된다.

가장 중요한 결함은 통계 코드의 단순한 `shift` 오류가 아니라 **시장자료 운영 빈티지**였다. 과거 Alpha Vantage 신규 weekly bar를 금요일 cutoff 시각으로 소급하면, 실제로는 며칠 뒤 처음 받은 값으로 과거 상태와 확률을 다시 만들 수 있었다. 이 결과는 pseudo-OOS이지 당시 발행 예측이 아니다.

이번 변경은 네 개의 시간축, source/operational as-of join, append-only ledger와 generation manifest를 구현했다. 이후 composite replay는 실제 수집된 close·dividend·adjusted-close 행으로 1,077주 reconstructed panel을 만들었지만, provider-current adjusted close에서 implied split을 구성했기 때문에 historical PIT 또는 operational OOS가 아니다. exact split을 요구한 strict replay는 12개 ETF의 split series 누락을 숨기지 않고 차단됐다. 별도로 reviewed generation `20260826T184946.198911Z`는 `operational_oos` 입력으로 생성돼 ledger 1건과 completed run-registry 이력을 남겼고, 공식 label과 dynamic champion을 유지했다. 이 generation은 이후 CI·Pages·8파일 byte readback과 실제 브라우저 QA를 통과했다([`release-evidence.md`](release-evidence.md)). 이는 실제 공개 전달 증거이지만 장기 prospective 성능 증거는 아니다.

## 1. `t → t+1` 학습 경계

[`src/regime_lab/analysis/validation.py`](../../src/regime_lab/analysis/validation.py)의 1주 benchmark는 다음 순서로 작동한다.

1. `next_states = states.shift(-1)`로 origin `t`의 target을 `S[t+1]`로 만든다.
2. 각 `test_position=t`에 대해 `train_stop=t-gap`으로 train prefix를 자른다.
3. 공식 `gap=1`이면 마지막 train origin은 `t-2`, 그 origin의 target은 `t-1`이다.
4. 따라서 학습 target이 완료된 뒤 한 주를 purge하고 origin `t`의 feature로 `t+1`을 예측한다.
5. feature frame, label, current state, calibration history, threshold history는 모두 origin 이후 행을 배제한다.

실제 보존 산출물 [`build/nonmarkov-actual-replay-20260826/artifacts/walk-forward-splits.csv`](../../build/nonmarkov-actual-replay-20260826/artifacts/walk-forward-splits.csv)을 재계산한 결과:

| 검사 | 결과 |
|---|---:|
| split 수 | 554 |
| `last_train_target < origin < target` | 554 / 554 |
| `gap=1` | 554 / 554 |

이는 통계 배열의 인과성이 맞다는 증거다. 하지만 입력 value가 그 origin 당시에 실제로 존재했는지까지 보장하지는 않는다. 그 별도 문제를 다음 절의 bitemporal 계약이 해결한다.

## 2. 2026-08-14 불리한 빈티지 사례

고정 회귀 fixture는 SPY 2026-08-14 weekly bar에 다음 시계를 부여한다.

| 시계 | UTC |
|---|---|
| 주간 관측기간 끝 | `2026-08-14` |
| 16:00 ET forecast cutoff | `2026-08-14T20:00:00Z` |
| source weekly finalization sensitivity | `2026-08-14T20:15:00Z` |
| provider first seen | `2026-08-20T01:33:35Z` |
| system retrieved | `2026-08-20T01:33:35Z` |

따라서 이 bar는 8월 14일 16:00 decision에도, 8월 15일 replay에도 operational input이 될 수 없다. `availability_basis=operational`에서는 8월 20일 최초 수신시각부터만 보인다. [`tests/test_data_bitemporal.py`](../../tests/test_data_bitemporal.py)가 이를 고정 검증한다.

보존된 두 산출물은 결론이 입력 vintage에 민감했음을 보여준다.
두 ignored build의 직접 확인값과 payload SHA-256은
[`tests/fixtures/pit-replay-2026-08-14.json`](../../tests/fixtures/pit-replay-2026-08-14.json)에
불리한 회귀 증거로 동결했다.

| 산출물 | 2026-08-14 관측 hard state | 관측 membership/구버전 확률 | 2026-08-21 Markov Risk-on 예측 |
|---|---|---|---:|
| [`build/weekly-automation/generation/regime-results.json`](../../build/weekly-automation/generation/regime-results.json) | `transition` | Risk-on 47.07%, Transition 49.34% | 12.43% |
| [`build/v5-next-preview/regime-results.json`](../../build/v5-next-preview/regime-results.json) | `risk_on` | Risk-on 49.74%, Transition 47.15% | 88.67% |

이 표는 서로 다른 세대 payload의 표시 필드명도 다르므로 확률의 semantic identity까지 동일하다고 주장하지 않는다. 그러나 같은 주의 hard state와 다음 주 Markov 예측이 크게 바뀐 사실은 직접 확인된다. 불리한 이 사례는 삭제하거나 평균으로 희석하지 않고 vintage 회귀의 고정 근거로 보존한다.

## 3. 구현된 bitemporal 계약

[`src/regime_lab/data/contracts.py`](../../src/regime_lab/data/contracts.py)의 `Observation`은 다음 identity를 저장한다.

- `observed_period_end`
- `source_released_at`
- `provider_first_seen_at`
- `system_retrieved_at`
- `revision_seq`
- `raw_sha256`

legacy `released_at`, `available_at`, `retrieved_at`은 기존 snapshot과의 호환을 위해 남긴다. operational eligibility는 `max(source_released_at, provider_first_seen_at)`이고, 시스템 수신도 decision 이전이어야 한다.

[`src/regime_lab/data/asof.py`](../../src/regime_lab/data/asof.py)의 `weekly_asof_join`은 가용성 기준을 명시적으로 요구한다.

- `availability_basis="source"`: historical/reconstructed 연구용 `available_at <= cutoff`
- `availability_basis="operational"`: `max(source_released_at, provider_first_seen_at) <= cutoff`

SQLite store는 새 세 열을 보존하고, writable legacy DB는 schema migration과 backfill을 수행한다. read-only legacy DB는 파일을 변경하지 않고 기존 시계로 명시 필드를 합성한다. 이 호환은 operational truth를 새로 만들어 낸다는 뜻이 아니며, legacy 행의 첫 수신시각은 당시 `retrieved_at` 이상으로만 해석해야 한다.

Alpha Vantage 신규 weekly row의 `available_at`을 금요일 16:00으로 덮어쓰던 collection override는 제거됐다. source finalization은 16:15 sensitivity로 보존하고 operational join은 다시 first-seen의 최대값을 사용한다.

PIT 총수익 challenger용 시장 입력도 feature 입력과 분리했다. [`config/series.json`](../../config/series.json)의 `research_fields=["dividend_amount"]`는 Alpha Vantage `TIME_SERIES_WEEKLY_ADJUSTED` 응답의 주간 배당액을 별도 연구 series로 보존한다. 최종 source-bound composite replay의 [`pit-replay-report.json`](../../build/label-bakeoff-final-20260827-r3/runs/20260826T160035.092517Z-80388cd7/pit-replay-report.json)에는 SPY·RSP·IWM·9개 섹터 ETF의 `dividend_amount`가 각각 2006-01-06–2026-08-21의 1,077개 finite row, 합계 12,924 row-slot으로 기록됐고 source-release 누락은 0이다. 이는 현금배당 event가 12,924번이라는 뜻이 아니라 0을 포함한 주간 관측행 수다. 어댑터는 이 행에 `research_role=pit_corporate_action_input`을 기록하고, [`src/regime_lab/dataset.py`](../../src/regime_lab/dataset.py)는 `research_fields`를 모델 required-series와 feature frame에서 제외하므로 기존 예측모형에 암묵적으로 들어가지 않는다.

[`src/regime_lab/analysis/pit_total_return.py`](../../src/regime_lab/analysis/pit_total_return.py)는 raw close, cash dividend, split coefficient와 다섯 빈티지 identity를 입력으로 받는다. `operational_oos`는 source release·first-seen·retrieval이 모두 decision 이전이어야 하고, `reconstructed_oos`도 first-seen/retrieval만 완화할 뿐 `source_released_at <= decision_at`을 반드시 만족해야 한다. 둘의 이름은 `operating-contract.json`에서 직접 읽어 다른 계약 계층과 drift하지 않는다. symbol별 result는 동일 evidence track·decision index·corporate-action 계약·input snapshot hash를 검증한 `PITTotalReturnPanel`로 묶여야만 v2 labeler에 들어가며, 임의로 이름만 바꾼 adjusted-close DataFrame은 거부된다. weekly Alpha 응답은 exact split coefficient를 주지 않는다([Alpha Vantage documentation](https://www.alphavantage.co/documentation/)). 그래서 최종 [`strict` replay](../../build/label-bakeoff-strict-final-20260827-r3/runs/20260826T155922.233456Z-92120dc9/pit-replay-report.json)는 SPY·RSP·IWM·XLB·XLE·XLF·XLI·XLK·XLP·XLU·XLV·XLY의 12개 `split_coefficient`가 모두 없음을 보고하고 `status=blocked_input_contract`, `replay_completed=false`로 끝났다. 단위 split 채움도 허용하지 않았다.

별도 composite sensitivity는 close·dividend·adjusted close의 공통 1,077주를 맞추고 current adjusted close에서 implied coefficient를 구성했다. 모든 12개 symbol이 `reconstructed_eligible_rows=1077`이었지만 `operational_eligible_rows_at_reconstructed_decision_clock=0`이었다. 공통 origin hash는 `f7a577b8f36f1a2f6d59d2325c3f4e6a48654abdf2bcb69b7ec8032f6f54cba7`이다. 이 경로는 `provider_first_seen_relaxed=true`, `adjusted_close_composite_is_historical_pit=false`, `adjusted_close_composite_algebraically_reproduces_adjusted_returns=true`를 명시한다. 따라서 실제 배당행 수집과 주간 정렬은 확인됐지만, exact split·당시 first-seen이 있는 PIT replay로 승격된 것은 아니다.

## 4. evidence track

| track | 허용 시계 | 용도 | 승격 증거 역할 |
|---|---|---|---|
| `operational_oos` | source finalization, provider first seen, system retrieval이 모두 decision 이전 | 실제 시스템 예측 | 운영 성능의 1차 증거 |
| `reconstructed_oos` | 공식 archive/release 시각으로 역사 재구성 | 긴 역사 연구·민감도 | 운영 발행 기록으로 표현 금지 |

Philadelphia Fed RTDSM의 real-time vintage 원칙을 적용하되([RTDSM](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists)), archive availability와 이 시스템의 first-seen을 합치지 않는다.

현재 repo-local [`publication/live/regime-results.json`](../../publication/live/regime-results.json)은 schema `2.1.0`, generation `20260826T184946.198911Z`, `evidence_track=operational_oos`, `origin_at=2026-08-21T20:00:00Z`, `decision_at=2026-08-26T18:49:46.198911Z`, `target_at=2026-08-28T20:00:00Z`다. 공식 champion은 `causal_dynamic_ensemble`이고 lifecycle은 `selected_by_gate + operating + reviewed_publication`이다. [`publication/live/generation-manifest.json`](../../publication/live/generation-manifest.json)은 schema `regime-generation-manifest/2`로 payload, V5–V4 comparison, [`selection-family-audit/v2`](../../publication/live/selection-family-audit.json), 39-file artifact inventory, input snapshot, label spec, execution spec을 같은 generation에 묶는다.

[`build/pages-workflow-package-final-20260827`](../../build/pages-workflow-package-final-20260827)은 위 네 data file을 `publication/live`와 byte-identical하게 담은 최종 로컬 `personal_noncommercial_live_derived` 패키지이며 raw observation을 포함하지 않는다. 이 패키지는 로컬 브라우저 QA 뒤 Pages에 배포됐고, 공개 URL에서 다시 받은 8개 파일도 전부 byte-identical했다([`release-evidence.md`](release-evidence.md)). 개발용 [`web/data/regime-results.json`](../../web/data/regime-results.json)이 frozen V4인 것은 source-tree legacy fallback과 최종 패키지 입력이 다르기 때문이다.

## 5. append-only forecast ledger

[`src/regime_lab/forecast_ledger.py`](../../src/regime_lab/forecast_ledger.py)는 SQLite append-only ledger를 구현한다. primary key는 다음 여섯 필드다.

`(origin_week, decision_at, target_at, label_spec_sha256, model_manifest_sha256, input_snapshot_sha256)`

각 행은 당시 current state/membership, 공식 champion 확률, model별 확률, selection, lifecycle, local publication 시각과 정확한 operational input revision 목록을 함께 보존한다.

보호 규칙:

- `decision_at < target_at`
- 모든 input period가 origin 이하
- 모든 `provider_first_seen_at`, `operating_available_at`, `system_retrieved_at`가 decision 이하
- 동일 key의 동일 내용 재삽입도 실패
- 동일 key의 다른 내용은 conflict로 실패
- SQLite trigger로 `UPDATE`·`DELETE` 금지
- local ledger 파일 권한 `0600`

CLI는 operational dataset일 때만 ledger entry를 구성하고 local generation cutover callback에서 append한다. 현재 [`forecast-ledger.sqlite3`](../../build/weekly-automation/generation-v5/forecast-ledger.sqlite3)에는 실제 operational row 1건이 있다. 그 key는 origin `2026-08-21`, decision `2026-08-26T18:49:46.198911Z`, target `2026-08-28T20:00:00Z`, label spec hash `bec1600aea104985405d0d5c2b3706088b885ef7aa788c57c9575aa884c5f3a7`, model manifest hash `c167514c2f5daa8353dc3a2ba951f7a8cc0aca16076ad78c3b77e611258093ca`, input snapshot hash `52373b551fcdc194af233d6c29a7da2930efb9b8077cae468254b3ab70582028`이다. [`run-registry.jsonl`](../../build/weekly-automation/generation-v5/run-registry.jsonl)은 run `20260826T163631.770407Z-live-build`의 `started → collecting → analyzing → completed` 전이와 같은 generation id를 보존한다. 이전 중단 이력은 별도 [`run-registry.jsonl`](../../build/weekly-automation/run-registry.jsonl)에 남아 있으며 삭제하지 않았다. 따라서 구조·synthetic뿐 아니라 **단일 로컬 operational forecast의 append 증거가 확인됐지만**, 표본 1건은 prospective 성능평가가 아니다.

## 6. generation·lifecycle 무결성

[`src/regime_lab/integrity.py`](../../src/regime_lab/integrity.py)는 whitespace와 key order에 무관한 `canonical_json_sha256_v1`을 사용한다. `generation-manifest.json`은 다음을 묶는다.

- payload contract hash
- comparison sidecar contract hash
- selection-family sidecar contract hash
- artifact inventory hash와 file count
- input snapshot as-of와 hash
- label spec registry/spec hash
- execution spec hash

payload의 manifest back-reference와 sidecar의 payload raw-byte binding도 재검증한다. lifecycle은 다음 조합만 허용한다.

- `selected_by_gate + candidate|reviewed + unpublished`
- `selected_by_gate + operating + reviewed_publication`

과거의 `reviewed_publication + provisional_predeployment` 모순은 Python과 browser contract에서 거부된다.

## 7. 남은 운영 완료 기준

1. 완료된 첫 operational weekly build와 같은 계약으로 다음 주 이후 generation을 누적해 ledger가 단일 행을 넘어서는지 확인한다.
2. 이미 수집된 주간 배당행의 향후 first-seen을 계속 보존하고, 별도 권리·시각 계약을 통과한 exact split event archive를 연결한다.
3. 새 generation마다 모든 사용 input revision을 ledger에 append하고 entry 수·hash를 독립 확인한다.
4. 같은 DB에 origin 이후 신규 bar/revision을 추가해 기존 ledger row가 byte-identical인지 확인한다.
5. 같은 입력을 `reconstructed_oos`로 재생할 때 별도 revision/산출물로 기록되는지 확인한다.
6. 휴장일, DST 전환, 월말, same-day release, 16:00/16:15 경계, late response를 full pipeline까지 통과시킨다.
7. target이 지난 latest forecast는 UI 현재 카드에서 숨기고 역사에는 남는지 DOM 계약 테스트로 확인한다. 실제 브라우저에서는 아직 유효한 현재 target과 남은 horizon 표시를 확인한다.
8. local tests, CI, package, 공개 배포, public readback을 별도 증거로 남긴다. 이번 변경은 local 구현·회귀, 최종 local package·browser QA, 원격 CI·Pages, 공개 byte readback까지 각각 확인했다.

## 8. 현재 판정

- `t → t+1` supervised split: **확인됨**.
- fold-local 적합과 `gap=1`: **확인됨**.
- first-seen 소급 방지: **코드·fixture 검증됨**.
- raw-close/dividend/split PIT 재구성: **코드·synthetic 검증됨; 12개 ETF dividend 각 1,077행과 reconstructed composite panel은 생성됨. 그러나 exact split strict replay는 12/12 입력 누락으로 차단됐고 operational eligible row는 0**.
- 2026-08-14 pseudo-OOS 문제: **불리한 사례로 보존됨**.
- append-only ledger: **구현·synthetic 검증 및 단일 operational forecast row 확인됨; 장기 누적·revision 불변성 운영 증거는 아직 부족**.
- 현재 reviewed generation: **`operational_oos`, dynamic champion, manifest/2·selection-family-audit/v2 결속 및 공개 배포·readback 확인**.
- historical label·shadow·5-track 연구: **계속 reconstructed OOS이며 operational generation과 혼합 금지**.
- 원격 배포·public readback: **완료**. 실제 prospective 성능 주장: **아직 불가**.
