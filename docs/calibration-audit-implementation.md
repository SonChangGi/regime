# 전환 확률 보정·평가 구간 방어 구현

2026-09-07 구현. 데이터 기준은 **2026-09-04 20:00 UTC**, 기존 generation은 `20260906T200946.236868Z`다. 기존 파생 OOS와 미성숙 예측을 재생했으며, 원자료/API 재감사·기초 모델 재학습·운영 DB 변경·live 발행은 수행하지 않았다.

## 구현한 계약

- `TransitionCalibrator`, 버전 **`transition-calibration/2`**. 후보는 identity, Platt, `0.75 × raw + 0.25 × Platt` 세 개로 고정한다. Platt 입력은 raw logit, `C=0.10`, solver `lbfgs`, max_iter 1000이다. 후보와 설정을 진단 성능에 맞춰 탐색하지 않았다.
- UTC `2000-01-07`을 기준으로 나눈 **26주 달력 블록**마다 선택과 계수를 고정한다. 모델·horizon별 최근 최대 3개 종료 블록에서 같은 origin의 binary Log loss를 비교한다. 각 검증 블록의 Platt 적합은 **`training.target_end < validation_block_start`**를 만족해야 한다. 선택에는 **`validation.target_end < selection_as_of`**인 성숙 표본만 사용한다. 달력 블록이 종료됐어도 아직 target이 확정되지 않은 행은 제외된다.
- 적합은 최소 `max(52, minimum_inner_predictions)`행, 비교는 최소 `max(26, minimum_inner_predictions)`행과 사건·비사건 각각 3건을 요구한다. 표본 부족 또는 적합 불가에는 사유가 있는 identity fallback을 반환한다. 비교에서 이긴 identity는 **정상 선택이며 fallback=False**다. 수치적 동률은 identity→축소→Platt 순서다.
- **2023-01-01 이후에는 선택과 계수를 모두 동결**한다. 기준일 이전 selection 행만 허용한다. 진단 표본을 나중에 추가하거나 그 결과를 바꿔도 선택·계수가 바뀌지 않는다. 이 규칙은 다항 인과 보정 연구의 기존 순차 적합 정책과 별개다.
- 로컬 캐시는 각 모델·horizon·시간 블록의 결과를 재사용한다. 블록당 적합은 검증용 최대 3회와 최종 적합 최대 1회다. 행별 반복 최적화나 추가 라이브러리는 없다.
- 전환 OOS·prospective 행에 버전, 선택 시각·범위, training/validation의 마지막 target과 표본 수, validation 블록 수, 후보별 Log loss, 축소 가중치를 기록한다. 기존 `p_change`, `calibration_method`, fallback 필드와 **4개 값 반환 인터페이스**는 유지한다.

## split 방어

연구 비교기, 직접 점수화/요약/paired-comparison 진입점, 선정 보조 평가기, 인과 보정 입력에서 제공된 split 문자열과 독립적으로 날짜를 검사한다.

- selection: `origin < target < 2023-01-01`.
- diagnostic: `2023-01-01 <= origin < target`.
- **`origin < cutoff <= target`은 양쪽 모두에서 의도적으로 제외**된다. 이를 selection 또는 holdout으로 표기하면 거부한다.
- 기준선과 후보의 split을 함께 잘못 바꿔 서로 일치해도 거부한다. 선정 평가의 기존 문서 형식·해시 구성은 바꾸지 않았다.

## 실제 파생 자료 재생 결과

`transition-oos-predictions.csv`의 **9,738행**, `transition-candidate-forecasts.csv`의 **108행**, 18개 모델·기간 조합을 재생했다. 아래는 해당 기간의 **동일 origin**에서 비교한 값이다. 4주는 188개, 13주는 179개 성숙 origin이며, 서로 다른 기간끼리 동일 표본이라고 주장하지 않는다.

| 기존 발행 모델 | 성숙 origin | 기존 보정 LL | 새 보정 LL | 기존 보정 Brier | 새 보정 Brier | cutoff에서 선택한 방식 |
|---|---:|---:|---:|---:|---:|---|
| 4주 binary_xgboost | 188 | 0.752858 | 0.730382 | 0.278280 | 0.265596 | identity |
| 13주 markov_hazard | 179 | 0.442010 | 0.442010 | 0.141378 | 0.141378 | Platt |

위 표는 **최종 정합화 전**이다. 선정 구간의 순차 재생 결과도 JSON에 포함했다. 13주 Markov의 선정 LL은 기존 0.635326→새 정책 0.582620이었고, 4주 XGBoost는 0.633713→0.636350으로 소폭 나빠졌다. 개선만 선별해 보고하지 않는다.

현재 발행 당시의 4·13주 모델과 **공식 1주 이탈확률을 고정**하고, 기존 one-week-anchored coherence를 identity 및 새 보정의 4·13주 값에 각각 함께 적용했다.

| 기간 | 기존 실제 발행 최종 LL | identity 최종 LL | 새 보정 최종 LL | 기존 실제 발행 최종 Brier | identity 최종 Brier | 새 보정 최종 Brier |
|---|---:|---:|---:|---:|---:|---:|
| 4주 / 188 origin | 0.738952 | 0.722548 | 0.722611 | 0.271567 | 0.261853 | 0.261992 |
| 13주 / 179 origin | 0.437687 | 0.319578 | 0.430576 | 0.139255 | 0.088981 | 0.135796 |

**13주 과보정 문제를 완전히 해소했다는 결과는 아니다.** 과거 69개 비교 origin에서 identity LL=0.799479, 축소=0.768430, Platt=0.688968이므로 동결 정책은 Platt를 선택했다. 이미 검토한 2023년 이후 진단의 identity 우위를 이유로 선택을 바꾸지 않았다. 13주 새 최종 수치의 개선은 4주 보정 변경에 따른 공동 정합화 효과이며 13주 Platt 자체가 개선된 것은 아니다. 이 결과는 새 미관측 홀드아웃 또는 재학습한 전체 시스템의 성능 증거가 아니다.

최신 origin의 비교는 다음과 같다. 미성숙 예측이므로 성과 점수를 붙이지 않았다.

| 기간 / target | 기존 실제 발행 | identity + coherence | 새 선택 보정 + coherence |
|---|---:|---:|---:|
| 4주 / 2026-10-02 | 50.678686% | 68.322277% | 65.986943% |
| 13주 / 2026-12-04 | 63.651609% | 77.767156% | 65.986943% |

192개 주의 identity·새 보정 최종 결과 모두 `p1 <= p4 <= p13`을 만족했고, 공식 1주 anchor 변경은 0건이다.

## 로컬 미리보기 JSON 계약

파일: `build/forecast-audit-improvements/calibration/calibration-audit.json`.

- `schema_version = regime-calibration-audit/1`, `role = research_preview_not_issued_forecast`, `data_as_of`, `source_generation_id`, `selection_protocol`을 제공한다.
- `rows` 36개: 모델·기간·split별 동일 표본 비교다. `raw_*`는 identity, **`calibrated_*`는 새 선택 보정**, **`final_*`는 새 보정의 최종 정합화**다.
- **`previous_calibrated_*`는 기존 보정**, **`previous_published_*`는 실제 과거 발행 최종 수치**다. 기존과 새 결과를 같은 이름으로 합치지 않는다. `identity_final_*`도 별도로 제공한다.
- final 비교가 실제로 계산된 4주 XGBoost·13주 Markov 진단 행만 `final_available=true`다. 나머지 final은 **null이며 0이 아니다**. 선정 행의 `selected_method=past_block_selection`은 과거 블록별 선택이며, `latest_selected_method`가 cutoff에서 동결한 최신 방식이다.
- `latest_rows` 18개에 원확률, 새 보정 전/후 정합화, 기존 보정, 기존 실제 발행확률을 각각 제공한다. `published_horizon_comparison`에는 실제 발행·identity·새 최종 결과의 주별 history와 최신값이 있다.
- `model_horizon_results`는 자세한 후보 선택 점수·범위·표본과 두 split의 지표다. `sources`에 입력 파일 SHA-256을 기록했다.

재생 명령:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/replay_transition_calibration_audit.py \
  --source publication/live/regime-results.json \
  --oos build/weekly-automation/generation-v5/artifacts/transition-oos-predictions.csv \
  --prospective build/weekly-automation/generation-v5/artifacts/transition-candidate-forecasts.csv \
  --output build/forecast-audit-improvements/calibration/calibration-audit.json
```

추적되는 재사용 명령은 `scripts/replay_transition_calibration_audit.py`다. `--selection-end`를 생략하면 입력 payload의 동결 계약을 사용한다. JSON을 먼저 직렬화·검증한 뒤 고유 임시 파일에 기록하고 fsync 후 원자적으로 교체한다. 원본 경로·원본 hardlink·live 아래 출력은 거부하며 실패 시 직전 결과를 보존한다. 입력 OOS가 실제 발행 당시 확률을 재현하는지도 확인한다. 입력 CSV와 live payload는 실행 전후 bytes 동일 여부를 검사한다. 입력 OOS SHA-256은 `77d6f044e2cd4e8262bb29f06d14978f92f598cc747e27f93dbbfff858c596ba`, live SHA-256은 `06c9193a3ee3016ecee3ae9f629130cf237f3af7aee3be4dde87090f2e5478ce`다.

## 통합 담당자에게 필요한 사항

1. 로컬 미리보기에는 JSON 전체를 `research.calibration_audit`에 연결한다. 실제 발행 이력 `weekly`는 보존한다. UI 담당자에게 OLD/NEW 필드와 null 의미를 전달했다.
2. 정상 `run_transition_benchmark`의 OOS 및 prospective 생성은 새 보정을 사용한다. `_calibrate_transition_probability(raw, history, minimum_rows=..., random_state=...)`의 반환 형식은 그대로다. origin을 생략하는 기존 운영 호출은 frozen 2023 cutoff에서 선택·적합한다.
3. **`scripts/audit_outputs.py`의 독립 감사도 통합했다.** 버전이 없는 기존 행과 명시적 v1은 원래 Platt 검증 경로를 유지한다. v2는 생산용 `TransitionCalibrator`를 호출하지 않고 후보 적합·과거 블록 점수·선택·확률과 모든 선택 메타데이터를 독립 재계산한다. OOS·챔피언 prospective·전체 후보 prospective 경로 모두 이 검사를 사용한다. 같은 generation의 버전 불일치, 버전 누락과 v2 metadata 혼용, 알 수 없는 버전은 거부한다. 선택 이력 내용 hash를 포함한 로컬 캐시로 변조된 원자료가 이전 계산을 재사용하지 못하게 한다. 기존 threshold·survival raw identity·payload/CSV 최종확률 대조는 유지했다.

## 검증

다음 9개 파일에서 미래 결과 변조·동일 target purge·블록 고정·3개 후보 선택·부족 표본/적합 실패·NaN/Inf·경계 주 제외·쌍방 split 위조·직접 평가 우회·기존 4-tuple 호환·coherence·선정 문서 통합을 검증했다.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider \
  tests/test_transition_calibration.py tests/test_analysis_transition_validation.py \
  tests/test_selection_evaluation.py tests/test_forecast_research_evaluation.py \
  tests/test_causal_calibration.py tests/test_directional_coherence.py \
  tests/test_selection_family_audit.py tests/test_transition_calibration_audit.py \
  tests/test_audit_outputs.py
```

결과: **212 passed, 1 skipped / 80.01초**. 기본적으로 비활성인 합성 V4 전체 bundle 검사가 skip 대상이며 별도 활성 실행으로 추가 확인한다. pandas 내부의 NumPy timedelta deprecation warning 1건 외 실패는 없었다. 지정 소스·테스트의 `git diff --check`도 통과했다.

실제 파생 OOS 9,738행과 prospective 108행을 대상으로 기존 버전 **9,846행**, 새 버전 **9,846행**을 각각 독립 감사기로 검증했다. 생산용 보정기 호출 없이 새 확률과 선택 메타데이터를 재계산했으며 모두 통과했다. 239개 캐시 블록 적합, 157.629초. 증거는 `build/forecast-audit-improvements/calibration/independent-audit-verification.json`이며 입력 bytes는 그대로였다. 전체 원자료 모델 재학습·주간 자동화·실제 발행을 수행한 결과는 아니다.

별도 `REGIME_RUN_V4_E2E=1` 전체 V4 검사는 **실패(224.18초)**했다. `generate_demo_payload(contract_version='v4', profile_name='quick')`가 현재 10개 모델을 생성하지만, 최상위 감사기는 동결된 V4 16개 모델을 요구해 `v4 quick candidate manifest must contain exactly 16 models`에서 중단했다. 보정 감사에 도달하기 전의 기존 roster 불일치이며, V4 동결 검사를 완화하지 않았다.

관련 원본 보존을 HEAD와 직접 대조했다. `src/regime_lab/demo.py`, `src/regime_lab/pipeline.py`, `src/regime_lab/analysis/models.py`, `config/operating-contract.json`은 파일 전체 bytes가 HEAD와 같고 `git diff`가 없다. `scripts/audit_outputs.py`는 보정 감사 부분을 수정했지만 **V4 roster 상수 및 모델 개수·집합 검사 블록은 HEAD와 동일**하다. 비교 SHA-256 증거는 `build/forecast-audit-improvements/calibration/roster-preservation-evidence.json`에 기록했다.

이 전체 검사에서 생성된 **실제 파이프라인 산출물**을 그대로 사용해 `audit_transition_outputs`를 직접 실행한 결과는 **통과**다. OOS **234행**, 챔피언 prospective **18행**, 6개 모델 전체 후보 prospective **108행**의 purge·후보 선택·보정 버전/수치/metadata·threshold·survival raw identity·payload 원확률/최종 coherence를 검증했다. 증거는 `build/forecast-audit-improvements/calibration/offline-transition-generation-audit.json`이다. 따라서 전환 감사 통합 통과와 최상위 V4 roster 실패를 구분한다.

전체 병렬 실행 로그의 보정 fixture 오류도 확인했다. 해당 실행은 `BenchmarkProfile` 필수 인자 6개가 빠졌던 수정 전 fixture를 수집한 상태였다. 현재 fixture는 필수 인자를 모두 지정하며, 파일 수정 없이 다시 실행한 최종 9개 파일 subset은 **212 passed, 1 skipped, 0 failed, 0 errors / 33.31초**다. 로그: `build/forecast-audit-improvements/calibration/final-subset-tests.log`. skip은 위에 별도 실패 원인을 기록한 opt-in 최상위 V4 검사이며 보정 CSV 테스트는 포함되어 통과했다.
