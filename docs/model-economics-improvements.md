# 국면 해석·방향성·지속기간 개선 계약

공식 다음 주 상태 예측과 기존 label 정의는 유지한다. 새 방향성 계약은
`canonical-one-week-joint-first-destination/2`이며, 이 표식이 없는 이전
reviewed V5는 기존 읽기 계약으로 검증한다. 새 생성물은 공식 1주 확률,
목적지별 누적 단조성, 총 이탈확률과의 합, 최종 OOS 평가의 재계산을 검증한다.

## 목적지 확률의 공동 정합성

현재 상태가 `i`이면 1주 안 첫 이탈 목적지가 `j`라는 사건은 다음 주 상태가
`j`라는 사건과 같다. 따라서 `j != i`에 대해 `F_j(1) = P(S[t+1]=j)`로
고정한다. 4·13주 총 이탈확률은 기존 총위험 정합성 계약의 결과를 유지한다.

가능한 목적지는 두 개다. 첫 목적지의 4·13주 확률을 `(x, y)`, 공식 1주
확률을 `a`, 총 이탈확률을 `(T1,T4,T13)`이라고 하면 가능한 영역은
`a <= x <= a+T4-T1`, `x <= y <= x+T13-T4`이다. 이 평행사변형 내부 또는
네 변으로의 직교투영 중 원래 예측과 L2 거리가 가장 짧은 점을 사용한다.
두 번째 목적지 확률은 각 horizon의 총위험에서 첫 목적지 확률을 뺀다.
이 절차에는 목표값이나 학습할 파라미터가 없다.

원래 방향행은 `weekly[].directional_risk_raw`에 보존한다. 최종 1주 모델명은
공식 next-week champion이며, 별도 방향 benchmark의 champion은 연구 평가
기록으로 유지한다. `coherence_evidence`에는 성숙한 사건의 raw/coherent
joint log loss, Brier, top-label ECE, 조건부 목적지 log loss와 재현용 OOS
행을 기록한다. 홀드아웃 구간의 변화는 사후 진단이며 새 모델 선정 근거가 아니다.

## 방향성 표본과 재사용

`standard`와 `full`은 각 horizon에서 가능한 전체 selection/diagnostic
origin을 사용한다. `quick`만 smoke 목적의 3개 상한을 유지한다.
모델, horizon, 학습 경계, 알려진 feature/state prefix, 코드와 실행 설정에
결속된 캐시는 새 주가 추가돼도 이전 origin의 예측을 재사용한다. 실패한
fallback은 캐시하지 않으며 손상된 캐시는 다시 계산한다.

오프라인 runner는 1·4·13주를 서로 독립적인 세 프로세스로 실행한다. 모든
worker는 같은 purging·모델·선정 조건을 사용한다. 최종 결과를 모아 기존
고정 CSV 직렬화 계약으로 sidecar와 해시를 생성한다.

## 지속기간의 표본 지원

기존 Kaplan–Meier, 현재 spell의 우측검열, episode bootstrap을 유지한다.
새 `regime-duration-support/2`는 현재 경과기간까지 도달한 완료 spell이
3개 미만이면 `insufficient_tail_support`로 표시하고 생존확률·RMST·구간을
비워 둔다. 전체 완료 spell의 기존 최소 5개 조건도 유지한다.

현재 age와 4·13주 시점의 위험집합 수, 최대 관측기간, 지원되는 잔여기간을
함께 제공한다. 현재 spell이 과거 완료 spell보다 길다는 이유만으로
“향후 생존확률 100%, 평균 잔여기간 52주”를 반환하지 않는다.

## 전체 label grid와 고정 대표 모델

`config/label-sensitivity-grid.json`의 243개 조합을 모두 평가하고, 기존
`v1_spy_hysteresis`를 별도 control로 함께 표시한다. 그리드의 실행 의미는
`symmetric_entry_exit_frozen_robust_score/1`로 기록한다.

- 추세는 지정 주수의 변동성 조정 로그가격 변화다. 스트레스는 지정 주수의
  변동성과 52주 낙폭을 각각 robust 표준화한 뒤 결합한다.
- 모든 위치·척도·분위 임계값은 초기 520주에서 동결한다.
- 진입은 `(q, 1-q)`, 이탈은 `(e, 1-e)`의 대칭 분위 임계값이다.
  `e > 0.5`의 경로 의존적인 겹침도 정의대로 유지한다.
- 최소 지속주수는 다음 전환 전에 관측되어야 하는 주수다.
- 평가는 selection 구간만 사용하며, 미래 수익률이 cutoff 이후까지 필요한
  행은 해당 horizon 통계에서 제외한다. 사후 수익률은 label 구성타당성
  설명 통계이며 실행 가능한 거래수익이 아니다.

모델 순위 점검의 대표는 결과를 보고 고르지 않는다. 중앙 `(13,8,.25,.5,2)`,
빠른 `(8,4,.3,.4,1)`, 느린 `(26,13,.2,.6,4)` 세 조합과 unchanged control을
고정한다. 동일한 소수 가격 피처와 현재 상태를 쓰는 ridge logistic(`C=.1`),
smoothed Markov, persistence를 공통 origin·1주 purge로 비교한다. 이 작은
고정 benchmark의 순위 안정성은 운영 전체 모델군을 재선정한 결과가 아니다.

## 호출과 산출물

공개 함수는 다음과 같다.

- `run_label_sensitivity(canonical, states=states)` → summary, 전체 grid 표,
  고정 대표 모델 OOS 표.
- `upgrade_directional_payload(payload, states)` → 기존 방향예측을 공동
  정합화한 unpublished copy와 raw/coherent 평가.
- `apply_full_directional_research(payload, states, benchmark)` → 전체 origin
  benchmark, 실제 실행 상한, 새 방향성 sidecar 해시를 결속한 unpublished copy.

수정된 payload에는 이전 publication approval이나 generation manifest를
승계하지 않는다. 원본 payload와 원자료는 수정하지 않는다.

```bash
PYTHONPATH=src /Users/changgison/projects/regime/.venv/bin/python \
  scripts/run_model_economics_audit.py labels
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/changgison/projects/regime/.venv/bin/python \
  scripts/run_model_economics_audit.py directional
```

입력은 `build/comprehensive/input/{canonical,states,features}.pkl`이고,
출력은 `build/comprehensive/model-economics/`다. 작업 중단 시 캐시를 보존해
같은 명령으로 재개할 수 있다. 데이터 수집이나 외부 배포는 하지 않는다.
