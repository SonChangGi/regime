# Regime 모델·데이터·Ablation 감사 보고서

- 감사 기준일: 2026-08-27
- 범위: 공식 모델 선택 증거, 운영·shadow·frozen 모델군, 5-track mechanism ablation, 신규 release-aware 데이터 계약
- 증거 구분: **구현됨**, **synthetic/계약 검증됨**, **repo-local operational OOS**, **기존 실데이터 reconstructed OOS**, **미실행·미수집**을 구분한다.
- 승격 원칙: 새 모델·데이터·정답지는 matched-origin 실데이터와 독립 audit 전에는 자동 승격하지 않는다.

## 결론

현재 repo의 공식 운영 champion은 `causal_dynamic_ensemble`이고, V4 Markov는 immutable regression baseline이다. 현행 reviewed generation의 2016–2022 matched selection 구간에서 dynamic과 multiscale ensemble은 모두 Markov 대비 기존 gate를 통과했다. multiscale의 log loss가 dynamic보다 `0.00019817` 낮았지만 `0.01` 단순성 허용폭 안이므로 complexity rank가 낮은 dynamic이 선택됐다.

이 선택을 prospective 성능 보장으로 해석하면 안 된다. 2023–2026 post-selection holdout에서는 `recency_weighted_xgboost_208w`가 dynamic보다 log loss `0.00673901` 낮았고, 그 차이는 material-regret threshold `0.05` 안이었다. selection lock 때문에 이를 사후 승격하지 않은 것은 적절하다. 현재 repo-local payload와 selection-family sidecar는 `evidence_track=operational_oos`로 결속됐지만, 이 표본의 사후 holdout 우위가 미래 성능 또는 원격 배포를 뜻하지 않는다.

운영 모델군 단순화, direct-jump TVTP, filtered HSMM, expanding-factor TVTP, BOCPD, 5-track mechanism ablation, MCS 보조평가, release-aware source catalog는 코드·계약 단계까지 구현됐다. 현행 [`selection-family-audit/v2`](../../publication/live/selection-family-audit.json)는 11-model×365-origin selection 행에서 MCS, sharpness, 상태별 recall, 전환 탐지지연, false alarms/year를 `operational_oos`로 독립 재계산했고, holdout 사용량 0과 기존 log loss·Brier exact-match를 확인했다. 새 11-model weekly generation도 실제로 완료됐다. 반면 최종 [`shadow-regime-audit-final-20260827-r2.json`](../../build/research-audits/shadow-regime-audit-final-20260827-r2.json)과 [`standard mechanism run`](../../build/mechanism-ablation-standard-final-20260827-r2/runs/20260826T163113.864460Z-223b1830/mechanism-ablation-report.json)은 latest-revised 역사 panel의 `historical_reconstructed_oos`이며 `automatic_promotion_eligible=false`다. 두 증거층을 섞지 않으며 공식 champion과 frozen V4 Markov 역할은 바뀌지 않는다. 신규 15개 source ingest와 장기 prospective operational OOS 성능 평가는 아직 없다.

## 1. 공식 dynamic champion과 frozen V4 Markov

현재 source of truth는 [`config/operating-contract.json`](../../config/operating-contract.json)이다. repo-local reviewed source인 [`publication/live/regime-results.json`](../../publication/live/regime-results.json)도 generation `20260826T184946.198911Z`의 dynamic champion과 `operational_oos`를 가리킨다. [`build/pages-workflow-package-final-20260827`](../../build/pages-workflow-package-final-20260827)은 이 payload와 manifest/2·두 sidecar를 byte-identical하게 패키징했고 local browser QA를 통과했다. 다만 개발용 [`web/data/regime-results.json`](../../web/data/regime-results.json)은 frozen V4 fallback이고 원격 배포·readback은 없으므로, 모델 identity·local predeployment·remote deployment를 분리해 해석해야 한다.

| 역할 | 모델 | 현재 의미 |
|---|---|---|
| 공식 운영 champion | `causal_dynamic_ensemble` | 현재 repo-local V5 publication의 선택·표시 모델 |
| runner-up | `causal_multiscale_ensemble` | gate 통과, 단순성 tie-break로 미선택 |
| frozen regression baseline | `markov` | V4 결과 재현과 비교 기준, 공식 champion 아님 |
| selection evidence track | `operational_oos` | 현행 reviewed generation·sidecar 계약; 장기 prospective 성능 보장은 아님 |

`publication/live`의 selection period는 2016-01-08–2022-12-30이고 모델별 365개 origin이다. 기존 paired moving-block bootstrap은 13주 block, 1,999 resample, seed 17을 사용했다.

| selection 모델 | Log loss | Brier | Markov 대비 Log loss 개선 | Holm 조정 p | gate | complexity rank |
|---|---:|---:|---:|---:|---|---:|
| Markov | 0.35751714 | 0.17892475 | 기준 | — | 기준 통과 | 2 |
| Dynamic ensemble | 0.32661584 | 0.16955457 | 0.03090130 | 0.024 | 통과·선택 | 15 |
| Multiscale ensemble | 0.32641767 | 0.16926038 | 0.03109946 | 0.024 | 통과·runner-up | 16 |

multiscale가 primary metric에서 근소하게 앞섰지만 두 모델의 차이는 `simplicity_tolerance=0.01` 안이다. tie-break 순서는 `complexity_rank → calibration_error → log_loss → model`이고 dynamic이 더 단순하며 selection calibration error도 `0.03029851`로 multiscale의 `0.03311770`보다 낮다. payload는 `selection_reason=simplicity_tiebreak_within_tolerance`, policy hash, complexity registry hash와 runner-up을 함께 보존한다.

## 2. 불리한 post-selection 결과도 보존한다

2023-01-13–2026-08-21 holdout은 모델별 189개 origin이며 selection 후 진단이다.

| 모델 | Holdout Log loss | Holdout Brier | 해석 |
|---|---:|---:|---|
| Recency-weighted XGBoost 208w | 0.47812216 | 0.28147054 | holdout 최저 log loss, selection gate에서는 Holm 미통과 |
| XGBoost | 0.48314902 | 0.28927750 | holdout 2위, selection gate에서는 Holm 미통과 |
| Dynamic ensemble | 0.48486117 | 0.29472318 | 사전 selection champion 유지 |
| Multiscale ensemble | 0.48522920 | 0.29506149 | dynamic과 유사 |
| Markov | 0.59748237 | 0.34722433 | frozen 비교 기준 |

holdout에서 recency-weighted XGBoost가 1위였다는 결과를 삭제하지 않는다. 그러나 holdout을 다시 selection에 사용하면 이중 선택이 된다. selection evidence에서 XGBoost와 recency-weighted XGBoost의 Holm 조정 p-value는 모두 `0.81`로 gate를 통과하지 못했다. 따라서 현재 증거로 가능한 결론은 “dynamic 선택이 동결 규칙을 통과했고 post-selection regret `0.00673901`이 설정된 0.05 안이었다”까지다.

[`publication/live/v5-vs-v4-comparison.json`](../../publication/live/v5-vs-v4-comparison.json)은 V5 Markov와 frozen V4 Markov의 공통 552개 key를 확인했다. selection 공통 365개, holdout 공통 187개에서 확률 token bytes와 numeric value가 정확히 같고 최대 절대차는 `0.0`이다. 이 parity는 baseline 재현성을 확인하지만 dynamic의 미래 우월성을 증명하지 않는다.

## 3. 운영 모델군 단순화 계약

새 operating contract는 연구 역할과 weekly 계산 역할을 분리한다.

### 필수 기준선과 weekly base

- 기준선: majority, persistence, Markov
- 추가 weekly base: XGBoost, PCA-ridge logistic, 208주 recency-weighted XGBoost/ridge, 208주 discounted Markov

### 핵심 후보

- causal dynamic ensemble
- causal multiscale ensemble
- XGBoost
- PCA-ridge logistic
- 208주 recency-weighted XGBoost/ridge
- 208주 discounted Markov

`xgb_hazard_destination`은 ensemble component로 별도 기록한다. direct jump를 허용하는 새 `direct_jump_tvtp_hurdle`은 shadow다.

### frozen reproduction only

다음 모델은 코드와 historical manifest를 삭제하지 않되 새 weekly core에서는 제외하는 계약이다.

- ridge logistic
- random forest
- extra trees
- histogram gradient boosting
- spline logistic
- calibrated linear SVM
- elastic-net logistic
- shrinkage LDA
- adjacent-only duration TVTP hurdle
- next-state transition logistic

과거 reviewed generation의 17-model roster는 [`config/operating-contract.json`](../../config/operating-contract.json)의 `historical_reviewed_rosters`에 manifest hash와 모델 순서로 보존된다. 현재 generation은 11-model operating roster로 실제 weekly full run을 완료했다. 한 번의 완료만으로 장기 계산시간 절감·성능 유지가 실증됐다고 말할 수 없고, frozen 17-model 결과 재현 경로도 삭제하지 않는다.

## 4. direct-jump와 온라인 shadow 모델

| 모델 | 구현 | synthetic 검증 | 실데이터 결과 | 운영 역할 |
|---|---|---|---|---|
| `direct_jump_tvtp_hurdle` | [`validation.py`](../../src/regime_lab/analysis/validation.py), [`transitions.py`](../../src/regime_lab/analysis/transitions.py)에서 `adjacent_only=False` | Risk-on↔Risk-off 직접전환 학습·확률 테스트 | bounded matched 10 origins: log loss 0.617273, Brier 0.424586; 같은 10-origin Markov는 1.387362, 0.930514 | 결과 선택에서 제외된 shadow |
| `filtered_hsmm` | [`shadow_regimes.py`](../../src/regime_lab/analysis/shadow_regimes.py)의 causal forward explicit-duration filter | prefix 안정성, direct-jump, no smoothing/target 검사 | 1,064주 causal filter 실행; latest `transition`, filtered membership 0.3928/0.5883/0.0189, MAP direct jump 1건 | shadow, canonical target 아님 |
| `bayesian_online_changepoint` | 같은 파일의 normal-mean run-length filter | 미래행 변경 불변성, shift detection, no target 검사 | 1,064주 실행; latest change probability 0.019206, MAP run length 18 | 전환 경보 shadow |
| `dynamic_factor_tvtp` | [`dynamic_factor_tvtp.py`](../../src/regime_lab/analysis/dynamic_factor_tvtp.py)의 expanding-prefix imputer·scaler·PCA + direct-jump TVTP | 행 순서상 미래 feature 변경 prefix 불변성, `last_train_target < origin`, direct destination, gap=1 | bounded matched 10 origins: log loss 1.190349, Brier 0.845260, transition recall 0 | structural-prefix shadow; origin별 input vintage가 없어 operational OOS 부적격 |

HSMM은 체류기간을 명시적으로 다루지만 smoothed ground truth로 쓰지 않는다([Chiappa](https://doi.org/10.1561/2200000054)). BOCPD는 온라인 변화점 경보이며 canonical 3상태 label을 대체하지 않는다([Adams–MacKay](https://arxiv.org/abs/0710.3742)).

최종 shadow audit는 2,801,234개 관측, 1,077주, 510개 feature를 읽었고 `evidence_track=reconstructed_oos`, `historical_market_vintage_certified=false`, `automatic_promotion_eligible=false`, `public_release_eligible=false`를 기록한다. 위 수치는 코드 등록 상태가 아니라 실제 역사 panel 실행 결과지만, latest-revised feature panel을 사용했으므로 실제 operational OOS라고 부를 수 없다. direct-jump TVTP가 같은 10-origin Markov보다 낮은 loss를 기록한 결과와 dynamic-factor TVTP의 transition recall 0을 모두 보존한다. 표본이 10개뿐이므로 어느 쪽도 모델 선택 증거가 아니다. `dynamic_factor_tvtp`는 정식 state-space dynamic-factor 추정기라고 과장하지 않고, 각 origin의 purged row prefix에서 imputer·scaler·PCA를 다시 적합하는 **expanding-factor TVTP**로 해석한다.

## 5. 자기정의적 예측력을 분리하는 5-track ablation

[`config/mechanism-ablation-v2.json`](../../config/mechanism-ablation-v2.json)과 [`src/regime_lab/analysis/mechanism_ablation.py`](../../src/regime_lab/analysis/mechanism_ablation.py)는 다음 다섯 track을 고정한다.

| track | 입력 | 비교 family |
|---|---|---|
| `state_only` | current state와 causal transition history만 | persistence·Markov |
| `label_mechanics` | label score, train-fitted boundary distance, causal duration만 | fixed XGBoost |
| `market_ex_label` | direct label 성분을 제외한 breadth·cross-asset | fixed XGBoost |
| `macro_rates_credit` | 거시·금리·유동성·신용만 | fixed XGBoost |
| `full` | 선언된 전체 feature role | fixed XGBoost |

feature role은 column 이름으로 추정하지 않고 caller가 exact-once manifest로 제공한다. 누락·중복·미선언 column은 실패한다. 모든 비교는 origin, target, actual, split, current state, train size, gap이 정확히 같아야 하고, family 간 순위를 내지 않는다. 이 구조는 “state-only Markov와 full XGBoost의 차이”를 곧바로 특정 feature group의 기여로 오해하는 것을 막는다.

[`tests/test_mechanism_ablation.py`](../../tests/test_mechanism_ablation.py)는 manifest와 matched-origin fail-closed 동작을 synthetic data로 검증했다. 최종 standard run의 [`mechanism-ablation-report.json`](../../build/mechanism-ablation-standard-final-20260827-r2/runs/20260826T163113.864460Z-223b1830/mechanism-ablation-report.json)은 실제 510-feature reconstructed panel에서 555개 공통 origin, selection 365개·holdout 190개를 사용했다. 다섯 track과 두 state-history baseline에서 총 3,330개 probability prediction을 생성했고, holdout은 selection에 사용하지 않았다. 결과는 derived-only private artifact이며 `operational_oos=false`, `selection_effect=none`, `automatic_promotion_eligible=false`다.

fixed-XGBoost feature-track의 190-origin holdout 결과는 다음과 같다. delta는 같은 estimator의 `full` 대비 paired 값이며 음수면 reduced track이 낮은 loss를 냈다는 뜻이다.

| feature track | Log loss | Δ Log loss vs full | Brier | Δ Brier vs full |
|---|---:|---:|---:|---:|
| `label_mechanics` | 0.459620 | -0.019298 | 0.278176 | -0.009319 |
| `full` | 0.478918 | 0.000000 | 0.287496 | 0.000000 |
| `market_ex_label` | 0.584055 | +0.105137 | 0.362601 | +0.075105 |
| `macro_rates_credit` | 1.079261 | +0.600343 | 0.655422 | +0.367926 |

불리한 전환 결과도 [`transition-diagnostics.csv`](../../build/mechanism-ablation-standard-final-20260827-r2/runs/20260826T163113.864460Z-223b1830/transition-diagnostics.csv)에 남긴다. full은 전환 event 39건 중 32건을 탐지하고 7건을 놓쳤으며 평균 지연 0.813주, false alarm 연 0.824건이었다. label mechanics도 32/39·평균 0.813주였고 false alarm은 연 0.549건이었다. market-ex-label은 31/39, 최대 지연 3주, false alarm 연 4.394건이었다. macro-only는 25/39, 14건 미탐, 평균 지연 1.8주·최대 16주, false alarm 79건/3.641년 즉 연 21.695건이었고 Transition 상태 recall도 19/78=`0.243590`이었다.

이 표는 현재 target의 예측가능성이 label score·boundary·duration 같은 자기정의적 mechanics에 크게 의존할 가능성을 보여주며, broad/cross-asset와 macro-only가 이 fixed-XGBoost 계약에서 full을 대체한다는 증거는 아니다. 반대로 label-mechanics-only의 holdout 우위도 post-selection retrospective 진단이므로 새 모델 선택에 쓰지 않는다. state-only Markov와 persistence는 `comparison_family=state_history_baselines`, `model_mechanics_comparable_to_full=false`, `cross_family_ranked=false`로 별도 보고된다. 그 log loss를 fixed-XGBoost feature-track과 직접 순위화하지 않으며, 이 ablation은 공식 dynamic champion을 바꾸지 않는다.

## 6. selection-family audit와 MCS

[`src/regime_lab/selection_family_audit.py`](../../src/regime_lab/selection_family_audit.py)는 `selection-family-audit/v2`를 구현한다. 전체 candidate, gate, complexity rank, champion, runner-up, fallback, policy hash와 공통 origin hash를 한 sidecar에 묶는다. `evaluation_split=selection`만 허용하고, 모든 target이 동결된 `selection_end`보다 앞서야 하며, 후보별 origin·target·actual·current state·train size·gap이 하나라도 다르면 실패한다. supplemental document에는 `holdout_rows_used=0`, `selection_effect=none`, `role=supplemental_not_selection_gate`를 고정한다.

새 V5 generation 경로는 [`cli.py`](../../src/regime_lab/cli.py)에서 source CSV를 쓴 뒤 `selection-family-audit.json`을 만들고 artifact inventory와 `generation-manifest.json`에 결속한다. integrity audit는 sidecar를 `selection-diagnostics.csv`와 `oos-predictions.csv`에서 다시 만들어 semantic equality를 확인한다. promotion·packaging도 sidecar가 같은 generation·candidate manifest에 묶이지 않으면 실패한다. 이 계약은 현재 [`publication/live/selection-family-audit.json`](../../publication/live/selection-family-audit.json)과 manifest/2에 실제 반영됐고 최종 local package에도 포함됐다. 원격 공개 배포는 아직 하지 않았다.

[`src/regime_lab/analysis/model_confidence_set.py`](../../src/regime_lab/analysis/model_confidence_set.py)는 Hansen–Lunde–Nason range-statistic MCS를 구현했다. 기본값은 alpha `0.10`, circular moving block 13주, 1,999 bootstrap, seed 17이며 완전한 wide matched-loss matrix만 받는다. 공통 bootstrap draw, centered/studentized differential, 순차 제거와 finite-resample `+1` p-value를 보존한다. [`tests/test_model_confidence_set.py`](../../tests/test_model_confidence_set.py)는 deterministic·degenerate·invalid input을 synthetic으로 검증했다.

이전 17-model generation을 수정하지 않고 만든 [`historical replay`](../../build/selection-family-replay-20260826/selection-family-audit.json)는 `evidence_status=historical_reconstructed_oos`, 17개 후보·365개 공통 selection origin으로 계속 보존된다. 현재 현행 sidecar는 이 predecessor와 별개로 `evidence_status=operational_oos`, 11개 후보·365개 공통 selection origin, `holdout_rows_used=0`, `selection_effect=none`을 기록한다.

현재 11-model MCS retained set은 `xgb_hazard_destination`, `causal_multiscale_ensemble`, `causal_dynamic_ensemble`, `xgboost`, `recency_weighted_xgboost_208w`다. 제거된 6개는 `majority`, `persistence`, `recency_weighted_ridge_logistic_208w`, `pca_ridge_logistic`, `discounted_markov_208w`, `markov`이고 종료 사유는 `equal_predictive_ability_not_rejected`다. 이전 17-model historical replay의 retained 4개와 혼합하지 않는다.

이 MCS는 기존 materiality, Holm, Brier degradation, fallback gate를 바꾸지 않는 보조 진단이다([Model Confidence Set](https://doi.org/10.3982/ECTA5771)). retained set은 새 champion 목록이나 자동 승격 후보가 아니며, 기존 `causal_dynamic_ensemble` 선택은 그대로다.

## 7. 지표 구현 범위

확률예측의 primary metric은 multiclass log loss, secondary는 multiclass Brier다. 둘 다 proper scoring rule로서 확률의 정확성과 과신을 함께 벌점화한다([Gneiting–Raftery](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)).

[`src/regime_lab/analysis/selection_evaluation.py`](../../src/regime_lab/analysis/selection_evaluation.py)는 selection prediction에서 log loss와 Brier를 독립 재계산해 기존 diagnostics와 `1e-12` 이내에서 일치하지 않으면 실패한다. 그 동일 확률행으로 normalized-entropy sharpness, 3상태 recall 표, destination detection delay, false alarms/year와 MCS를 만든다. `historical_reconstructed_oos`, `operational_oos`, `synthetic_fixture` 외의 상태는 허용하지 않으며 top-level selection-family 상태와 nested 상태가 다르면 실패한다.

현재 `operational_oos` selection-family sidecar에서 dynamic champion의 독립 재계산 결과는 다음과 같다.

| 지표 | 값 | 계약상 해석 |
|---|---:|---|
| Log loss | 0.32661584 | 기존 primary와 exact-match |
| Brier | 0.16955457 | 기존 secondary와 exact-match |
| normalized-entropy sharpness | 0.63455140 | 0은 균등확률, 1은 퇴화확률 |
| 평균 최대상태 확률 | 0.87756785 | calibration과 별개인 집중도 |
| Risk-on recall | 142/155 = 0.91612903 | 다음 주 actual state 기준 |
| Transition recall | 88/107 = 0.82242991 | 다음 주 actual state 기준 |
| Risk-off recall | 97/103 = 0.94174757 | 다음 주 actual state 기준 |

전환 진단은 불리한 결과도 그대로 보존한다. 실제 전환 event 36건 중 destination을 다음 전환 전까지 탐지한 건은 34건, 미탐지는 2건이었다. 그러나 **동시점 destination 탐지는 0건**이었고, 탐지된 34건의 평균·중앙·최대 지연은 모두 1 forecast week였다. false alarm은 약 6.995년 노출에서 2건, 연환산 `0.28590411`건이었다. 이는 dynamic이 해당 matched selection sample에서 전환 시점에는 대체로 기존 상태를 유지 예측하고 한 주 뒤 destination을 인식했다는 뜻이며, 미래 전환 탐지 보장이 아니다.

기존 expected calibration error·accuracy·balanced accuracy·macro F1·transition precision/recall, fallback, paired block bootstrap·Holm은 원래 selection evidence에 남는다. 새 지표는 이를 교체하지 않고 같은 selection-only 행에서 독립 검산한 보조 산출물이다. current reviewed publication과 최종 local package에는 sidecar가 포함됐지만, 단일 predeployment generation은 장기 prospective 성능이나 원격 사이트 상태를 증명하지 않는다.

## 8. 신규 데이터 catalog는 수집 결과가 아니다

[`config/release-source-catalog.json`](../../config/release-source-catalog.json)은 15개 source를 등록한다.

- Philadelphia Fed ADS·RTDSM
- Fed H.4.1·H.8·SLOOS
- OFR FSI
- Fed near-term forward spread·EBP
- Cboe VIX 16:00 control·16:15 sensitivity·term structure
- DOL weekly claims, BLS, BEA, Census EITS

현재 15개 모두 `enabled=false`, `ingested=false`다. 12개는 `planned`, Cboe 3개는 `rights_review_required`다. [`src/regime_lab/data/release_archive.py`](../../src/regime_lab/data/release_archive.py)는 collector가 아니라 release catalog와 `ReleaseRecord`의 PIT 기반이다.

구현된 기반은 다음을 다룬다.

- official schedule, exact timestamp, source-file vintage, date-only, first-seen-only의 구분
- source release, provider first-seen, system retrieval, revision, raw hash
- operational OOS와 reconstructed OOS의 별도 eligibility
- New York DST, 월말, same-day release, Friday 16:00/16:15, late response, future period 방지

[`tests/test_release_archive_sources.py`](../../tests/test_release_archive_sources.py)는 이 시간 규칙을 synthetic record로 검증한다. 실제 archive download, parser, series reconciliation, coverage, provider rights 승인은 하지 않았다.

데이터의 연구 목적은 다음과 같다.

- ADS·RTDSM: mixed-frequency activity와 vintage 민감도([ADS](https://www.federalreserve.gov/econres/ifdp/real-time-measurement-of-business-conditions.htm), [RTDSM](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists))
- OFR FSI: 금융 스트레스 context([OFR FSI](https://www.financialresearch.gov/financial-stress-index/))
- near-term forward spread: 단순 10Y–2Y와 다른 정책기대 정보([Fed NTFS](https://www.federalreserve.gov/econres/feds/the-near-term-forward-yield-spread-as-a-leading-indicator-a-less-distorted-mirror.htm))
- EBP: revision-prone 월별 신용 shadow([Fed EBP](https://www.federalreserve.gov/econres/notes/feds-notes/updating-the-recession-risk-and-the-excess-bond-premium-20161006.html))
- VIX: 16:00 control과 16:15 calculation-window sensitivity 분리([Cboe VIX FAQ](https://www.cboe.com/tradable_products/vix/faqs))

이 데이터는 predictor/context이며 canonical equity label에 섞지 않는다.

## 9. 실제 bake-off·승격 게이트

아래는 앞으로의 완료 조건이며 현재 통과 사실이 아니다.

1. 신규 source별 archive parser, raw hash, rights profile, release timestamp, first-seen ledger와 coverage report를 인증한다.
2. 신규 feature는 legacy/full control과 같은 origin·target·actual·label spec·execution spec으로 평가한다.
3. 완료된 555-origin 표준 5-track의 paired 결과를 독립 재계산하고, 대체 label·origin vintage·시장시대별로 자기정의성 결론이 유지되는지 확인한다.
4. operating 11-model roster와 frozen 17-model roster를 같은 historical origin에서 비교해 성능·fallback·학습시간·복잡도 절감을 함께 본다.
5. 이미 생성된 reconstructed shadow output을 더 긴 matched sample·origin vintage에서 재검증하되 canonical label이나 champion 후보군에 자동 합류시키지 않는다.
6. 신규 roster·label·data run에서도 log loss를 1차, Brier·calibration·sharpness·state recall·transition delay·false alarms/year를 2차로 같은 계약 아래 재계산한다.
7. block bootstrap/Holm과 MCS를 같은 complete matched-loss matrix에서 실행하고, 현재 13주 control 외 block-length 민감도를 보고한다.
8. 생성된 v1·PIT-composite·broad-equity reconstructed label에서 표준 모델·데이터 ablation 결론이 유지되는지 확인하고, exact-split PIT가 확보되면 다시 실행한다.
9. 모든 결과를 `operational_oos`와 `reconstructed_oos`로 분리하고 불리한 결과도 보존한다.
10. 독립 audit와 local full CI를 통과해도 자동 승격하지 않고 별도 review decision을 남긴다.

## 10. 현재 판정

- 공식 모델: **`causal_dynamic_ensemble`**.
- frozen V4 Markov: **정확 재현된 regression baseline, 공식 champion 아님**.
- migration 상태: **repo-local reviewed source와 최종 local package는 dynamic·operational_oos·manifest/2·selection-family-audit/v2로 결속; 개발용 web data는 V4 fallback; remote deploy·readback 미실행**.
- dynamic 선택 증거: **현행 sidecar의 365 matched selection origin에서 기존 gate 통과; MCS는 보조 진단이며 champion 변경 없음**.
- recency-weighted XGBoost holdout 우위: **불리한 post-selection 결과로 보존, regret 0.00673901; 재선택 근거로 사용하지 않음**.
- 새 weekly core roster: **계약·validator 및 11-model 실제 weekly full generation 완료; 장기 prospective 증거는 아직 부족**.
- direct-jump TVTP·HSMM·expanding-factor TVTP·BOCPD: **코드·synthetic 검증 및 reconstructed 실데이터 shadow 실행 완료; bounded 10-origin 결과와 dynamic-factor transition recall 0 보존. operational OOS·승격 증거 아님**.
- 5-track ablation: **510-feature×555 matched origins의 standard run 완료(선택 365·holdout 190); label mechanics 의존 신호와 macro-only의 큰 손실·전환 false alarm을 보존. historical reconstructed·fixed-XGBoost feature-track 진단이며 자동 승격·cross-family ranking·champion 변경 없음**.
- generic selection audit·MCS·보조지표: **현행 manifest/2 generation과 final local package에 11-model×365-origin `operational_oos`로 반영; holdout 사용 0·selection effect 없음. 17-model historical reconstructed replay는 별도 보존**.
- 신규 15개 source: **catalog·PIT foundation만 구현, 전부 미수집**.
- 수익·성과 해석: 이 보고서는 모델 선택과 데이터 품질 감사이며 매매전략, 미래수익, 손실회피 또는 원금보장을 뜻하지 않는다.
