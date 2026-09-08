# 국면 예측 로컬 미리보기

미리보기: <http://127.0.0.1:8879/?forecast_model=evolving_boundary_gjr_skewt&forecast_horizon=4&model=evolving_boundary_gjr_skewt&window=52#history>

배포 주소: <https://sonchanggi.github.io/regime/>

Pages는 검증된 `publication/live/forecast-enhancements.json`을 필수로 포함합니다. 모델 비교, 추가 정보, 후보 기록, 운영 진단은 같은 세대의 결과로 묶이며, 원자료와 로컬 원장은 배포하지 않습니다.

`모델 검증`에서 모델·예측 기간·평가 기간을 선택하면 확률, 성적, 예측 이력이 함께 바뀝니다. `뷰 링크 복사`로 현재 선택을 다시 열 수 있습니다.

| 화면 | 확인할 내용 |
|---|---|
| 국면 판단 | 실제 국면 점수, 전환 경계 거리, 전주 변화 |
| 예측·적용값과 연구 비교 | 공식 1주 예측, 4·13주 최초 이탈의 적용값·보정안, 기간 말 국면 후보 |
| 전환 경보 | 1주 만료·사건 1:1 대응·오경보 예산, 기존 정책과 비교 |
| 후보 주간 기록 | 실제 발행, 결과 대기, 만기 평가 |
| 국면별 시장 움직임 | 현재 국면을 맞춘 위험 신호와 이후 하락·변동성 |
| 안정성 | GJR·EWMA를 포함한 5개 국면 정의와 1·4·13주 비교 |
| 추가 정보 | Cleveland 물가, NY Fed 성장, 준비금 수요 탄력성 |

## 결과 파일

- `build/forecast-upgrade-20260907/modeling/forecast-enhancements.json`: 모델·보정·경보·경제적 검증
- `build/forecast-upgrade-20260907/information/additional-information.json`: 새 자료와 같은 표본 비교
- `build/forecast-upgrade-20260907/operational/verification.json`: 복사 원장 발행·평가 및 최신 예측 일치 검증
- `build/forecast-upgrade-20260907/preview-build.json`: 미리보기 입력·산출물 검증
- `build/forecast-upgrade-20260907/alert-policy-verification.json`: 경보 재평가와 기존 모델 수치 보존
- `build/forecast-upgrade-20260907/weekly-candidates/candidate-summary.json`: 후보 주간 발행·평가 집계
- `build/forecast-upgrade-20260907/finalization-20260908/verification.json`: 최종 테스트·브라우저·원본 보존 검증

모델 점수는 선정 구간, 과거 진단, 실제 발행 성적으로 구분합니다.

## 다시 열기

프로젝트 루트에서 실행합니다.

```sh
.venv/bin/python -m http.server 8879 --bind 127.0.0.1 --directory build/forecast-upgrade-20260907/preview
```

## 미리보기 다시 만들기

```sh
.venv/bin/python scripts/build_forecast_upgrade_preview.py \
  --features build/forecast-upgrade-20260907/operational/preparation-bundle/features.pkl \
  --operational build/forecast-upgrade-20260907/operational/operational-diagnostics.json \
  --weekly-candidates build/forecast-upgrade-20260907/weekly-candidates/candidate-summary.json
```

새 자료는 아래 명령으로 갱신합니다. `--refresh`를 생략하면 저장한 스냅샷을 재사용합니다.

```sh
.venv/bin/python scripts/run_forecast_macro_information.py \
  --features build/forecast-upgrade-20260907/operational/preparation-bundle/features.pkl \
  --refresh
```

복사 원장의 발행·평가 상태는 다음 명령으로 확인합니다.

```sh
.venv/bin/python scripts/manage_local_forecast.py summary \
  --workspace build/forecast-upgrade-20260907/operational/workspace
```

모델 전체 실험은 `scripts/run_forecast_enhancements.py --help`, 복사 원장 준비·발행·평가는 `scripts/manage_local_forecast.py --help`에서 입력 경로를 확인할 수 있습니다. GJR 후보 실험에는 `research-volatility` 선택 의존성을 사용합니다.

## 후보 주간 발행·평가

새 관측 주의 실험이 완료되면 같은 workspace에 아래 명령을 실행합니다. 이전 발행의 만기 성적을 먼저 기록하고 새 1·4·13주 예측을 고정합니다. 발행 마감은 다음 NYSE 거래 주의 첫 개장 시각이며, 재실행은 최초 예측과 발행 시각을 보존합니다.

```sh
.venv/bin/python scripts/run_weekly_candidate.py \
  --workspace build/forecast-upgrade-20260907/weekly-candidates \
  --payload publication/live/regime-results.json \
  --enhancements build/forecast-upgrade-20260907/modeling/forecast-enhancements.json \
  --states build/forecast-upgrade-20260907/operational/preparation-bundle/states.pkl \
  --input-manifest build/forecast-upgrade-20260907/operational/preparation-bundle/input-manifest.json \
  --artifact-manifest build/forecast-upgrade-20260907/modeling/run-manifest.json
```

성적은 모델별 계산 코드·설정 버전으로 집계합니다. 재실행해도 최초 예측과 발행 시각은 유지하며, 수정된 정답은 별도 기록합니다.

주간 자동화는 `build.forecast_enhancements_research: true`와 `build.forecast_candidates_weekly: true`로 연결했습니다. 검증을 마친 세대를 바탕으로 이전 예측을 평가하고 새 예측을 발행합니다. 캐시를 이용한 재시도에도 적용됩니다. 기존 실행 시각은 유지합니다.

일반 생성에서는 artifacts의 `forecast-enhancement-states.pkl`, `forecast-enhancement-input-manifest.json`과 검증된 `generation-manifest.json`을 사용합니다. 새 세대가 완성된 뒤 교체하며, 생성 실패 시 이전 결과를 유지합니다.
