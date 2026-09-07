# 종합 개선 적용 및 검증

`codex/comprehensive-improvements` 작업본에서 확률 정합성, 투자 연구, 운영 준비, 화면을 함께 개선했다. 발행된 주간 예측과 기존 원장은 보존하고 새 연구는 자동 승격하지 않는다.

## 결과 확인

현재 확인 대상은 **2026-09-04 관측, 공식 예측 192주**를 보존한 `build/release-20260904/pages-package/`다. Pages에 전달하는 것과 같은 정적 파일 묶음이며 패키지 15개 파일의 해시 검증을 통과했다. 실제 패키지의 데스크톱·모바일 검증과 최종 전체 검사 1,417개를 통과했다. 별도 옵션으로 실행하는 장시간 V4 검사 1개는 건너뛰었다.

결정 → 모델 → 자산 → 성과 순으로 탐색한다. 자산 화면에서 **평균 기준**을 바꾸고 셀을 선택하면 표본과 구간을 확인할 수 있다. 성과 화면의 **4주 개선 전략 비교**는 새 배분 연구를 연다. 나머지 해석·검증·운영 상세는 페이지 맨 아래 한 곳에 모았다.

작업본 `/Users/changgison/projects/regime-improvements`의 `publication/live`에는 심사된 발행 파일 네 개를 반영했다. 동결한 발행 원본과 원장은 `build/release-20260904/source/`에 보존했다. **이 배포 전 검증 단계에서는 커밋·push·실제 배포를 수행하지 않았다.**

최신 방향성 연구는 과거 6,492행과 최신 18행을 평가했고 fallback은 모두 0건이다. 학습 호출을 금지한 캐시 재생은 0.647초에 완료됐으며 원 실행과 정확히 일치했다. 192주 공식 1주 예측은 보존됐고, 1주 확률 일치·기간별 누적확률 단조성·확률 범위 검증도 통과했다. 라벨 민감도는 243개 설정을 평가했다.

| 개선 | 구현 및 검증 |
|---|---|
| 방향 확률 | 1주 공식 예측에 고정하고 4·13주 목적지 누적확률을 공동 제약에 투영. raw/coherent 결과를 보존하고 독립 재계산 |
| 지속기간 | 현재 경과기간·horizon별 과거 지원이 부족하면 생존확률과 RMST 외삽을 차단 |
| 배분 | 4주 OOS 확률 decoder, alpha 만기와 위험예산 리밸런싱 분리, 10/20bp 비용, 초기배분·정적보유·위험비교 기준선 |
| 경보 | 선정구간의 연 4/8/12회 오경보 예산, 위험악화/모든이탈 분리, 변동성·낙폭 기준선, 경제손실 행렬 |
| 불확실성 | 243개 라벨 설정과 고정 대표모델 비교, ablation block CI·사건조건 성과 |
| 추가 연구 | Cboe 기간구조·VVIX와 4/13주 직접 하방분포 비교. ADS 빈티지·SLOOS·CMDI 공개 자료와 수집 해시 보존 |
| 실제 운영 | 원장을 읽기전용으로 진단. 정시율·지연·연속 완료·확률점수·동일주 기준선. 정상 주간 발행 전에 새 연구 네 블록을 함께 조립 |
| 화면과 용량 | 최근 26주 core, 104주 단위 해시 결속 이력, 필요 시 지연 로딩. 평균 방식·모의 잔고·실행 목표 구분 |

## 로컬 확인과 재현

작업본 루트에서 정적 패키지를 열려면 다음과 같이 실행한다. 패키징 이전 미리보기 대신 최종 파일 묶음을 그대로 제공한다.

```bash
python -m http.server 8767 --bind 127.0.0.1 \
  --directory build/release-20260904/pages-package
```

이 명령을 실행한 뒤 <http://127.0.0.1:8767>에서 확인한다. 이번 검증에 사용한 Python은 `build/release-20260904/validation-venv/bin/python`이다. 아래 연구 명령의 `python`도 이 실행 파일로 바꾸어 재현한다.

이번 연구의 입력·산출물은 `build/release-20260904/`, 추가 원자료는 그 아래 `additional-sources/`에 있다. 브라우저에는 `pages-package/`의 검증된 파생 JSON만 전달한다. 과거 Cboe·CMDI·SLOOS 다운로드를 당시 실제 수집 자료로 소급 표시하지 않으며, ADS는 과거 빈티지와 공개 시점을 반영한다.

배분·경보·원장 진단의 독립 재현 예시다. 동결 입력과 원장은 읽기전용이며, 새 결과를 별도 디렉터리에 기록한다.

```bash
python scripts/run_decision_research.py \
  --canonical-cache build/release-20260904/input/canonical.pkl \
  --source-artifacts build/release-20260904/source/artifacts \
  --source-payload build/release-20260904/source/publication/regime-results.json \
  --ledger build/release-20260904/source/forecast-ledger.sqlite3 \
  --output-directory build/release-20260904/reproduced-decision
```

## 다음 주 발행의 연구 유지

정상 `standard/full` live V5 빌드는 발행 직전에 같은 generation의 가격·모델 예측·전환 예측·자산 성과·ablation 결과로 배분 v2, 조기 경보, 운영 진단, 하방·추가 자료 연구를 조립한다. 공식 예측·선정·실행 신호를 보존하고, 연구가 완료되지 않으면 새 결과를 발행하지 않는다. 입력·출력 해시와 계산 시간은 `research.extensions.build`에 남는다.

추가 공개 자료는 주차가 바뀌면 갱신하고, 같은 주 재시도는 해시가 고정된 스냅샷을 재사용한다. 운영 진단은 이번 발행의 원장 추가 전까지 기록된 실제 발행을 집계한다. quick/demo 경로에는 새 수집을 연결하지 않았다.

9월 4일 실제 입력 재생은 네트워크 호출을 차단한 상태에서 **104.594초**에 완료됐다. 배분·경보·하방·ablation 출력은 독립 연구 결과와 정확히 일치했고, 운영 진단은 계산 시각을 제외하고 일치했다. 공식 192주 예측·선정과 발행 원장은 변하지 않았다.

## 빠른 주간 예측 준비

`python scripts/prepare_operational_forecast.py --help`의 명시적 입력으로 최신 주 하나만 계산한다. 동결 선정 계약, 피처·상태 해시, 과거 전문가 예측의 최신성을 확인한 후 Markov/XGBoost/이탈모형을 각각 한 번만 학습한다. 발행 마감이 지나거나 입력 주가 오래됐으면 학습 전에 차단한다. `--research-replay` 결과는 실제 발행에 사용할 수 없다.

준비 명령은 자동화 등록, 주문, 기존 발행 원장 추가를 하지 않는다. 기존 주간 자동화에 연결하기 전에도 로컬에서 동일 입력의 확률 일치와 처리시간을 재현할 수 있다. 실제 운영 성과는 앞으로 정시에 발행하고 평가가 완료된 주만 누적한다.

같은 입력·계약의 재시도는 준비 파일을 덮어쓰지 않고, 다른 예측으로 충돌하면 거절한다. 이 별도 준비 명령과 정상 주간 빌드의 연구 조립은 서로 다른 경로다.

## 해석

이 변경은 새 후보가 반드시 수익을 높인다는 뜻이 아니다. 비용과 기간을 바로잡은 4주 alpha는 실제 반복 매매가 가능해졌지만 이번 구간에서는 기본 위험관리 정책보다 수익이 낮았다. Cboe 증분도 기간별 차이가 있어 후보 비교로 남긴다. 국면은 위험 상태의 정의이며 Risk-off의 과거 평균수익이 양수라는 이유로 매수 신호로 뒤집지 않는다.

9월 4일 기준 방향 확률을 정합화한 뒤 진단 Log loss는 1주 `0.527247 → 0.494972`, 4주 `0.926930 → 0.910322`, 13주 `0.733228 → 0.734179`이었다. 사건의 의미는 바로잡았지만 13주 점수는 소폭 악화했다. 전체 기간 재평가에서도 방향성 연구 champion은 경험적 first-passage 기준선으로 유지됐다.

위험 비교 기준선은 후보의 실제 실행 비중과 이전 26주 공분산으로 60/40의 투자 비중을 줄인다. 레버리지를 쓰지 않고 같은 리밸런싱 밴드를 적용하므로, 후보와 실현 변동성이 정확히 같다는 뜻은 아니다. 초기 비중 매수·보유와 정적 60/40 매수·보유를 함께 비교해 최초 주식 비중의 효과와 반복 의사결정의 효과를 구분한다.

[원 감사](audits/2026-09-06-comprehensive-audit.md) · [현재 발행 스냅샷](current-snapshot.md) · [최종 검증 보고서](release-validation-2026-09-07.md)

## 최종 확인

최종 전체 pytest는 CI와 동일한 Python 3.13.13·Node 24.7.0에서 **1,417개 통과·실패 0개·별도 V4 검사 1개 건너뜀**으로 완료됐다. 로그는 `build/release-20260904/final-ci-runtime-tests.log`에 있다. Pages와 동일한 정적 패키지로 11개 모델, 과거 날짜 링크 복원, 평균·기간·경보 필터, 성과 토글을 실제 브라우저에서 확인했다. 1280px·390px 화면에 가로 넘침이 없고 콘솔 오류·경고도 없었다. 세부 근거는 위 최종 검증 보고서에 기록했다.

## 모델 선택 연결 수정

모델 비교의 요약·상세 지표가 운영 모델 값에 고정되던 오류를 수정했다. 선택 모델의 동일한 2023년 이후 평가 행에서 직접 읽으며, 비교 차트와 표도 선택을 표시한다. 하위 순위 모델도 선택하면 차트에 포함한다. 전환 지표는 위험 악화와 완화를 포함하는 최빈국면 변화 포착임을 명확히 했다.

앞선 UI 수정 검증에서는 11개 모델을 실제 브라우저에서 전환해 요약·상세·예측·차트·표의 연결을 확인했다. 예측 국면별 자산 성과와 과거 날짜의 새로고침 복원, 지연 도착 시 사용자 선택 보존, 누락 값, 접근성, 패키징을 포함한 관련 검사 193개가 통과했다. 선택 제어 수정은 모델 평가 값이나 원 발행 데이터를 바꾸지 않았다. 이 단계의 기록은 `build/comprehensive/model-selection-verification.json`에 있으며, 최종 9월 4일 패키지 검증과 구분한다.

## 검증된 발행 후보 준비

완성된 연구를 Pages에서 쓰는 경로와 계약으로 준비한다. 아래 명령은 현재 `publication/live`를 덮어쓰지 않으며 커밋·push·배포를 수행하지 않는다. 출력 디렉터리는 새 경로여야 한다. 재시도 시에는 다른 출력 경로를 사용해 이전 정상 후보를 보존한다.

```bash
python scripts/prepare_comprehensive_release.py \
  --source build/release-20260904/source/publication/regime-results.json \
  --source-manifest build/release-20260904/source/publication/generation-manifest.json \
  --source-artifacts build/release-20260904/source/artifacts \
  --ledger build/release-20260904/source/forecast-ledger.sqlite3 \
  --preview build/release-20260904/preview/data/regime-results.json \
  --research-root build/release-20260904 \
  --output-root build/release-20260904/release-candidate-recheck
```

스크립트는 원 발행 세대·선정 근거·입력 해시를 검증하고, 공식 예측과 모델 선정, 기존 원장 의사결정 및 주차 변경을 거부한다. 새 연구의 입력·출력·directional CSV를 대조한 뒤 기존 frozen V4 비교, selection-family 재검증, 발행 심사, Pages 패키징과 공개 경계 검사를 그대로 실행한다. `artifacts/baselines/v4-20260821`의 검증된 frozen 기준선도 필요하다.

성공하면 지정한 출력 디렉터리의 `publication/live`에 논리 경로가 `publication/live`인 네 개의 검증된 발행 파일이 생기고, `public-dashboard`에 실제 정적 패키지가 생성된다. `release-preparation.json`은 검증 결과와 파일 해시를 기록한다. 이번에 완료한 준비 결과는 `build/release-20260904/release-candidate/`에 보존돼 있다. 재현용 입력과 원장은 private artifacts의 해시 근거로만 사용하며 정적 패키지에 포함하지 않는다. 검증 실패 시 이번 시도의 staging만 제거한다.
