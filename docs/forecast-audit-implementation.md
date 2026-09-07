# 국면 예측 감사 개선 구현

2026-09-07 감사의 개선안을 순서대로 구현한다. 기존 데이터·API를 재감사하지 않으며, 원자료·운영 원장·기존 발행본을 보존한다. 커밋·푸시·배포 전 별도 로컬 미리보기에서 결과를 검토한다.

| 단계 | 구현 범위 | 확인 기준 |
|---|---|---|
| A | 국면 전용 불변 평가 원장, 실제 저장·발행시각, 참조 해시, 확률 검증, 전문가 이력·split 방어 | 자산 실행정보 누락과 독립적으로 평가, 늦은 발행 제외, 오류주입 회귀검사 |
| B | 과거 구간만 사용하는 무보정/Platt/축소 보정 선택, 국면 중심 화면, 날짜 이동, 지속기간 CI·표본, 모델/KM 구별 | 다음 평가 블록 고정, 실제 화면 입력→출력 검증 |
| C | 버전 있는 경계 설정, 비대칭 충격 후보, 악화/회복 hazard, 다상태 경로, 경보 예산·사건·외부 결과 검증 | 동일 origin, 시간 순서 보존, 공식 모델 자동 교체 없음 |
| D | VIX3M, 알려진 발표 일정, CFTC 포지션, 계획된 EBP의 증분 실험 | 무료 출처, 관측·공개·수집시각 보존, 검증할 수 없는 과거 일정은 prospective-only |
| E | 고정 연구 프로토콜·예측의 실제 발행 기록과 이후 평가 | 과거 재생을 실제 발행으로 기록하지 않음, 미래 결과 성숙 후 독립 평가, 자동 승격 없음 |

## 로컬 미리보기

`http://127.0.0.1:8781/`에서 확인한다. 첫 화면은 국면, 다음은 전환·경보와 모델 검증이다. 선택한 주의 발행·대상 시각과 기간별 확률을 함께 보여주고 이전 주·다음 주·최신 주 이동을 공유한다. 4·13주 모델 예측과 과거 Kaplan–Meier 기준률, 현재 지속기간 추정의 구간·표본 수를 구분한다.

연구 비교 모델을 바꾸면 다음 주 확률과 해당 모델의 경로가 실제로 바뀐다. 경로를 산출하지 않는 모델은 빈 경로를 명시한다. 위험회피 **진입**, 기간 중 위험회피 **관측**, 기간 말 국면은 별도 사건이다. 모델 검증에는 악화·회복 사건 수, 실제 운영 표본 수, 보정 전후 및 추가 정보 비교가 있다. 각 연구 패널의 전체 결과 링크로 근거를 확인할 수 있다.

반복 설명은 제거하고 해석·한계·계산 기준을 페이지 맨 아래의 **‘해석·계산 기준’** 한 곳에 모았다. 이 영역은 기본으로 접혀 있다. 보정 비교는 4·13주 핵심 2행을 먼저 제시하고 전체 모델 36행은 별도로 펼친다.

기존 발행 예측은 당시 기록을 보존하며 새 보정 결과는 비교 패널에서 보여준다. 미리보기 검토 후 커밋·push·배포 승인을 받았다. 실제 배포에는 아래 정상 생성 경로와 배포 패키지 검증을 적용한다. 미리보기는 검증한 별도 generation을 원자적으로 연결하며 오류가 나면 이전 미리보기와 사용 중인 발행본을 유지한다.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/build_forecast_audit_preview.py \
  --forecast-research build/forecast-audit-improvements/model-research/forecast_research.json \
  --calibration build/forecast-audit-improvements/calibration/calibration-audit.json \
  --information build/forecast-audit-improvements/new-information/forecast-information.json \
  --operational build/forecast-audit-improvements/operational-evaluation-v2/operational-diagnostics.json
.venv/bin/python -m http.server 8781 --bind 127.0.0.1 \
  --directory build/forecast-audit-improvements/preview
```

## 구현과 실제 결과

확률 전용 평가는 `forecast_probability.py`의 별도 append-only 테이블에 저장한다. 실제 DB 저장·발행 시각을 확인하고 예측 참조 해시를 같은 트랜잭션에서 대조한다. NaN·음수·잘못된 합계와 공식 상태에 맞지 않는 전문가 이력을 거부한다. 선정·진단 split도 origin과 target에서 독립 계산한다. 순수 계약·점수 계산과 저장·CLI 조립을 구분했고 기존 투자 평가 테이블은 다시 쓰지 않는다.

운영 원장의 **복사본**에서 4건이 성숙했다. 같은 대상 주의 재발행 2건을 중복 표본으로 세지 않아 **독립 대상은 2주**다. 결과를 보기 전에 정한 최초 검증 발행 정책의 Log loss는 1.3539, Brier는 0.8310이다. 1건은 아직 성숙하지 않았다. 표본이 작으므로 모델 우위 근거로 사용하지 않는다. 원래 발행·투자 평가 해시는 보존됐다.

확률 보정은 무보정·Platt·25% 축소 Platt 후보를 완료된 과거 26주 블록에서만 비교하고 다음 블록에 고정한다. 동결 선정 cutoff 이후 진단 결과로 후보나 계수를 조정하지 않는다. 현재 발행 모델의 누적확률 정합화까지 적용한 재구성 결과는 다음과 같다. 이는 이미 본 구간의 진단 비교다.

| 기간 | 동일 성숙 origin | 이전 최종 Log loss | 새 최종 Log loss |
|---|---:|---:|---:|
| 4주 | 188 | 0.7390 | 0.7226 |
| 13주 | 179 | 0.4377 | 0.4306 |

13주 진단에서는 무보정 결과가 더 좋지만, 그 결과를 보고 정책을 바꾸지 않았다. 192주 전체의 4·13주 누적확률 단조성 위반은 0건이다. [보정 방식과 독립 재계산](calibration-audit-implementation.md)에 선택 근거와 버전 호환을 기록했다.

고정 모델 실험은 556개 공통 origin 중 191개 진단 origin을 비교했다.

| 모델 | Log loss | Brier | 악화 포착 | 회복 포착 |
|---|---:|---:|---:|---:|
| 기존 경계 과거충격 | 0.3656 | 0.2234 | 0/19 | 16/21 |
| 비대칭 변동성 후보 | 0.3664 | 0.2230 | 0/19 | 16/21 |
| 방향·지속기간 hazard | 0.5463 | 0.3326 | 0/19 | 0/21 |
| 공식 앙상블 | 0.4950 | 0.3006 | 0/19 | 3/21 |

비대칭 후보의 전반적 우위를 확인하지 못했다. 방향 hazard는 Markov 경로 기준선보다 13주 진입 점수가 나았지만 기간 말 국면 점수는 악화했다. 선정 당시 연 4회 오경보 예산을 적용해도 이후 실현 오경보는 경계 5.46회/년, hazard 8.74회/년으로 초과했다. 개선되지 않은 결과도 보존하고 운영 모델로 승격하지 않는다.

다상태 경로는 상태와 지속기간을 매주 갱신한다. 경제적 공변량은 origin 값에 고정하는 조건부 전망이므로 미래 가격 경로를 모의했다고 해석하지 않는다. 별도 에피소드·지속성·향후 변동성·하방사건 검증도 포함했다. [실험 프로토콜](forecast-audit-research-protocol.md)과 `build/forecast-audit-improvements/model-research/`에 전체 수치와 행 단위 근거가 있다.

VIX3M은 기존 변동성 변수와 동일한 모델·표본을 유지한 증분 실험에서 Log loss가 0.373190에서 0.373862로 소폭 악화했다. FOMC, CPI·고용, CFTC TFF, EBP는 무료 공식 자료를 실제 저장하고 현재 시점의 피처까지 계산했다. 최초 수집 이전의 이용 가능성을 복원하지 않았으므로 해당 블록의 과거 평가 표본은 0이다. BLS 직접 다운로드가 막힌 환경에서는 공식 웹 표현을 저장해 사용하며 오래된 일정은 미상으로 처리한다. [추가 정보의 출처·시점·재실행 방법](forecast-new-information-research.md)에 상세히 기록했다.

전향 기록은 별도 연구 원장으로 구현했다. `capture-inputs → prepare → issue → evaluate → summary` 명령으로 입력의 실제 저장, 고정 모델의 새 실행, 불변 발행, 1·4·13주 성숙 평가를 연결한다. 모델·라벨·소스·시각을 결속하고 사건 수·누락·동일 표본 점수로 검토 준비 여부를 판정한다. [사용 방법](forecast-prospective-ledger.md)과 `build/forecast-audit-improvements/prospective-preview-final/`에 현재 결과의 동결본 및 첫 기준일이 2026-09-11인 미래 템플릿을 준비했다.

실제 로컬 입력으로 `capture-inputs`와 `prepare`도 실행했다. 네 모델을 새로 적합한 확률은 앞선 연구 결과와 일치했고 최대 차이는 0이었다. 이 확인에서는 원장 생성·등록·발행을 하지 않았다. 증거는 `build/forecast-audit-improvements/prospective-prepare-check/verification.json`이다.

## 검증 기록

### 정상 생성·배포 연결

정규 live V5 `standard`·`full` 생성은 동일한 입력·국면 이력·OOS 예측으로 모델·경로, 확률 보정, 신규 정보의 세 연구 블록을 생성한다. 입력·레시피·출처·결과 해시와 기준 시점을 검증한 캐시만 재사용한다. 세 블록이 누락되거나 기준 시점이 다르면 기존 결과를 대체하기 전에 실패한다. 새 연구 결과는 챔피언 선정과 이미 발행한 예측을 바꾸지 않는다.

현재 발행본의 연구 추가는 `scripts/prepare_forecast_audit_release.py`로 별도 디렉터리에 준비한다. 정규 생성과 같은 함수를 실행하고, 공식 예측·선정·라벨 보존을 확인한 뒤 기존 review·generation manifest 절차로 결속한다.

배포 패키지는 화면의 세 연구 블록에서 `forecast_research.json`, `calibration_audit.json`, `forecast_information.json` 다운로드 파일을 직접 생성한다. 각 파일은 publication manifest에 포함되며, 파일 자체의 해시뿐 아니라 검토한 payload 내용과의 일치도 검사한다. 개인 경로·원자료를 담은 연구 캐시와 부가 HTML은 복사하지 않는다.

기존 보정 버전이 없는 운영 이력은 v1으로 재현하고, 명시적인 v2 이력에만 새 보정을 적용한다. 과거 이력을 읽었다는 이유만으로 당시의 운영 확률이 새 정책으로 바뀌지 않게 한다.

기존 재현 감사기도 현재 계약에 맞췄다. 정합화 v2의 standard 실행은 전체 origin을 사용하며, CSV는 저장된 float64를 정확히 복원해 수치상 동률이 파일 해석 때문에 뒤집히지 않게 한다. 원래 방향 예측은 독립적으로 원본 산출물과 대조하고, 화면에 표시하는 정합화 결과는 별도로 계약을 재생해 확인한다. 모델 선정 규칙과 기존 발행 수치는 보존한다.

2026-09-07 최종 발행 준비에서는 현재 192주 발행본의 V5 전수 감사를 통과했다. 지속기간 support/2의 완료 구간 수와 현재 연령 표본도 독립 재계산했다. 세 연구 블록과 확률 전용 운영 평가의 결과는 앞서 검토한 미리보기와 정확히 일치했다. 공개 파일은 18개이며 실제 배포 패키지의 다운로드 3개까지 검증했다. 전체 결과는 `build/forecast-audit-improvements/release-20260907/release-preparation.json`에 있다.

최종 로컬 패키지는 `http://127.0.0.1:8782/`에서 검토했다. 모델 변경·이전 주·최신 주 이동, 보정 188/179주, 운영 평가 2주, 설명의 기본 접힘과 JavaScript 오류 0건을 확인했다. 매매 예정시각 기준 발행 기록과 예측 대상 시각 기준 확률 평가를 구분해 표시한다. `release-browser-verification.json`과 `release-final-package-verification.json`에 확인 내용을 기록했다.

신규 정보의 스냅샷·최초 저장 시각 파일 12개는 정상 주간 캐시로 이관했고 모든 바이트 해시가 일치한다. 기존 DB와 발행 원장은 그대로 유지했다. 이관 근거는 `normal-source-cache-installation.json`, 기존 발행본의 로컬 백업은 `pre-release-publication/`이다.

배포 전 회귀는 Python 3.13.13·Node 24.7.0으로 실행했다. 전체 실행에서 1,918개 통과·1개 선택 검사 제외를 확인했고, 최신 연구 payload와 캐시 키에 맞지 않던 화면 검사 3개를 수정했다. 해당 두 파일을 다시 실행해 108개 모두 통과했다. 전체 실행 기록은 `release-final-tests.xml`, 수정 후 결과는 `release-final-regression-recheck.xml`이다. GitHub Actions에서는 최종 커밋 전체를 다시 검사한 뒤에만 배포한다.

### 로컬 미리보기 단계의 검증

- 운영 원장 원본 SHA-256: `87b88bbcfa0922eefd6925db3fdc04d75d0c534f4c26fef0d3c4a2db228dceae` — 복사·평가 후 동일.
- live payload SHA-256: `06c9193a3ee3016ecee3ae9f629130cf237f3af7aee3be4dde87090f2e5478ce` — 미리보기 조립 후 동일.
- 모델 기본 설정: HEAD 대비 1,079행 입력과 59개 교정 예측 일치. 새 모델 연구의 코드·입력·결과 해시 일치.
- 전향 원장·준비 경로 전용 테스트 86개 통과. 실제 입력 캡처와 네 모델 새 실행도 별도 검증.
- 보정: 기존 방식 9,846행과 새 방식 9,846행을 생산 함수를 재사용하지 않는 감사기로 각각 재계산해 통과. 실제 오프라인 파이프라인의 전환 OOS 234행, 챔피언 18행, 전체 후보 108행도 통과.
- 선택적으로 실행한 구형 V4 합성 전체 감사는 기존 10개 생성 모델과 동결 V4 16개 계약의 불일치로 실패했다. 관련 생성 코드와 roster 계약이 HEAD와 동일함을 확인했고 검사를 완화하지 않았다. 이는 위 전환 감사 통과와 별도 결과이며 상세 근거는 보정 문서에 있다.
- 최종 전체 회귀검사: **1,802 passed, 1 skipped, 0 failures, 0 errors**, 368.48초. 결과는 `build/forecast-audit-improvements/final-tests.log`와 `final-tests.xml`에 있다. 제외한 검사는 위 구형 V4 선택 검사다.
- 최종 UI 전용 검사 184개 통과. 실제 브라우저에서 첫 로딩, 모델 변경, 과거 주·최신 주 이동, 보정·추가 정보·운영 수치, 하단 기본 접힘을 확인했다. 390px 모바일 가로 넘침과 브라우저 JavaScript 오류는 없었다. `browser-verification.json`에 기록했다.
- 최종 원본 보존·Git HEAD·변경 파일 해시와 검증 연결은 `build/forecast-audit-improvements/verification.json`에 있다.

전체 검사 재실행:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider
```
