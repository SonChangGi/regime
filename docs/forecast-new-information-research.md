# 국면 예측 신규 정보 연구

이 모듈은 추가 정보의 **증분 효과**를 검증하는 별도 연구 경로다. 운영 수집기, 원시 DB, 원장, 챔피언, `publication/live`에 쓰지 않는다. 모든 신규 다운로드와 결과는 `build/forecast-audit-improvements/new-information` 아래에만 저장한다.

## 실행과 통합

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/run_forecast_information_research.py \
  --input /Users/changgison/projects/regime-improvements/build/release-20260904/input \
  --existing-sources /Users/changgison/projects/regime-improvements/build/release-20260904/additional-sources
```

위 경로는 확인된 로컬 **읽기 전용** 예시다. 기존 파생 입력은 manifest의 프레임 해시, 최종 기준일, 현재 발행 이력의 origin/target 국면과 대조한다. 예제 입력의 기준일은 `2026-09-04T20:00:00+00:00`이며 현재 국면 이력과 일치한다. 원본 발행 payload의 해시와 파생 manifest의 원본 payload 해시가 같다고 가정하지 않는다. 연구는 현재 payload에 이미 저장된 경계 모델 예측을 기준선으로 사용한다.

- `--offline`: 네트워크 없이 저장된 신규 스냅샷 재사용. 기존 3개 Cboe 자료도 위 경로에서 읽는다.
- `--refresh`: 신규 출처의 버전을 추가한다. 이전 스냅샷과 첫 수집 시각을 지우지 않는다.
- `--as-of 2026-09-07T12:00:00Z`: 실무 피처 조회 기준. 실제 자료 수집 시각을 바꾸지 않으므로 그때 아직 수집되지 않은 값은 이용할 수 없다.
- `--bls-web-extract /absolute/path/official-extract.txt`: 직접 HTTP 경로가 막힌 환경에서 공식 CPI·고용 일정의 웹 도구 렌더링 원문을 가져온다. 실제 원문과 URL을 저장하고 **가져온 현재 시각**부터 사용한다. 과거 timestamp를 지정할 수 없다.
- `--ebp-records /absolute/path/releases.json`: 선택적으로 기존 `ReleaseRecord` 필드를 가진 JSON 배열을 읽는다. 기본 실행은 공식 무료 EBP CSV를 별도 저장소에 수집한다. 기존 DB에는 접근하지 않는다.

기본적으로 기존 Cboe 스냅샷을 재사용한다. `--existing-sources`를 생략한 경우에만 비교에 필요한 VIX/VIX9D/VVIX 무료 공식 CSV를 별도 저장소에 함께 내려받는다. 기존 수집기를 실행하거나 기존 데이터/API의 품질을 다시 감사하는 작업은 없다.

통합 결과:

| 파일 | 용도 |
|---|---|
| `summary.json` | 출처별 성공·실패, 시점 계약, 입력/코드/설정 해시, 동일 표본 점수, 현재 이용 가능한 피처 |
| `forecast-information.json` | `research.forecast_information`에 넣을 작은 UI 어댑터 (`regime-forecast-information/1`) |
| `preview.html` | 로컬 읽기용 비교 화면. 외부 배포 없이 열 수 있음 |
| `oos-predictions.csv` | 기준 경계·동일 조건 control·VIX3M 후보의 origin/target/확률/마지막 학습 target |
| `market-features.csv`, `market-lineage.json` | 시장 피처와 출처별 SHA, 관측일·이용시각·수집시각 |
| `fomc-calendar-versions.json`, `runtime-calendar.json` | 일정 버전과 현재 알려진 미래 발표 피처 |
| `tff-observations.csv`, `runtime-positioning.json` | 계약별 관측 포지션과 현재 이용 가능한 쏠림 피처 |
| `runtime-ebp.json` | 기존 ReleaseRecord 계약을 이용한 현재 월간 EBP 값·관측 월·수집시각 |
| `sources/` | 내용 해시로 보존한 원문 blob과 시간별 manifest |

Python에서는 `build_forecast_information(summary)`로 UI 계약을 만들 수 있다. 각 행은 후보-동일 조건 control 또는 후보-기존 경계 기준선이다. 두 경우 모두 **그 행의 동일 origin 집합으로 다시 계산한 기준선 점수**를 사용한다. 원래 더 넓은 기간의 요약 점수와 비교하지 않는다. 이 함수는 payload에 쓰지 않으며 부모 preview 조립 경로에서 선택적으로 넣는다.

## 시점 계약과 출처

| 정보 | 피처와 경제적 질문 | 과거 이용 가능성 |
|---|---|---|
| VIX3M | VIX3M/VIX, 4주 변화. 기존 VIX·VIX9D/VIX·VVIX 대비 변동성 기간구조의 추가 정보 | CSV가 원래 공개시각을 제공하지 않으므로 `source_released_at=null`. 이전 관측일 종가만 쓰는 `reconstructed_market_prior_day` 진단을 별도로 허용한다. 실제 당시 파일 버전이나 발행 성능으로 주장하지 않는다. 실무 조회는 첫 수집 이후만 허용한다. |
| FOMC, CPI, 고용 | 다음 7일의 **알려진 예정 발표** 수, 다음 알려진 발표까지 일수. 방향을 미리 아는 입력이 아님 | 첫 저장 시각보다 과거에는 사용하지 않는다. 현재 달력의 DTSTAMP·과거 회의 날짜·페이지 수정일을 최초 공지일로 간주하지 않는다. |
| CFTC TFF | E-mini S&P 500 `13874A`, 10년 국채 `043602`의 자산운용자·레버리지 펀드 순포지션/OI, 정확한 4주 변화, 이전 52주 백분위 | 통상 화요일 관측→금요일 15:30 ET 공개지만 휴일·지연을 단순히 +3일로 처리하지 않는다. 역사적 실제 공개시각이 없으므로 첫 저장 이후에만 이용한다. |
| EBP | 이미 계획된 월간 신용 위험선호 입력 | 기존 ReleaseRecord의 공개·공급자 첫 관측·시스템 수집시각 중 최댓값 이후에만 이용. 현재 수정본을 과거 월말부터 사용하지 않는다. |

공식 무료 출처와 확인한 사실:

- [Cboe VIX3M CSV](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv): 날짜와 OHLC를 제공한다. 원래의 일별 공개 timestamp를 제공하는 파일은 아니다.
- [연준 회의 일정](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm): 다음 회의에서 확인되기 전까지 예정 회의일은 잠정적일 수 있다. 월을 가로지르는 범위의 종료일을 파싱하고 날짜 단위임을 보존한다.
- [BLS 공식 일정](https://www.bls.gov/schedule/): 일정은 수시 갱신되며 [iCalendar](https://www.bls.gov/schedule/news_release/bls.ics)를 제공한다. 스크립트는 ICS 실패 시 [CPI 공식 HTML](https://www.bls.gov/schedule/news_release/cpi.htm)과 [고용 공식 HTML](https://www.bls.gov/schedule/news_release/empsit.htm)을 시도한다. 이번 직접 HTTP 경로는 연간 일정·발표별 페이지·인쇄 페이지까지 403이었다. 공식 페이지의 웹 도구 렌더링은 접근 가능해 현재 표를 원문 그대로 저장하고 대체 표현임을 명시했다. CPI·고용 각 13개 날짜/시각을 파싱했으며 다음 7일 CPI 1건·고용 0건을 실제 조회했다. 향후 직접 HTTP가 계속 막히면 새 웹 도구 추출을 공급해야 하며, 7일 후에는 오래된 스냅샷을 미상으로 처리한다. 합성 자료 테스트도 시간대·folded line·일정 변경·취소·HTML fallback을 검증한다.
- [CFTC 공개 규칙](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm): 일반적 공개 시간, 휴일 예외, 전체 역사적 공개일 목록 부재를 설명한다. [연도별 futures-only 자료](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm), [필드 정의](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/cotvariablestfm.html)를 사용한다. ZIP은 메모리에서 제한된 단일 CSV/TXT만 읽고 경로를 추출하지 않는다. futures-only와 combined를 섞지 않는다.
- [연준 EBP 설명](https://www.federalreserve.gov/econres/notes/feds-notes/updating-the-recession-risk-and-the-excess-bond-premium-20161006.html)과 [공식 CSV](https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv): 매월 전체 역사가 수정될 수 있는 연구 자료다. 이번에 실제 CSV를 받아 643개월을 기존 ReleaseRecord 인터페이스로 연결했다. 원문 월초 날짜는 그 월의 식별자이므로 경제적 관측 종료일은 월말로 기록한다. 정확한 실제 공개시각은 미상이며, first-seen 정책으로 기존 인터페이스의 공개시각 필드를 채웠다는 메타데이터를 보존한다.

일정은 provider별 **최신 전체 버전**을 선택하므로 삭제·이동한 일정을 이전 버전과 합산하지 않는다. 이후 수정이 과거 조회를 바꾸지 않는다. 7일 넘게 확인되지 않은 일정이나 미래 범위가 부족한 일정은 미상으로 둔다. 알려진 예정 발표 0건이 돌발 발표까지 없다는 보장은 아니다. Cboe 7일, TFF 21일, EBP 100일을 넘는 관측은 명시적으로 이용 불가이며 0으로 대체하지 않는다. TFF 백분위는 현재 보고서를 제외하고 이전 52주 중 최소 26건에서 계산한다. 빠진 보고서는 네 행 이동으로 4주 변화인 것처럼 채우지 않는다.

## 동결 실험

설정은 `config/forecast-new-information.json`에 있다. 현재 경계 모델의 저장된 확률·현재 국면을 공통 입력으로 사용하며 control은 기존 변동성 3개, 후보는 VIX3M 비율과 4주 변화 2개를 추가한다. 동일한 로지스틱 회귀와 C=0.1, 최소 과거 104개 표본, 스케일러를 사용한다. 경계 확률 75%와 보조 확률 25%를 섞는다. class weight, 후보별 표본, 진단 구간 최적화는 사용하지 않는다.

학습·스케일링은 `target_date < origin_date`인 완료 자료만 사용한다. 결측이 있으면 양쪽 학습·평가에서 같은 행을 제외한다. 2023년 cutoff를 origin·target에서 다시 계산해 잘못된 split 문자열을 거부한다. 다음 시점에 완료된 진단 자료가 고정 업데이트 규칙에 따라 학습에 들어갈 수 있으나, 이는 새로운 미관측 홀드아웃이라는 뜻이 아니다. 이후 예측 결과와 피처를 바꿔도 이전 예측이 같음을 테스트한다.

평가는 Log loss, 3상태 Brier, 정확도, 최빈 도착 국면 기준 전환/악화/회복 포착과 오경보, 악화/회복 확률의 binary Log loss·Brier·AP를 포함한다. 방향 점수는 그 방향으로 갈 수 없는 상태를 제외한 at-risk origin에서 계산하고 표본 수를 함께 준다. 최근 52개 origin 주의 점수도 제공한다. 경보 임계값을 이번 진단 성능에 맞춰 선정하지 않는다.

일정·CFTC·EBP는 실제 저장된 과거 피처가 축적되면 같은 실행 스크립트에서 동일 조건의 보조 실험을 수행한다. 현재 0개인 과거 피처를 복원하거나 합성하지 않으며, 충분한 완료 학습표본 이전에는 `not_evaluable`을 기록한다. 다음 주 국면 추가 정보 실험과 다주 위험 경로 모델의 통합은 서로 다른 계약이다.

## 2026-09-07 실행 결과

현재 발행본의 공통 진단 **191주**에서 다음 결과를 얻었다.

| 모델 | Log loss | Brier | 악화 포착 | 회복 포착 |
|---|---:|---:|---:|---:|
| 기존 경계 과거충격 | 0.365558 | 0.223439 | 0/19 | 16/21 |
| 동일 조건 control | 0.373190 | 0.228416 | 0/19 | 15/21 |
| VIX3M 추가 후보 | 0.373862 | 0.228832 | 0/19 | 15/21 |

후보-control Log loss 차이는 **+0.000673**, Brier 차이는 **+0.000416**이다. 이번 고정 보조 모델 실험에서는 VIX3M의 증분 개선이 없고 기존 경계 모델도 능가하지 못했다. VIX3M이 모든 모델·기간에서 무용하다는 결론은 아니다. 운영 승격하지 않는다.

선정 구간은 학습 준비 이후 260주, 진단은 191주로 양팔의 표본이 같다. 최초 105개 origin은 엄격한 target 시차와 최소 학습표본 때문에 제외됐다. 학습 실패는 0건이다. FOMC·BLS·TFF·EBP의 역사적 이용 가능 origin은 0개이며 현재 시점 조회는 정상 계산했다. EBP 최신 관측 월은 2026년 7월, 값은 -0.319112793%p로 표시한다. 최신 다운로드를 9월 경제 관측값처럼 표시하지 않는다. 모든 정보는 연구용으로 유지한다.

검증 명령: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider tests/test_forecast_new_information.py`.
