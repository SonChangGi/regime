# 국면 예측 연구의 전향 기록·평가

새 연구 모델의 **실제로 발행한 예측**을 별도 SQLite 원장에 보존한다.
공식 `ForecastLedger`, 실행가격·분배금, 현재 DB, 발행 파이프라인과 독립적이다.
현재 구현 작업에서는 실제 원장 생성·프로토콜 등록·발행을 수행하지 않았다.
테스트의 원장과 시각은 모두 임시 경로의 합성 사례다.

## 지금 확인할 결과

`build/forecast-audit-improvements/prospective-preview-final/preview.md`에 실제 연구 결과로
만든 미리보기가 있다. 네 모델의 원래 확률을 보존하고 `reconstructed_market`을
유지한다. 증명되지 않은 과거 입력 공개·수집 시각은 null이다.

- `protocol.preview.json`: 실제 현재 결과에 대응하는 검토용 고정 계약.
  첫 관측 기준이 이미 지났으므로 신규 전향 프로토콜로 등록할 수 없다.
- `frozen-preview.json`: 최신 모델·기준모형의 정확한 확률과 내용 해시.
- `protocol.future-template.json`: 아직 등록하지 않은 미래 일정·모델·평가 조건.
- `information-set.preview.json`: 원래 입력 해시와 알려진 범위의 가용성.
- `preview-manifest.json`: 원본 해시, 실제 생성 시각, 발행 부적격 사유.

다음 명령은 **미리보기만** 만든다. 원장 경로를 받지 않으며 DB를 열지 않는다.
아래는 최종본의 생성 예시다. 출력 디렉터리는 새 경로여야 하므로 이미 존재하는 최종본을
재생성할 때는 새 디렉터리를 지정하고 후속 명령의 `--protocol`도 같은 디렉터리로 맞춘다.
현재 실행·통합 기준은 최종 소스·준비 코드·런타임 해시를 검증한
`prospective-preview-final/protocol.future-template.json`이다. 이전 미리보기는 보존한다.

```sh
.venv/bin/python scripts/manage_forecast_research_ledger.py template \
  --block build/forecast-audit-improvements/model-research/forecast_research.json \
  --label-spec config/label-spec.json \
  --version forecast-audit-prospective-v1 \
  --output-dir build/forecast-audit-improvements/prospective-preview-next
```

템플릿은 기존 `weekly_decision_at`의 **금요일 뉴욕 16시**를 사용한다.
휴일인 금요일에도 모델의 주간 기준시각을 유지하며 실제 마지막 거래일 종가 시각으로
바꾸지 않는다. 2026-09-04 기준 13주 목표는 DST를 반영한
2026-12-04T21:00:00Z이다. 104개 발행 원점과 마지막 원점의 13주 성숙 꼬리를 만든다.
`--monitoring-origins`와 완전한 `--criteria` JSON으로 등록 전에 계획을 조정할 수 있다.
달력·모델·기준모형·라벨·조건을 변경하면 새 버전과 별도 증거 계열이 필요하다.

## 미래 입력에서 실행 가능한 준비 경로

`prepare`는 과거 `forecast_research.json`을 읽어 분류를 바꾸지 않는다.
실제 미래 입력을 저장한 시각·내용 해시를 검증하고 **고정 모델을 새로 실행**한다.
그 실행에서 생성한 최신 예측에만 `prospective_inputs`와 모델 버전·레시피 증명을 붙인다.
재구성한 과거 점수와 과거 예측 행은 전향 결과에 포함하지 않는다.

필요한 입력은 기존 주간 작업이 생성하는 신뢰할 수 있는 로컬 파생 캐시
`canonical.pkl`, `states.pkl`, `input-manifest.json`, 공식 OOS CSV,
공식 `state-label-history.csv`다. 입력의 최신 기준일이 고정 일정의 원점이어야 한다.
새 캐시가 아직 없으면 준비를 거부하며 오래된 입력의 날짜를 변경해 대체하지 않는다.

다음은 **향후 검토 후 실행할 명령 예시**다. 지금 실행하지 않았다.
`FUTURE_INPUT_DIRECTORY`, `FUTURE_SOURCE_OOS.csv`, `FUTURE_STATE_HISTORY.csv`는
그 주의 실제 주간 작업 결과를 가리켜야 한다.

```sh
# 1. 그 주의 실제 입력을 현재 시각에 별도 보존. 모델 발행은 하지 않는다.
.venv/bin/python scripts/manage_forecast_research_ledger.py capture-inputs \
  --protocol build/forecast-audit-improvements/prospective-preview-final/protocol.future-template.json \
  --input FUTURE_INPUT_DIRECTORY \
  --source-oos FUTURE_SOURCE_OOS.csv \
  --state-history FUTURE_STATE_HISTORY.csv \
  --output-dir build/forecast-forward/20260911/input-capture

# 2. 위 입력으로 고정 모델을 새로 계산하고 발행 전 검토용 JSON 생성.
.venv/bin/python scripts/manage_forecast_research_ledger.py prepare \
  --protocol build/forecast-audit-improvements/prospective-preview-final/protocol.future-template.json \
  --capture-dir build/forecast-forward/20260911/input-capture \
  --output-dir build/forecast-forward/20260911/prepared
```

`capture-inputs`는 파일을 별도 디렉터리에 복사해 fsync하고 실제 저장 시각을 기록한다.
이는 **해당 바이트를 이 시각에 로컬에서 관측했다는 증거**다. 역사적 원자료의 공개일이나
과거 시점의 가용성을 새로 주장하지 않는다. 원본 파일 해시·프레임 해시·최신 기준일·공식
국면 일치를 검사하고, 작업 도중 입력이 변하면 완성본을 설치하지 않는다.

`prepare`는 다음을 실행 전후에 검사한다.

- 라벨 명세 파일의 버전·SHA-256과 원래 고정한 연구 프로토콜.
- 모델 코드, 준비 모듈·스크립트, 관련 소스의 고정 해시와 NumPy·pandas·SciPy·scikit-learn 버전.
- 입력 캡처의 내용·바이트 수·실제 저장 시각, 매니페스트 해시와 공식 국면 일치.
- 원점 ≤ 입력 저장 ≤ 입력 캡처 ≤ 현재 시각 < 고정 발행 마감.
- 새 실행의 모델 집합, 동일 원점, 학습 목표 < 예측 원점, 1·4·13주 목표 일정.

실행 결과는 `producer-latest.json`, `information-set.json`, `frozen.json`이다.
DB·프로토콜·실제 발행은 변경하지 않는다. 기존 원료 캐시를 생성하는 주간 작업과의
자동 연결 및 스케줄 등록은 이 모듈에 포함하지 않았다. 운영자는 위 명령으로 직접 준비할
수 있고 부모 파이프라인은 동일 API를 호출할 수 있다. 새 데이터 후보는 별도 레시피·입력
계약으로 등록해야 하며, 이 준비 경로의 기본 모델은 감사한 네 모델로 한정된다.

## 명시적 등록·발행

다음 단계 역시 **향후 실행 문서**이며 현재 수행하지 않았다.
등록은 미래 템플릿의 **첫 원점 이전**에 해야 한다. 예측 발행은 각 원점 이후에만 된다.

```sh
.venv/bin/python scripts/manage_forecast_research_ledger.py init \
  --ledger /ABSOLUTE/RESEARCH-ONLY/forecast-research.sqlite
.venv/bin/python scripts/manage_forecast_research_ledger.py register \
  --ledger /ABSOLUTE/RESEARCH-ONLY/forecast-research.sqlite \
  --protocol build/forecast-audit-improvements/prospective-preview-final/protocol.future-template.json

.venv/bin/python scripts/manage_forecast_research_ledger.py issue \
  --ledger /ABSOLUTE/RESEARCH-ONLY/forecast-research.sqlite \
  --frozen build/forecast-forward/20260911/prepared/frozen.json \
  --publication /ABSOLUTE/RESEARCH-ONLY/issued/20260911.json
```

마지막 `issue`는 새 **로컬 연구 발행 파일**을 실제 생성한다. 외부 사이트 전달이나 배포를
주장하지 않는다. 실제 DB 저장 시각과 파일 공개 직후 시각을 각각 런타임 UTC로 기록한다.
`--now`, `--published-at`, 과거 재생 발행 옵션은 없다. 템플릿의 발행 마감은 원점 후 60시간으로,
기존 일요일 작업이 완료될 시간을 포함하면서 다음 월요일 미국 개장 전으로 제한한다.
발행 시점에 예측 생성 후 1시간이 지났으면 다시 준비해야 한다.

이미 발행한 동일 원점·내용의 재시도는 최초 영수증을 반환한다. 다른 내용은 충돌이다.
기존 파일을 덮어쓰지 않는다. 파일 생성 실패 시 원장에는 `unpublished` 예측만 남고
점수에는 포함하지 않는다. 마감 전 동일 내용으로 새 경로에 재시도할 수 있다.
원장 저장 후 파일 공개 직전에 만료되면 발행을 거부하고 미발행 상태로 남긴다.

SQLite 네 테이블의 UPDATE/DELETE는 트리거로 거부한다. 읽을 때도 내용 해시와
프로토콜·발행·평가 연결을 검증한다. 별도 경로의 연구 DB만 받으며 기존 운영 DB를
찾거나 초기화하지 않는다. 이는 로컬 불변 기록이며 외부 공증 타임스탬프는 아니다.

## 공식 국면만으로 평가

`evaluate`에는 실행가격·분배금·자산 수익률이 필요 없다. 다음 JSON을 제공한다.
`label.sha256`은 발행 프로토콜에 고정한 라벨 명세 해시다. `available_at`은 제공한
공식 상태 스냅샷이 실제로 이용 가능해진 시각이며 미래 값을 넣을 수 없다.

```json
{
  "schema_version": "regime-forecast-official-states/1",
  "label": {"version": "market-causal-3state-v1", "sha256": "64-character-frozen-hash"},
  "available_at": "2026-09-18T20:05:00+00:00",
  "observations": [
    {"date": "2026-09-11T20:00:00+00:00", "state": "risk_on"},
    {"date": "2026-09-18T20:00:00+00:00", "state": "transition"}
  ]
}
```

```sh
.venv/bin/python scripts/manage_forecast_research_ledger.py evaluate \
  --ledger /ABSOLUTE/RESEARCH-ONLY/forecast-research.sqlite --states OFFICIAL_STATES.json
.venv/bin/python scripts/manage_forecast_research_ledger.py summary \
  --ledger /ABSOLUTE/RESEARCH-ONLY/forecast-research.sqlite \
  --protocol-sha256 REGISTERED_PROTOCOL_SHA256
```

원점부터 목표까지 중간 주가 하나라도 없으면 해당 기간은 대기한다. 목표가 성숙하고
공식 경로가 완전할 때만 한 번 기록한다. 새 스냅샷에 이전 행이 그대로 있으면 재시도는
멱등적이다. 이미 평가한 공식 경로가 수정되면 충돌을 보고하고 해당 평가 배치를 롤백한다.
기존 증거를 새 라벨·개정 상태로 조용히 덮어쓰지 않는다.

| 평가 목표 | 정의 | 점수 |
|---|---|---|
| endpoint | h주째 공식 상태 | 3상태 Log loss·Brier |
| first_departure | 현재 상태를 처음 떠난 목적지 또는 no_departure | 4범주 Log loss·Brier |
| any_risk_off_entry | 미래 경로의 비위험회피→위험회피 진입 여부 | 이진 Log loss·Brier |
| any_risk_off_occupancy | 미래 h개 관측 중 위험회피가 한 번이라도 존재 | 이진 Log loss·Brier |

위험회피 원점에서 계속 위험회피이면 점유는 참, 새로운 진입은 거짓이다.
이탈했다가 복귀하면 진입도 참이다. 한 주 모델은 next_state에서 다른 목표를 정확히
유도할 수 있다. 저장 확률은 재정규화·대체하지 않는다. 점수 계산의 log(0)만
고정한 1e-15로 제한하며, 다범주 Brier는 제곱오차 합이다.

## 지속 평가와 승격 검토 조건

후보와 기준모형은 동일 원점·동일 라벨·동일 목표를 하나의 예측 묶음으로 발행한다.
모델 누락이나 서로 다른 원점을 교집합으로 축소하지 않는다. 초기 템플릿의 조건은 다음과 같다.

| 조건 | 제안값 |
|---|---:|
| 기간별 짝지은 성숙 원점 | 52 이상 |
| 실제 1주 악화 / 회복 | 각각 12 이상 |
| 서로 다른 위험회피 진입 에피소드 | 6 이상 |
| 기간별 이진 사건 / 비사건 | 각각 8 / 20 이상 |
| 필수 발행 누락 | 0 |
| 성숙한 발행의 미평가 | 0 |
| 후보−기준 Log loss 차이의 95% 구간 상한 | 0 이하 |
| 후보−기준 Brier 평균 차이 | 0 이하 |
| 1주 악화 방향 포착률 | 25% 이상 |
| 실제 유지 주 중 잘못된 전환 경보 | 10% 이하 |

에피소드는 1주 실제 진입의 서로 다른 목표일로 센다. 중첩 13주 창의 진입 사건 수를
독립적인 위기 수로 합산하지 않는다. 악화 포착은 예측 최빈 상태가 실제 악화 방향을
맞혔는지이며 목적지까지 정확히 일치한 정확도와 구분한다.

불확실성은 같은 원점의 손실 차이에 대한 고정 13주 원형 블록 부트스트랩이다.
중간 발행 원점이 빠졌으면 주를 이어 붙여 구간을 계산하지 않는다.
95% 구간은 지속 모니터링 중 설명용이며 순차 검정이나 모델군 다중비교 보정을 대체하지 않는다.
위 숫자는 검토용 정책이며 검정력 보장이 아니다. 등록 후 결과를 보고 완화하지 않는다.

모든 조건을 통과해도 `manual_review_ready`만 반환하고 자동 승격하지 않는다.
부족하면 `collecting_evidence`와 미충족 조건·사건 수·누락 원점이 보인다.
유한한 등록 일정이 끝나면 `issuance_calendar_exhausted`와 최종 성숙일을 표시한다.
처음 계획한 기간 내 사건이 부족해도 경과 주수만으로 통과시키지 않는다.

## Python 통합 계약과 검증

```python
from regime_lab.research.forecast_prospective import (
    ForecastResearchLedger, build_protocol_template, capture_research_inputs,
    prepare_research_forecast, freeze_research_forecast,
)

# 기존 latest 블록의 순수 검토. DB를 열지 않는다.
frozen = freeze_research_forecast(block, protocol=protocol, information_set=information_set)

# 미래 실제 입력 캡처로 새 모델 실행. 아직 발행하지 않는다.
prepared = prepare_research_forecast(protocol=protocol, capture_directory=capture_dir)

# 별도 명시적 발행 단계. path에는 기본값이 없다.
ledger = ForecastResearchLedger(explicit_research_path)
# ledger.issue(prepared["frozen"], publication_path=explicit_new_local_file)
```

`latest` 항목은 `model`, `origin_date`, `current_state`, `next_state`와 `paths`를 사용한다.
`paths`는 `first_departure` 또는 `first_destination`+`no_departure`, `endpoint` 또는
`endpoint_probabilities`를 지원한다. 충돌하는 별칭, NaN, 음수, 합계 오류, 비일관적인
1주·다주 확률은 거부한다. 원래 부동소수 확률값을 그대로 동결한다.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider tests/test_forecast_prospective.py
```

전용 검증은 합성 임시 DB의 1·4·13주 경로, 위험회피 재진입, 시각·해시·라벨 변조,
중간 관측 누락, 발행 실패·재시도, 불변 트리거, 같은 원점의 점수, 사건 부족·누락 원점,
수동 검토 준비 조건, CLI 미리보기·DST, 실제 입력 캡처와 새 계산 준비 경로를 다룬다.
준비 경로의 단위 테스트는 모델 계산을 합성 fixture로 대체하므로 새 모델 자체의
실증 성능 검증과 구분한다. 실제 모델 계산 결과는 별도 연구 실행의 산출물이다.
