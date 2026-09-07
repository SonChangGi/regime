# 정상 주간 국면 연구 생성

표준·전체 V5 live 생성은 `compose_live_publication_research`에서 국면 경로, 확률 보정, 신규 정보 비교를 함께 생성한다. 공식 주간 국면·운영 예측·선택 결과를 유지하며, 연구 후보의 자동 승격은 하지 않는다. quick/demo/replay 경로는 기존처럼 수집·연구 생성을 건너뛴다.

## 라이브러리 인터페이스

```python
from regime_lab.research.forecast_publication import build_forecast_publication_research

result = build_forecast_publication_research(
    payload, canonical, states, baseline,
    transition_predictions, transition_candidates, cache_directory,
    existing_sources=existing_cboe_directory,
    information_sources=stored_optional_sources,  # 최초 이관할 때만 필요
    offline=False,
    progress=print,
)
# result["blocks"]: forecast_research / calibration_audit / forecast_information
# result["provenance"]: 동일 생성·입력·소스·계산법을 묶는 공개 가능한 영수증
```

`transition_predictions`는 완료된 모든 후보·기간의 OOS 예측이며, `transition_candidates`는 아직 만기가 오지 않은 모든 후보의 예측이다. `TransitionBenchmarkResult.latest_candidate_forecasts()`를 사용한다. 공개 파일이나 다른 생성의 결과를 임의로 합치지 않는다.

`existing_sources`는 기존 VIX·VIX9D·VVIX 파일을 읽기만 하는 manifest 디렉터리다. `information_sources`는 VIX3M·일정·포지션·EBP 등의 `.blob`과 시각별 `.json`을 보관한 디렉터리다. 현재 검증 자료를 이관할 때는 `build/forecast-audit-improvements/new-information/sources`를 사용할 수 있다. `offline=True`는 해당 저장본으로 계산하며 새로운 공개시점을 만들어내지 않는다.

## 검증과 실패 동작

- canonical·공식 상태·주간 표시의 시점과 상태를 대조한다. 기본 모델의 origin·다음 주 target·실제 상태·확률을 확인한다.
- 완료된 전환 사건은 같은 공식 상태에서 다시 확인한다. 미래 후보는 기간별 마지막 1·4·13개 origin과 모든 모델을 요구하며, 실제값이 섞이거나 누락되면 중단한다.
- 경로 모델에는 1·4·13주 전체 경로와 최신 예측이 필요하다. 모든 모델은 같은 OOS origin을 사용한다.
- 보정 비교는 기존 v1과 새 v2 생성 모두 처리한다. 저장된 확률과 1주 기준점을 먼저 재현하고, 별도의 v2 보정 연구를 수행한다. 기존 발행 수치는 바꾸지 않는다.
- 신규 정보가 없으면 `not_evaluable`과 블록별 사용 가능 상태를 남긴다. 평가 표본이나 과거 이용시점을 합성하지 않는다.
- `research.extensions.build.forecast_audit_present=true`는 세 블록·동일 provenance·출력 해시·고정 다운로드 링크가 모두 유효하다는 계약이다. 오래된 marker 없는 공개 자료는 계속 읽을 수 있다.

필수 연구 예외는 주간 생성 호출자에게 전파된다. 원자적 generation 교체와 현재 예측의 ledger 기록보다 먼저 실패하므로 이전 정상 발행본을 보존한다.

## 캐시와 최초 확인 시각

정상 캐시는 `research-cache/forecast-publication` 아래에 둔다.

| 경로 | 역할 |
|---|---|
| `information-source-versions` | 계속 누적하는 원본 스냅샷과 실제 최초 저장 시각 |
| `information-inputs/YYYY-MM-DD` | 해당 cutoff에서 한 번 확정한 소스·실패 상태·계산용 입력 |
| `results/<cache_key>` | 공식 입력·소스 스냅샷·코드·라이브러리 버전·고정 프로토콜로 식별한 완성 블록 |

같은 cutoff 재시도는 검증된 저장본을 사용한다. 다음 cutoff에서는 추가 정보만 갱신하며 이전 스냅샷을 유지한다. 코드가 바뀌면 같은 소스 시점으로 연구 결과를 다시 계산한다. 손상된 소스나 일부만 남은 완성 캐시는 자동 무시하거나 다른 주의 결과로 대체하지 않고 실패한다.

격리된 릴리스 캐시에서 처음 생성했다면, 다음 정상 주간 실행 전 `information-source-versions`를 정상 캐시의 동일 폴더로 이관한다. 이름 충돌 시 바이트가 같은지 확인하고 기존 시각별 JSON을 덮어쓰지 않는다. 이 과정은 원본 DB나 forecast ledger를 변경하지 않는다.

공개 블록에는 논리적 출처 ID·해시·시각과 파생 결과만 담는다. 기계 경로·원자료·수집 오류 전문은 로컬 캐시에 남긴다. 각 블록의 다운로드는 `./data/forecast_research.json`, `./data/calibration_audit.json`, `./data/forecast_information.json`으로 고정하며 packager가 payload의 해당 블록으로 직접 생성한다.

## 검증 근거

전용 검사는 정상 생성 연결, 세 블록 marker, 누락·오래된 시점·잘못된 확률·미래 후보·경로, 캐시 손상, 재시도, 다음 주 수집과 최초 확인 시각 보존, v1/v2 보정 비교, 동결 시점의 소스 연도 재현을 확인한다. 기존 정보 수집·보정 독립 감사 검사도 함께 실행한다.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q \
  tests/test_forecast_publication.py tests/test_publication_research.py \
  tests/test_forecast_new_information.py tests/test_transition_calibration_audit.py
```

실제 릴리스 실행 및 전체 회귀 결과는 `docs/forecast-audit-implementation.md`와 릴리스 영수증에서 확인한다.
