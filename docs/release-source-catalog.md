# Release-aware source catalog

`config/release-source-catalog.json`은 새 거시·금리·신용·변동성 원천을 실제
모델에 넣기 전 적용하는 실행용 PIT 계약이다. 기존
`config/structural_v6_research.json`의 연구 의도와 source id를 보존하되, 원천마다
공식 URL, 빈도, 예상 지연, 시각 의미, 빈티지 정책, 권리 profile, 공개 역할과
`enabled`/`ingested`/`status`를 같은 스키마로 정규화한다.

2026-08-27 현재 모든 항목은 `enabled=false`, `ingested=false`다. 이 카탈로그는
수집 완료나 모델 사용을 주장하지 않는다. OFR FSI는 strict parser만 구현되어 실제
ingest나 모델 사용을 뜻하지 않는다. Cboe 세 항목은 현 약관 검토 결과 서면 license
전에는 수집을 시작할 수 없어 `blocked_pending_written_license`이며, ingest는 구현하지
않는다.

## PIT 시장 총수익 입력

거시 release catalog와 별도로, 기존 Alpha Vantage weekly-adjusted 응답에 포함된
`dividend_amount`를 [`config/series.json`](../config/series.json)의
`research_fields`로 분리했다. 다음 실제 23-symbol 수집부터 같은 provider call 안에서
raw close와 함께 first-seen을 보존하며, 기존 모델 feature에는 들어가지 않는다.
주간 분석 canonical에는 audit column으로만 materialize되어 decision shadow의
split-safe price-only 분해에 사용된다. 예측 feature selector에서는 계속 제외된다.
weekly 응답에는 split coefficient가 없으므로 별도 corporate-action source가 권리·시각
검토와 회귀 테스트를 통과하기 전까지 PIT total-return challenger는 실데이터로 만들지
않는다. adjusted close의 현재 조정계수를 과거 PIT event로 역산하지 않는다
([Alpha Vantage documentation](https://www.alphavantage.co/documentation/)).

## 시간 계약

각 `ReleaseRecord`는 다음 다섯 축과 원문 digest를 보존한다.

- `observed_period_end`: 경제·시장 관측 대상 기간의 끝
- `source_released_at`: 공식 시각, 공식 파일 vintage 시각, 또는 카탈로그에 명시된
  보수적 date-only proxy
- `provider_first_seen_at`: 프로젝트가 그 revision을 provider에서 처음 본 시각
- `system_retrieved_at`: 해당 raw payload를 실제 저장한 시각
- `revision_seq`, `raw_sha256`: 같은 기간의 revision 순서와 원문 identity

`operational_oos`의 가용시각은
`max(source_released_at, provider_first_seen_at)`이고,
`reconstructed_oos`는 공식 archive 시각만 사용한다. 두 track은 합치지 않는다.
`observed_period_end` 또는 가용시각이 decision보다 미래인 행은 항상 제외한다.

뉴욕 금요일 16:00 cutoff는 `zoneinfo`로 계산해 DST를 반영한다. 날짜만 있는 release는
원천별로 다음 자정 또는 다음 eligible weekly decision으로 미룬다. first-seen-only
원천은 실제 수신 전 시각을 복원하지 않는다. 따라서 월말, 같은 날 발표, 늦은 provider
응답도 암묵적인 자정/장마감 소급 없이 처리된다.

## 후보와 현재 admission 상태

| Source id | 원천 | 빈티지 역할 | 현재 상태 |
|---|---|---|---|
| `philadelphia_ads` | Philadelphia Fed ADS all vintages | 공식 timestamp vintage | planned, not ingested |
| `philadelphia_rtdsm` | Philadelphia Fed RTDSM | date-only vintage를 보수적으로 정렬 | planned, not ingested |
| `board_h41` | Fed H.4.1 dated releases | 중앙은행 유동성 archive | planned, not ingested |
| `board_h8` | Fed H.8 dated releases | 은행 신용 archive | planned, not ingested |
| `board_sloos` | Fed SLOOS releases | 대출 기준·수요 archive | planned, not ingested |
| `ofr_fsi` | OFR FSI | first-seen snapshot, 과거는 sensitivity | parser implemented, not ingested |
| `board_ntfs` | Fed near-term forward spread | 입력 vintage를 가진 shadow | planned, not ingested |
| `board_ebp` | Fed excess bond premium | revision-prone monthly shadow | planned, not ingested |
| `cboe_vix_1600_control` | VIX 16:00 control | exact timestamp only | blocked pending written license |
| `cboe_vix_1615_sensitivity` | VIX 16:15 sensitivity | 16:00 cutoff 뒤 값으로 분리 | blocked pending written license |
| `cboe_vix_term_structure` | timestamped VIX curve | synchronized exact timestamp | blocked pending written license |
| `dol_weekly_claims` | DOL claims archive | weekly release snapshot | planned, not ingested |
| `bls` | BLS employment/CPI archives | monthly release snapshot | planned, not ingested |
| `bea` | BEA GDP/PCE archives | estimate-vintage snapshot | planned, not ingested |
| `census_eits` | Census EITS archives | release-specific timestamp | planned, not ingested |

## Admission gate

한 원천을 `enabled=true`, `ingested=true`로 바꾸기 전에 다음 증거가 필요하다.

1. official archive parser와 raw SHA-256 snapshot이 재현된다.
2. source release, first-seen, retrieval 시각이 모두 timezone-aware로 저장된다.
3. correction/revision fixture와 future-prefix 불변성 테스트를 통과한다.
4. provider rights와 derived-only 공개 범위를 다시 확인한다.
5. 동일 origin의 shadow ablation을 완료하고 불리한 결과도 보존한다.

카탈로그 loader는 `ingested=true`인데 `enabled=false`이거나 상태가 `ingested` 또는
`certified`가 아닌 모순을 거부한다. 실제 feature admission은 이후 pipeline wiring과
독립 검증을 별도로 요구한다.
