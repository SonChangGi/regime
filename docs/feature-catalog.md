# Structural feature catalog

구조 피처는 기존 v3 입력에 더해지며 기존 이름·라벨·PIT 계약을 바꾸지 않는다.

| Block | Inputs | Weekly features | Causal contract |
|---|---|---|---|
| Sector breadth | XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY | 1/4주 상승·하락 비율, 중앙값·MAD 분산, 상승 리더 HHI, breadth 가속도, coverage | 해당 주까지 관측된 조정종가만 사용 |
| Broad / size / style | SPY, QQQ, IWM, DIA, RSP | 1/4주 상승 비율, coverage | sector와 별도 단면 |
| Cross-asset | SHY, IEF, TLT, HYG, LQD, GLD, UUP | 1/4주 상승 비율, coverage | equity breadth와 별도 단면 |
| Treasury curve | DGS3MO, DGS1, DGS2, DGS5, DGS7, DGS10, DGS20, DGS30 | Nelson–Siegel level, slope, curvature, coverage | 고정 λ=0.0609/month, 최소 4개 만기, 매주 독립 적합 |
| Bank credit | TOTBKCR, TOTCI, DPSACBW027SBOG, H8B3094NCBA | 총대출·C&I 4/13주 로그증가, C&I 비중, 예금조달 비율, 차입 비율, coverage | H8 차입금만 millions→billions 변환 |
| Release innovation | UNRATE, PAYEMS, INDPRO, RSAFS, HOUST, CPIAUCSL, PCEPI, GDPC1 | 신규 기간·revision event, delta, 과거 event 중앙값·MAD 표준화, coverage | 현재 event 이전 최대 12개 event만 사용, 최소 4개, non-event signal=0 |

`WeeklyDataset.feature_group_manifest`는 모델 입력의 각 열을 정확히 한 블록에 배정한다. 이를 기준으로 common-origin block ablation을 수행할 수 있다.
