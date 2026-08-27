# V5 공개 결정 — 2026-08-28

## 결정

검토 완료 generation `20260827T150939.045526Z`를 주간 운영·공개 스냅샷으로
사용한다. 운영 champion은 `causal_dynamic_ensemble`, frozen 회귀 기준은 Markov로
분리한다. 멀티스케일·구조 확장·FX 맥락은 연구 결과로 유지하며 자동 승격하지 않는다.

## 모델·확률 근거

- 기준 데이터: `2026-08-21T20:00:00+00:00`, 공개 190주, latest fallback 0건
- matched selection OOS: 365 origins. 동적 앙상블 Log loss `0.326616`, Markov
  `0.357517`, 개선 `0.030901`, Holm 보정 p-value `0.024`, fallback 0건
- runner-up 멀티스케일 앙상블과의 차이는 단순성 허용범위 안이어서 사전등록된
  simplicity tie-break로 동적 앙상블을 유지
- 1·4·13주 이탈확률은 341 matched origins, 1,023 probabilities에서 독립 재계산.
  순서 제약 투영 전후 Brier는 모두 `0.179021`
- 확률 품질: Log loss `0.484861`, Brier `0.294723`, calibration error `0.058874`
- 조기경보: 실제 이탈 39건 중 정시 탐지 3건, recall `7.7%`, precision `60.0%`,
  false alarms `0.55/년`. 이 경고는 공개 화면의 모델 상태에 유지

## 경제적 해석

10bp 편도 비용과 1주 실행 지연을 반영한 188주 재구성 OOS에서 확률 shadow의
연환산 수익률은 `7.8%`, Sharpe `0.78`, CER `6.2%`, 최대 낙폭 `-12.0%`, 연
회전율 `656.2%`였다. 같은 구간 SPY B&H는 연환산 수익률 `21.3%`, Sharpe
`1.43`, CER `18.3%`였다. 따라서 확률 shadow는 투자전략으로 승격하지 않고
prospective ledger에서만 계속 기록한다.

## 공개 결속

- reviewed publication SHA-256:
  `b05deacbf914c13629f912838a112514fb72644126c5d0580e390f69ded05ff3`
- V5/V4 comparison SHA-256:
  `66385e06970fa4752bc91be348aa3d10f1f24416bd21038ea540de941a8fa3f2`
- selection-family audit SHA-256:
  `8d4c38fb5b6090ee04a49c456354e433a53e5baa6f1b3f0af66ce9cad3c17a5f`
- generation manifest SHA-256:
  `835e470d5ae10bce4093772b57738e4f042b3f04c2e69fc26bd9d222f165df3f`

공개 패키지는 allowlist 11개 파일과 파생 결과만 포함한다. 첫 화면은 core
`3,384,259` bytes로 렌더링하고 research `1,467,587` bytes는 독립 검증 후
지연 결합한다.
