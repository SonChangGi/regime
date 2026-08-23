# V5 공개 결정 — 2026-08-23

## 결정

검토 완료 V5 파생 스냅샷을 주간 자동화와 GitHub Pages의 공개 계약으로 사용한다.
core champion은 Markov를 유지한다. `causal_multiscale_ensemble`과 H.10 FX 변형은
승격하지 않는다. 수동 build/demo의 기본 계약은 frozen V4 재현을 위해 그대로 둔다.

## 실데이터 검증 결과

- 표준 V5 payload: `data_as_of=2026-08-21T20:00:00+00:00`, 190 공개 주,
  latest forecast fallback 없음
- V5 Markov 대 frozen V4 Markov: 공통 OOS 552개, selection 365개와 diagnostic
  187개에서 확률 float·직렬화 token과 Log loss·Brier·balanced accuracy가 모두
  정확히 일치
- 다중 기억 앙상블: selection Log loss 개선 0.0310996563으로 사전등록 최소 0.05에
  미달해 비승격. diagnostic 개선은 선택 근거로 사용하지 않음
- FX ablation: 59개 공통 OOS origin에서 Broad·bilateral·전체 FX 세 변형 모두
  control보다 Log loss가 악화해 0/3 통과, 비승격
- model health: `review_due`; `weak_generalization`, `calibration_drift`를 공개 상태에
  보존

## 공개 결속

- raw candidate SHA-256:
  `7523a617600cb858ec26a1ed737fc03e5012ff0f34a6a968949f54ac39ddbd85`
- reviewed publication SHA-256:
  `042ed7eeeac7d038b723ad6c0031cb728d14a2bd09f4848a2f2b2d39f14cb105`
- reviewed comparison SHA-256:
  `380fb236b16c01a53e0ee565d6d71d3e977d1a403e8d0ed08d7db170b13b355f`
- frozen V4 payload SHA-256:
  `e58eda3f5519e1c3c340c671e6c6c1c69279dae068f9c21f9bedfde22e03b96b`

공개 payload의 `publication_review`가 검토 시각·candidate hash·champion·두 비승격
결정을 기록한다. `v5-vs-v4-comparison.json`은 공개 payload bytes와 frozen V4
inventory에 결속된 파생 진단만 포함한다. 원 H.10/Alpha/ALFRED 관측과 로컬 DB,
모델 artifact는 공개하지 않는다.
