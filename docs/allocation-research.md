# T+1 자산배분 구현 원칙

## 요약

이 전략의 목적은 완료된 주 `t`에서 다음 주 국면 확률을 확정하고, 첫 정규
거래일 시가에 체결해 `t+1` 주의 수익을 얻는 것이다. 동결한 pre-2023 표본에서
국면별 다음 주 상대수익을 추정하고, 그 기대값만큼 60/40을 완만하게 기울인다.

핵심 규칙은 다음과 같다.

- 목표가중치: target-state 상대수익 기대값으로 60/40을 기울인 연속형 aim
- 거래: 5%p no-trade band, 50% partial adjustment, 주간 one-way 10% cap
- 비용: risky-asset 거래 명목에 기본 10bp, 20bp 스트레스
- 비교: 동일 집행 엔진의 cost-aware 60/40, SPY, 국면·모멘텀 ablation
- 섹터: 월 1회 중기 모멘텀과 국면 상대수익을 결합한 최대 15% shadow sleeve
- 선택: pre-2023 추정치와 2023+ 재구성 OOS를 분리하고 운영 원장은 동결

## 1. 완료된 정보만 다음 주 첫 시가에 집행한다

완료된 시장주의 종가까지 이용할 수 있었던 정보로 다음 주 국면 확률과
목표가중치를 계산한다. 주문 시점은 다음 주 첫 정규 거래일 시가다. 월요일이
휴장이라면 화요일 시가가 된다. 주문 결정이 이 시각에 늦으면 그 주는 거래하지
않고 기존 포지션을 유지한다.

성과는 동일한 체결 계약으로 비교한다.

- 예측 origin: 완료된 주 `t`
- 목표 주: 바로 다음 시장 주 `t+1`
- 진입: 목표 주 첫 정규 거래일 시가
- 보유 성과: 목표 주 시가부터 주간 종가까지
- 리밸런싱 전 가중치: 직전 보유분에 야간 가격 변화를 반영한 drifted weight

국면 예측 정확도와 투자 성과는 별도 지표다. 높은 분류 정확도가 자동으로 높은
자산배분 수익을 뜻하지 않으므로, 모든 후보는 동일 주차의 60/40과 비용 차감
효용으로 다시 평가한다.

## 2. 60/40에서 기대 상대수익만큼 움직이는 aim

### 2.1 target-state payoff decoder

동결한 pre-2023 selection 표본의 최근 최대 520주에서 목표 주 시가→종가의
`SPY-TLT` 상대수익을 실제 목표 국면 `S(t+1)`별로 추정한다. 유효 표본은 최소
52주이며, 상태별 표본평균은 52주의 0 prior로 축소한다.

```text
payoff[state] = n_state/(n_state + 52) * mean(SPY-TLT | S(t+1)=state)
expected_payoff = sum(p_forecast[state] * payoff[state])
```

`sigma_rel`은 같은 selection 표본의 주간 `SPY-TLT` 변동성이다. 실제 aim은 다음
식으로 계산한다.

```text
risk_scale = min(1, 0.03 / sigma_rel)
tilt = confidence * 0.20
       * tanh(expected_payoff / (0.25*sigma_rel))
       * risk_scale
aim_spy = clip(0.60 + tilt, 0, 1)
aim_tlt = 1 - aim_spy
```

여기서 3%는 포트폴리오 변동성 목표가 아니라 상대수익 decoder의 크기 제한이다.
`confidence`는 selection 예측이 52개 이상이고 해당 모델의 Log loss가 majority보다
낮을 때 0.50, 아니면 0이다. 0.75와 1.00은 사전 명세에 남아 있지만 pre-2023
예측확률이 없어 v1에서 선택하지 않는다.

## 3. 거래는 aim까지 한 번에 가지 않는다

aim과 drifted pre-trade weight 사이의 investor one-way 거리를 계산한다. `x`는
risky-asset 가중치, `c`는 현금 비중이다.

```text
one_way_gap = 0.5 * (sum(abs(delta_x)) + abs(delta_c))
```

다음 순서로 실제 주문 목표를 만든다.

1. `one_way_gap < 5%`이면 거래하지 않는다.
2. 거래할 때는 차이의 50%만 반영한다.
3. 조정 후 주간 investor one-way turnover가 10%를 넘으면 이동폭을 비례 축소한다.
4. 동적 후보의 기대편익이 예상 거래비용의 두 배 이하면 거래하지 않는다.

```text
partial_target = pretrade + 0.50*(aim - pretrade)
executed_target = cap_one_way_move(partial_target, pretrade, 0.10)
expected_benefit = dot(executed_target - pretrade, expected_returns)
cost_hurdle = 2 * (cost_bps/10000) * risky_asset_full_L1
```

이 규칙은 신호가 한 주 뒤 사라질 수 있다는 점과 반복 매매 비용을 함께 반영한다.
느리게 소멸하는 신호에는 여러 주에 걸쳐 접근하고, 작은 확률 변화에는 반응하지
않는다.

회전율과 비용은 다음처럼 분리한다.

- investor one-way turnover: `0.5*(sum(abs(delta_x)) + abs(delta_c))`
- risky-asset full-L1: `sum(abs(delta_x))`
- 거래비용률: `10bp * risky-asset full-L1`

기본 비용은 10bp, 스트레스는 20bp다. 최초 현금 진입은 aim까지 한 번에 집행하고
band·partial adjustment·주간 cap·경제성 hurdle을 적용하지 않지만 비용은 차감한다.
화면의 연환산 one-way 회전율은 이 최초 진입을 제외한다. 최초 진입 포함 수치와
full-L1 수치는 별도 필드로 보존한다. `transaction_cost_rate_sum`은 주별 비용률의
단순 합이며, 순자산 경로에는 각 주 비용을 복리로 반영한다.

## 4. 비교 전략

모든 비교 전략은 같은 주차, price-only 수익, 현금 수익, band·partial·cap과
비용을 사용한다. 기대수익 벡터가 있는 동적 후보에만 경제성 hurdle을 적용한다.

- `realistic_60_40`: aim을 SPY 60%, TLT 40%로 고정한 실행 기준
- `spy_buy_and_hold`: SPY 100%
- `regime_only`: 2절의 SPY/TLT aim
- `momentum_only`: 60/40에 섹터 모멘텀 sleeve만 결합
- `combined`: 국면 aim과 섹터 결합 점수를 함께 사용

v1 allocation gate에는 trailing-volatility-targeted 60/40을 넣지 않는다. 2절의
`sigma_rel`은 포트폴리오 위험 타기팅이 아니라 국면 tilt의 크기 조절에만 쓴다.

## 5. 섹터는 월 1회 중기 모멘텀 shadow sleeve로 제한한다

전통적인 경기국면별 추천 섹터표는 사용하지 않는다. 장기 실증에서 산업 모멘텀은
관찰되지만, 완벽한 경기순환 타이밍을 가정한 섹터 로테이션도 비용 후 우위가 거의
사라진다. 따라서 섹터는 core 자산배분을 대체하지 않고 작은 shadow sleeve로만
평가한다.

### 5.1 신호와 리밸런싱

목표 주 첫 정규 거래일의 달이 바뀔 때 직전 완료 주 종가까지의 신호로 순위를
갱신한다. 달력상 월 1회이며, 단순히 4주마다 거래하는 규칙이 아니다.

각 섹터의 SPY 대비 누적수익을 사용한다.

```text
momentum_score
  = 0.5*z(relative_return_26_to_4_weeks)
  + 0.5*z(relative_return_52_to_4_weeks)
```

최근 4주는 단기 반전과 월말 잡음을 줄이기 위해 제외한다. 상위 3개 섹터를
동일가중하고, sector sleeve는 전체 포트폴리오의 15%와 현재 주식 비중의 25% 중
작은 값으로 제한한다. 각 섹터의 전체 포트폴리오 비중은 최대 5%이며 재원은 SPY
비중에서 차감한다.

### 5.2 비교 후보와 역사적 일관성

후보는 두 개만 둔다.

- momentum-only
- 80% momentum + 20% regime-conditioned score

결합형이 momentum-only를 비용 차감 기준으로 이기지 못하면 국면 정보의 섹터
기여는 없는 것으로 처리한다. 과거 구간에는 당시 존재했던 9개 legacy sector만
사용한다. Real Estate와 Communication Services는 실제 상품·분류 도입 후 104주가
지난 시점부터 포함하며 현재 11개 섹터 구성을 과거에 소급하지 않는다.

섹터 sleeve도 포트폴리오 전체의 5% no-trade band, 50% partial adjustment,
주간 one-way 10% cap 안에서 함께 집행한다.

## 6. 선택 게이트는 비용 차감 성과로 결정한다

### 6.1 비교 계약

payoff와 모멘텀 기울기는 pre-2023 selection 표본에서 동결한다. 전략 성과는
2023+ 재구성 OOS의 공통 실현 주차에서 비교하며 운영 예측 원장은 선택에 쓰지
않는다. 10bp와 20bp 비용을 계산하고, certainty-equivalent return은 다음 식이다.

```text
CER = annualized_geometric_return - 0.5*3*annualized_volatility^2
```

### 6.2 실행 선택

`combined`는 다음 조건을 모두 통과해야 실행 기준으로 선택된다.

- selection 예측 52개 이상이며 모델 Log loss가 majority보다 낮을 것
- pre-2023 payoff 표본이 최소 52주일 것
- pre-2023 월별 상위 섹터의 평균 SPY 대비 수익이 양수일 것
- 10bp에서 cost-aware 60/40 대비 누적수익과 CER가 모두 높을 것
- 20bp에서도 cost-aware 60/40 대비 CER가 높을 것
- 연환산 investor one-way turnover 150% 이하
- 최대낙폭이 cost-aware 60/40보다 2%p 넘게 악화되지 않을 것
- 최초 진입을 제외하고 최소 2회의 실제 리밸런싱이 있을 것
- 10bp 누적수익과 CER가 `regime_only`, `momentum_only`보다 모두 높을 것

어느 하나라도 실패하면 60/40이 실행 기준으로 유지된다. 최초 현금 진입만 있고
반복 주문이 없는 정적 보유 결과는 동적 전략의 성공으로 인정하지 않는다.

### 6.3 운영 원장

현재 aim, target, 주문, 현금 수익률과 비용은 운영 예측 원장에 hash와 함께
동결한다. 만기 평가는 나중에 다시 계산한 가중치가 아니라 이 동결 target을
사용한다. 운영 원장은 v1 선택 게이트와 분리한다.

## 7. 핵심 1차 출처

### 자산배분 서적

- Ilmanen, **Expected Returns: An Investor's Guide to Harvesting Market
  Rewards** (Wiley, 2011): 과거 평균 하나가 아니라 이론·실현치·전망 지표를 함께
  보고 주식·채권·모멘텀의 기대수익을 판단하는 틀.
  [출판사 원문](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118467190)
- Ang, **Asset Management: A Systematic Approach to Factor Investing**
  (Oxford University Press, 2014): 자산 이름보다 서로 겹치는 위험요인과 투자자의
  불리한 국면을 중심으로 배분을 해석하는 틀.
  [출판사 원문](https://academic.oup.com/book/3342)

### 견고한 자산배분과 거래 마찰

- Jensen, Kelly, Malamud & Pedersen, **Machine Learning and the Implementable
  Efficient Frontier**, *Review of Financial Studies* (2026): 비용 차감 OOS
  frontier와 신호 지속성을 직접 최적화해야 한다.
  [저널 원문](https://academic.oup.com/rfs/advance-article/doi/10.1093/rfs/hhag022/8524346)
- Barroso & Saxena, **Lest We Forget: Learn from Out-of-Sample Forecast Errors
  When Optimizing Portfolios**, *Review of Financial Studies* (2022): 과거 OOS
  예측오차를 이용한 input 축소.
  [저널 원문](https://academic.oup.com/rfs/article-pdf/35/3/1222/46617212/hhab041.pdf)
- Tu, Chen & Xing, **Robust Portfolio Selection with Smart Return Prediction**,
  *Economic Modelling* (2024): 기대수익 예측과 robust portfolio decision의 결합.
  [저널 원문](https://www.sciencedirect.com/science/article/pii/S0264999324000750)
- DeMiguel, Garlappi & Uppal, **Optimal Versus Naive Diversification**,
  *Review of Financial Studies* (2009): 추정오차를 고려한 단순 기준 포트폴리오의
  필요성.
  [저널 원문](https://academic.oup.com/rfs/article-lookup/doi/10.1093/rfs/hhm075)
- Gârleanu & Pedersen, **Dynamic Trading with Predictable Returns and Transaction
  Costs**, *Journal of Finance* (2013): aim portfolio와 partial adjustment.
  [저널 원문](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12080)
- Bacchetta & van Wincoop, **International Portfolio Choice with Frictions**,
  *Review of Financial Studies* (2023): 기대수익 변화에 대한 점진적 포트폴리오 반응.
  [저널 원문](https://academic.oup.com/rfs/article/36/10/4233/7126484)

### 변동성 타기팅

- Moreira & Muir, **Volatility-Managed Portfolios**, NBER Working Paper 22208,
  이후 *Journal of Finance* (2017): 높은 변동성에서 위험을 줄이는 긍정적 근거.
  [NBER 원문](https://www.nber.org/papers/w22208)
- Cederburg et al., **On the Performance of Volatility-Managed Portfolios**,
  *Journal of Financial Economics* (2020): 103개 전략에서 실시간 OOS 성과와
  구조 안정성의 반증.
  [저널 원문](https://www.sciencedirect.com/science/article/pii/S0304405X2030132X)
- DeMiguel, Martín-Utrera & Uppal, **A Multifactor Perspective on
  Volatility-Managed Portfolios**, *Journal of Finance* (2024): 비용과 trade
  netting을 포함한 최신 검증.
  [저널 원문](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13395)

### 섹터 모멘텀과 분류

- Moskowitz & Grinblatt, **Do Industries Explain Momentum?**, *Journal of
  Finance* (1999): 산업 모멘텀의 기초 실증.
  [저널 원문](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00146)
- Vanstone, Hahn & Earea, **Industry Momentum: An Exchange-Traded Funds
  Approach**, *Accounting & Finance* (2021): sector ETF 기반 모멘텀 실증.
  [저널 원문](https://onlinelibrary.wiley.com/doi/abs/10.1111/acfi.12724)
- Molchanov & Stangl, **The Myth of Business Cycle Sector Rotation**,
  *International Journal of Finance & Economics* (2024): 경기순환 섹터표의 비용 후
  반증.
  [저널 원문](https://onlinelibrary.wiley.com/doi/10.1002/ijfe.2882)
- S&P Dow Jones Indices, **An Overview of GICS and S&P U.S. Sector & Select
  Industry Indices** (2026): 2016·2018년 섹터 구조 변경과 현재 분류 계약.
  [기관 원문](https://www.spglobal.com/spdji/en/documents/education/education-an-overview-of-gics-and-sp-500-sector-industry-indices.pdf)
