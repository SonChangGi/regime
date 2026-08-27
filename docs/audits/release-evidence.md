# Regime 의사결정형 릴리스 증거

- 검증 기준일: 2026-08-28
- 구현 커밋: `c74d14b4477e82569bcc31f583c066c76a3b9e36`
- 데이터 generation: `20260827T150939.045526Z`
- 데이터 기준시각: `2026-08-21T20:00:00+00:00`
- 공개 URL: <https://sonchanggi.github.io/regime/>

## 결론

[GitHub Actions run 33091774127](https://github.com/SonChangGi/regime/actions/runs/33091774127)의 build와 deploy가 성공했다. Ubuntu CI는 `1,147 passed, 1 skipped, 6 warnings`였고, allowlist로 고정한 `personal_noncommercial_live_derived` 패키지만 Pages에 올렸다. 공개 URL에서 다시 내려받은 11개 파일은 로컬 승인 패키지 `build/pages-workflow-package-decision-grade-20260828`과 모두 byte-for-byte 일치했다.

배포 모델은 `causal_dynamic_ensemble`, 동결 회귀 기준선은 V4 Markov다. 최신 관측 주의 Transition membership은 61.9%, 다음 주 Transition 예측확률은 91.9%다. 현재 국면 이탈확률은 1주 8.1%, 4주 51.3%, 13주 89.0%로 공개 화면과 payload가 동일하다.

## 1. CI와 last-good 보존

최종 성공 전에 두 실행이 Pages upload 전에 중단됐다.

1. run `33089938026`: Linux `ARG_MAX`를 넘긴 브라우저 fixture 전달과 로컬 `.venv` 경로 의존 테스트를 발견했다.
2. run `33090654106`: 남아 있던 한 건의 `node -e` 호출을 발견했다.
3. run `33091774127`: 모든 Node fixture를 stdin으로 전달하고 현재 Python interpreter를 사용해 build와 deploy가 성공했다.

앞선 두 실행은 artifact upload 전 종료돼 기존 공개 결과를 덮어쓰지 않았다. 최종 실행은 final-SHA 확인, 전체 테스트, 생성 contract 검증, 공개 경계 검사, package 검증, Pages upload와 deploy를 차례로 통과했다.

## 2. 공개 패키지 readback

공개 URL의 각 파일을 cache-busting query로 새 임시 디렉터리에 내려받아 로컬 승인 패키지와 비교했다. 결과는 `11/11 byte parity ok`였다.

| 파일 | 공개 SHA-256 |
|---|---|
| `index.html` | `56b1a000e2708b670b62ee6b7f876c14f37cec97eee44993ef2d42d73b586f55` |
| `app.js` | `700936a639955ba083d2fe6f8f2f8258bffdce9dc08aa7daf1e46c1736d4a353` |
| `styles.css` | `ad9423885fcd7862ac1a61d18681add9cbaecf35888efc2562483fe437f96b07` |
| `operating-contract.generated.js` | `5078635e5a1fb2b9300e56f045b51c4e85ee047a21a3f64a15cd4e69261c9a20` |
| `data/regime-core.json` | `241fdb26159ddef52c77d21c9fd3fe2b06bd0c3256672edf268054c57cc7355c` |
| `data/regime-research.json` | `2930e8481848d3485b5116345b433dbb2e42638913b29f2a6ccb154bce60e931` |
| `data/regime-results.json` | `b05deacbf914c13629f912838a112514fb72644126c5d0580e390f69ded05ff3` |
| `data/v5-vs-v4-comparison.json` | `66385e06970fa4752bc91be348aa3d10f1f24416bd21038ea540de941a8fa3f2` |
| `data/generation-manifest.json` | `835e470d5ae10bce4093772b57738e4f042b3f04c2e69fc26bd9d222f165df3f` |
| `data/selection-family-audit.json` | `8d4c38fb5b6090ee04a49c456354e433a53e5baa6f1b3f0af66ce9cad3c17a5f` |
| `publication-manifest.json` | `1d530f5728ed57ab2af614744db228fa766311503e2fb4654c3046c161964b70` |

core payload는 3,384,259 bytes로 전체 payload의 72.83%다. 첫 화면은 core만으로 완성되고 research sidecar는 독립적으로 로드된다. 공개 inventory에는 원시 관측치가 없다.

## 3. 경제·분석 검증

- 방향 예측 OOS 1,440건과 실운영 예측 18건을 분리했다.
- 조건부 자산성과는 18,882개 outcome, 54개 통계 셀을 같은 자산·같은 보유기간 B&H와 비교한다.
- 표본 단위 평균과 episode-equal 평균, 95% whole-episode bootstrap 구간을 함께 공개한다.
- 선택 평가는 365개 matched origin, holdout 189개, fallback 20개를 사용한다.
- 다중모형 비교는 Holm 보정과 Model Confidence Set을 적용했고 retained set은 5개다.
- 확률 품질은 log loss 0.485, Brier 0.295, calibration error 0.059다.
- 조기경보는 event 39건 중 on-time 3건, recall 7.7%, precision 60.0%, 연환산 false alarm 0.55회다.
- 비용 반영 확률 의사결정 shadow는 188주에서 연환산 수익률 7.8%, Sharpe 0.78, CER 6.2%, MDD -12.0%, 연환산 회전율 656.2%였다. SPY B&H와 60/40 비교결과를 같은 카드에 보존했다.

## 4. 실제 브라우저 QA

- 공개 desktop 1280×900과 mobile 390×844에서 horizontal overflow 0
- light/dark 테마, 키보드 포커스, 모바일 section navigation 정상
- 관측 주·모델·기간·basis·보유기간·자산의 URL 상태 복원 정상
- 현재 membership과 다음 주 예측확률을 서로 다른 시점 의미로 표시
- 자산성과 카드에 point estimate, 95% CI, B&H, excess, 표본 수, episode 수 표시
- research sidecar를 의도적으로 차단해도 core와 자산성과 화면 유지
- 공개 desktop/mobile console warning·error 0

## 5. 자동화 복구

공개 readback 뒤 LaunchAgent를 최종 `main` 커밋 기준으로 다시 설치했다. 설치 직후 실행은 공개 target이 이미 최신임을 확인하고 `stage=already_current`로 종료했다.

- `operational=true`
- `scheduler_healthy=true`
- `publication_current=true`
- installed/loaded/configuration/provider-rights/authorization 모두 정상
- `consecutive_failures=0`
- runtime fingerprint: `2f95cfb20ddfac479d6ae9c0e148743d68c16230f791acd8c96cc7c3f8d3b67d`
- 설치 plist SHA-256: `57f7a5134c6444c066cba2a984518238ea610d0e9a97be7b7722f1f5b2235176`

운영 run registry에는 `started → collecting → analyzing → completed → publication_reviewed → published` 전이가 checksum과 함께 남아 있다.
