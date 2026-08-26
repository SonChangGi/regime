# Regime 배포·공개 readback 증거

- 검증 기준일: 2026-08-27
- 구현 release commit: `ce73b253032cd1cbd510630bafc52c920d3e6958`
- 공개 URL: <https://sonchanggi.github.io/regime/>
- 증거 범위: CI, Pages 배포, 공개 CDN 바이트 readback, 실제 브라우저 QA, 주간 자동화 복구를 각각 분리한다.

## 결론

구현 release는 [GitHub Actions run 33023844247](https://github.com/SonChangGi/regime/actions/runs/33023844247)에서 build와 deploy가 모두 성공했다. Ubuntu CI는 `1,078 passed, 1 skipped, 6 warnings`였고, allowlist 기반 `personal_noncommercial_live_derived` Pages artifact만 업로드했다. 공개 사이트에서 다시 내려받은 8개 파일은 로컬 승인 패키지 [`build/pages-workflow-package-final-20260827`](../../build/pages-workflow-package-final-20260827)과 byte-for-byte 동일했고 [`scripts/verify_public_package.py`](../../scripts/verify_public_package.py)도 통과했다.

이 증거는 generation `20260826T184946.198911Z`가 실제 공개 consumer까지 전달됐음을 뜻한다. challenger label, shadow model, reconstructed ablation을 승격하거나 장기 prospective 성능·투자성과를 보장하는 증거는 아니다.

## 1. CI와 last-good 보존

최종 성공 전에 두 실행이 package upload 이전에 실패했다.

1. run `33021777847`: macOS와 Linux의 마지막 부동소수점 bit 차이를 raw-byte digest로 비교한 V1 회귀와, Linux `ARG_MAX`를 넘긴 `node -e` 테스트가 실패했다.
2. run `33022807558`: Node 입력은 stdin으로 고쳤지만 platform-dependent 수치 digest가 남아 실패했다.
3. run `33023844247`: V1은 고정 legacy reference implementation과 상태 SHA를 직접 비교하고, Node는 동일 assertion program을 stdin으로 전달해 성공했다.

앞선 두 실행은 공개 artifact 생성 전 중단됐으므로 기존 last-good Pages는 덮어쓰지 않았다. 최종 run은 stale-main 거부, exact release manifest·public boundary 검증, Pages upload와 deploy를 모두 통과했다.

## 2. 공개 패키지 readback

공개 URL에서 cache-busting query와 함께 다음 8개 파일을 새 임시 디렉터리에 내려받아 로컬 승인 패키지와 `diff -rq`로 비교했다. 차이는 0건이었다.

| 파일 | 공개 SHA-256 |
|---|---|
| `index.html` | `4acb699deed43c3f8c2a2d637db4d50e94c161c6983d6c763c667d5d7ae4aa63` |
| `app.js` | `77e9000e8e5bc0a8151a184c26920ec9b30c82af8baae3497df7b4fa50d459a3` |
| `styles.css` | `e36a95335a2e1501dbb4751d1a11456bb60809184bf3724dc2b7c7a81bd81937` |
| `data/regime-results.json` | `2ce89c38e49e2c186badf2810bb71fcbdfef6573906c4cf51ca3f904d6c0841b` |
| `data/v5-vs-v4-comparison.json` | `96afe3833e592ce13304cce5f800234816dec40872131d822ca2a646b0914d13` |
| `data/generation-manifest.json` | `93d80ac8741feaa68732a7b668921401c6707ed80c2a32320d03dcbd94f4fe28` |
| `data/selection-family-audit.json` | `288f44225e11e8107d8664f2dfb4a261c592f092ea069256b653407956bd217f` |
| `publication-manifest.json` | `c7030be563bf555318206ebb4261533825bacd07b8f634bb31c6e0e5abb95c34` |

독립 package verifier의 결과는 `ok=true`, `payload_mode=live`, `comparison_included=true`, `selection_family_included=true`, `payload_data_as_of=2026-08-21T20:00:00+00:00`였다. raw observation이나 비승인 provider payload는 공개 inventory에 없다.

## 3. 실제 브라우저 QA

공개 URL을 실제 in-app browser에서 다시 읽었다.

- 1280×720, 1024×768, 390×844에서 document horizontal overflow 0
- light/dark 전환과 keyboard focus 표시 정상
- console error 0
- canonical label이 SPY 추세·변동성·낙폭 기반임을 표시
- 관측 membership과 `t → t+1` 예측확률을 별도 패널·문구로 표시
- origin, 실제 발행시각, target, remaining horizon과 `operational_oos` 상태를 표시
- 공식 dynamic model, frozen V4 baseline, 연구 모델과 runner-up을 분리

## 4. 자동화 복구

공개 readback 뒤 `.venv/bin/regime-lab automation run --config config/automation.json --force-blocked-recovery`를 실행했다. target과 public data-as-of가 모두 `2026-08-21T20:00:00+00:00`라 provider 재수집·재학습 없이 `status=succeeded`, `stage=already_current`로 종료했다. `consecutive_failures=0`, `error_code=null`, `retry_class=null`, `operational=true`, installed/loaded/configuration/provider-rights/authorization 상태가 모두 정상이고 lock holder는 없다.
