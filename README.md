# US Equity Regime Lab

미국 증시의 **현재 주간 국면**과 **다음 주 국면 확률**을 point-in-time
정보로 산출하고, 날짜를 선택해 결과를 탐색하는 로컬 연구 프로젝트입니다.
실제 투자 주문·자산배분 백테스트는 범위에 포함하지 않습니다. 공개 페이지는
검토를 마친 live v5 개인·비상업 파생 결과 스냅샷을 제공합니다.

## 무엇을 보여주나

- `risk-on / transition / risk-off` 현재 국면과 세 상태의 확률
- 같은 시점에서 계산한 다음 주 국면 확률
- 다음 주 권위 예측과 분리해 비교하는 1·4·13주 내 `any-departure` 위험
- 예측 entropy·confidence와 별도로 표시하는 explicit-duration shadow nowcast
- 추세·스트레스·거시·금융여건 score와 주요 driver
- 동일한 walk-forward 조건에서 비교한 next-state·전환위험 모델 leaderboard
- `dataAsOf`, 미국 동부시간 cutoff, source freshness·권리·장애 상태

대시보드는 Python 결과를 다시 계산하지 않습니다. Python 파이프라인이
`web/data/regime-results.json`을 만들고, dependency-free 정적 웹 앱이 그
계약을 읽습니다. GitHub Pages는 원자료나 DB가 아니라 별도로 검증된
`publication/live/regime-results.json`, hash-bound V5/V4 비교 sidecar와 정적 자산
allowlist만 배포합니다.

공개 live 파생 결과: [sonchanggi.github.io/regime](https://sonchanggi.github.io/regime/)

## v5 운영·연구 계약

주간 자동화와 공개 계약은 v5다. 수동 `build`·`demo` CLI 기본값은 V4 재현을 위해
유지하며, raw V5 후보 생성은 `--contract v5`를 명시해야 한다. 첫 실제 표준 실행의
matched OOS 검토에서 Markov는
frozen v4 Markov와 정확히 일치해 champion을 유지했다. 다중 기억 앙상블과 FX
ablation은 사전등록 gate를 통과하지 못해 core 예측에 승격하지 않았다. 공개
payload에는 검토 결정을 기록하고, 비교 결과는 파생값 전용 sidecar로 함께 배포한다.

```bash
.venv/bin/regime-lab demo --contract v5 \
  --output build/v5-demo/regime-results.json \
  --artifacts build/v5-demo/artifacts
# 권리 확인과 Keychain 설정을 마친 live build에도 --contract v5를 명시
.venv/bin/regime-lab build --contract v5 --alfred-rights-confirmed \
  --output build/v5-live/regime-results.json \
  --artifacts build/v5-live/artifacts
```

H.10 prospective history는 모델 학습과 분리해 가볍게 축적할 수 있다. 이 명령은
`--contract v5`를 반드시 요구하고 live build와 같은 DB lock을 사용한다. H.10
private snapshot DB와 파생 상태 receipt만 갱신하며 공개 payload·artifacts·자동화
health와 V5 모델 결과는 변경하지 않는다.

```bash
.venv/bin/regime-lab collect-h10 --contract v5
# 기본 receipt: build/v5-h10/collection-receipt.json
# 필요할 때만 명시적인 model cutoff와 격리된 receipt 경로를 지정
.venv/bin/regime-lab collect-h10 --contract v5 \
  --as-of 2026-08-21T20:00:00+00:00 \
  --receipt build/v5-h10/collection-receipt.json

# 공식 release archive 민감도 bootstrap/증분 갱신
.venv/bin/regime-lab collect-h10 --contract v5 \
  --official-release-archive-ingest \
  --archive-start 2022-01-01 \
  --archive-through 2026-08-21 \
  --as-of 2026-08-21T20:00:00+00:00
# 기본 receipt: build/v5-h10/archive-collection-receipt.json
```

월요일 H.10 공개 이후 매주 한 번 실행하면 실제 first-seen 기준 표본이 누적된다.
Receipt에는 원자료 없이 snapshot 증감, source/FX 상태, last-good 사용 여부,
156주 공통표본 readiness와 archive의 파생 lineage 개수·정책·인덱스 SHA-256만
기록한다. Archive 명령은 공식 JSON 인덱스를 매회 검증하고 새 release event만
가져오며, 중단된 page는 비공개 DB cache에서 재개한다. 기존 DB도 zero-delta
refresh에서 관측값을 다시 쓰지 않고 cache 기반 lineage provenance를 승격한다.
스케줄 설치는 별도 opt-in 작업이다.

실제 profile·표본 상한·bootstrap 횟수와 사전등록 override는
`model.execution_parameters`에 SHA-256으로 결속한다. quick 데모의 199회와
standard/full의 1,999회를 같은 실행으로 오인하지 않는다.

v5는 현재 hard state에 대한 `membership`과 다음 주 `forecast probability`를
분리한다. 1·4·13주 방향성 위험은 horizon 끝 상태가 아니라 **처음 현재 상태를
떠날 때 도착하는 상태**를 예측하며, 기존 any-departure 확률과 합계가 일치한다.
현재 spell은 우측 검열해 상태별 Kaplan–Meier 조건부 생존율과 52주 제한
평균 잔여기간(RMST)을 산출한다.

Federal Reserve H.10의 Broad·AFE·EME 달러 지수와 고정 9개 통화쌍을
`양수 = USD 강세`로 정규화한다. 최초 수집 이전의 공개시점을 소급 복원하지 않고,
관측·수정값은 실제 `first_seen` 이후 origin에만 투입한다. FX는 v4 control,
Broad 추가, bilateral 추가, 전체 FX 추가의 네 shadow ablation은 공통 156주가
쌓인 뒤 9/9 pair, 104주 expanding 학습, 1주 target purge, 고정 L2 multinomial
logistic으로 동일 origin에서 평가한다. 화면용 context의 6/9 기준과 모델 평가의
9/9 기준은 다르다. gate 통과도 자동 승격하지 않고 별도 승격 심사를 거친다.
평가 행은 원 FX 값 없이 `fx-ablation-oos.csv`에 남겨 metric과 gate를 재검산한다.
국면별 자산 통계는 state `t`를
관측한 뒤 `t+1`부터 측정하는 설명 통계이며 매매·배분 신호가 아니다.

세부 계약은 [v5 사전등록](docs/structural-v5-preregistration.md),
[V5 공개 결정](docs/v5-release-decision.md), [방법론](docs/methodology.md),
[피처 카탈로그](docs/feature-catalog.md),
[연구 참고문헌](docs/references.md)에 고정한다. 공개물에는 파생 결과와 상태만
포함하며 H.10 원자료·로컬 DB·provider payload는 포함하지 않는다.

## 빠른 실행: 격리된 연구용 데모

```bash
python3 -m venv .venv
# Apple Silicon에서 XGBoost가 요구하는 OpenMP runtime
brew install libomp
.venv/bin/pip install -e '.[test,hmm]'
.venv/bin/regime-lab demo \
  --output build/public-demo-source/regime-results.json \
  --artifacts build/public-demo-artifacts
.venv/bin/python scripts/package_public_demo.py \
  --payload build/public-demo-source/regime-results.json \
  --output dist/public-demo
.venv/bin/python -m http.server 8765 --directory dist/public-demo
```

브라우저에서 `http://127.0.0.1:8765/`를 엽니다. 데모는 UI와 전체 분석
경로를 재현하기 위한 고정 seed 모의 자료이며, 실제 미국 시장 판단으로
표시되거나 사용되지 않습니다. 패키징 스크립트는 `index.html`, `styles.css`,
`app.js`와 검증된 synthetic payload만 새 디렉터리에 복사합니다. payload가
`meta.mode=demo`가 아니거나 모든 source가 `synthetic_fixture`가 아니면 실패하며,
기존 출력 디렉터리를 덮어쓰지 않습니다.

## 공개 live 파생 결과 갱신

`web/data/regime-results.json`, `artifacts/latest/`, SQLite와 provider raw/cache는
계속 로컬 전용입니다. 공개 갱신 시 검토 완료 payload와 그 bytes에 결속된 비교
sidecar를 `publication/live/`에 함께 반영하고 다음 패키징 계약을 확인합니다.

수동 검토는 raw V5의 validate·독립 audit, frozen V4 matched comparison,
`promote_v5_publication.py`, 검토 완료 bytes에 대한 comparison 재생성, 재검증 순서로
수행합니다. 원본 candidate·artifact를 공개 디렉터리에 직접 복사하지 않습니다.

```bash
.venv/bin/regime-lab validate build/v5-live/regime-results.json
.venv/bin/python scripts/audit_outputs.py \
  --payload build/v5-live/regime-results.json \
  --artifacts build/v5-live/artifacts --mode live
.venv/bin/python scripts/compare_v5_to_frozen_v4.py \
  --v5-artifacts build/v5-live/artifacts \
  --v5-payload build/v5-live/regime-results.json \
  --v4-artifacts artifacts/baselines/v4-20260821 \
  --output build/v5-live/v5-vs-v4-comparison.json
.venv/bin/python scripts/promote_v5_publication.py \
  --candidate build/v5-live/regime-results.json \
  --comparison build/v5-live/v5-vs-v4-comparison.json \
  --output build/v5-release/regime-results.json
.venv/bin/python scripts/compare_v5_to_frozen_v4.py \
  --v5-artifacts build/v5-live/artifacts \
  --v5-payload build/v5-release/regime-results.json \
  --v4-artifacts artifacts/baselines/v4-20260821 \
  --output build/v5-release/v5-vs-v4-comparison.json
.venv/bin/regime-lab validate build/v5-release/regime-results.json
.venv/bin/python scripts/audit_outputs.py \
  --payload build/v5-release/regime-results.json \
  --artifacts build/v5-live/artifacts --mode live
```

```bash
.venv/bin/python scripts/package_public_demo.py \
  --payload publication/live/regime-results.json \
  --comparison publication/live/v5-vs-v4-comparison.json \
  --publication-mode live-derived \
  --acknowledge-personal-noncommercial-publication \
  --output dist/public-dashboard
.venv/bin/python scripts/verify_public_package.py dist/public-dashboard
```

패키지는 `index.html`, `styles.css`, `app.js`, 파생 결과 JSON, V5/V4 비교 sidecar,
manifest만 포함합니다. 검토 완료 V5 표식·최소 52주·정확한 source 이용범위·V4
Markov exact parity·금지된 원자료 필드·credential 패턴·파일 inventory·해시를 모두
검사하며, DB·원관측치·모델 artifact는 복사하지 않습니다.
`main`의 Pages workflow도 API를 호출하지 않고 이 공개 스냅샷만 재검증해
`https://sonchanggi.github.io/regime/`에 배포합니다.

## 주간 수집·재학습·배포 자동화

자동화는 private revision DB와 macOS Keychain을 보존하기 위해 두 실행면으로
분리합니다. 로컬 macOS LaunchAgent가 수집·재학습·감사·공개 후보 승격을 담당하고,
GitHub-hosted Pages workflow는 provider 호출 없이 검증된 파생 결과만 배포합니다.

```bash
# API 호출 없이 cutoff, origin/main, 공개 dataAsOf와 실행 필요 여부 확인
.venv/bin/regime-lab automation preflight

# 아래 두 권리 범위를 직접 확인한 뒤 LaunchAgent를 설치·갱신합니다.
# 설치 직후 catch-up due-check를 한 번 수행합니다.
.venv/bin/regime-lab automation install \
  --alfred-rights-confirmed \
  --acknowledge-personal-noncommercial-publication

# 등록 및 최근 실행 health 확인
.venv/bin/regime-lab automation status
```

LaunchAgent는 로컬 시각 03:17·09:17·15:17·21:17과 로그인 시에 누락분을
확인하지만, Python gate가 새 Friday 16:00 `America/New_York` cutoff에서 24시간이
지난 경우에만 실제 작업을 시작합니다. 실패 시에도 기록된 `next_retry_at` 전에는
provider나 모델을 다시 실행하지 않으며, 인증·권리·schema·dirty checkout 문제는
로컬 상태가 바뀔 때까지 차단합니다. 공개 snapshot이 이미 같은 cutoff이면
Keychain·provider·모델을 건드리지 않고 종료합니다.

확인된 transient 원인을 코드나 로컬 환경에서 수정한 뒤 예약 시각 전에 즉시
재검증해야 할 때만 `regime-lab automation run --force-retry`를 사용합니다. 이
operator override는 transient 시간 지연만 건너뛰며 quota와 blocked gate, provider
예산, AC, Git 및 publication 검증은 우회하지 않습니다. LaunchAgent 자체는 이
옵션을 사용하지 않습니다.

실제 실행은 다음 순서를 fail-closed로 적용합니다.

1. 단일-process lock, clean tracked working tree, `origin/main` source 일치와 Git
   fetch/push 경계를 확인합니다. 새 수집 직전에만 180일 만료의 로컬 권리 확인 파일과 AC
   전원을 확인합니다. 설치 명령의 첫 번째 동의는 ALFRED를 로컬에 저장해 ML 학습에
   쓰는 권리, 두 번째 동의는 파생 결과를 개인·비상업 공개하는 권리를 뜻합니다.
2. Alpha Vantage 23개 종목과 bounded retry 2회를 합친 25 credits 전체가
   rolling-24h ledger에 들어갈 수 있는지 예약 없이 먼저 확인하고, 실제 collector가
   다시 하나의 transaction으로 원자 예약합니다. Alpha는 120초 timeout·최대 3회,
   ALFRED는 60초 timeout·최대 4회로 제한합니다.
3. V5 build가 같은 DB lock 아래 H.10 prospective snapshot을 한 번 갱신합니다.
   Alpha/ALFRED/H.10 전체 pass가 exact cutoff와 source health gate를 통과한 경우에만
   분석을 시작합니다. provider가 degraded이거나 수집 후 AC 전원이 분리되면 분석
   전에 중단하고 secret-free receipt와 다음 재시도 시각을 남깁니다.
4. 수동 build 경로와 분리된 `build/weekly-automation/generation-v5/`에서 `standard`
   live build를 수행한 뒤 SQLite `quick_check`, payload validation,
   `audit_outputs.py --mode live`, frozen V4 matched comparison을 통과시킵니다. 검토
   gate가 유지될 때만 별도 promotion 단계가 공개 표식을 추가합니다.
5. 정확한 cutoff, 세 source `ok`, 최신 주 `ok`, Alpha coverage, forecast fallback
   부재를 확인합니다. 허용된 모델 진단 경고는 그대로 공개하되 provider degradation은
   자동 공개하지 않습니다.
6. 사용자 working tree와 분리된 임시 checkout에서 preflight로 고정한 원격 SHA와
   publication 경로의 regular-file 상태를 다시 확인한 뒤
   `publication/live/regime-results.json`과 검증된 비교 sidecar를 함께 commit/push합니다. Pages
   workflow는 배포 직전 `main` SHA가 자기 커밋과 같은지 확인합니다. 완료 후 public
   payload·comparison bytes, manifest SHA-256·`dataAsOf`·HTML을 다시 읽습니다. 불일치하면 같은 검증
   snapshot을 유지하는 빈 복구 commit을 원격 HEAD 고정 하에 push하여 Pages를 다시
   시작하고, 공개 파일이 정확히 일치할 때까지 기다립니다. GitHub CLI token에는
   의존하지 않습니다.

상태, lock, 검증된 재시도 후보와 launchd 로그는 Git에서 제외된
`build/weekly-automation/`에 저장합니다. 권리 확인은 credential 없이
`data/automation/authorization.json`에 `0600`으로 저장됩니다. push나 Pages가
실패하면 이전 공개 페이지는 유지되고, 다음 실행은 source tree·핵심 config hash·
`generation_id`가 모두 같은 cached candidate부터 재개해 provider 호출과 장시간
재학습을 반복하지 않습니다. 수동 live build와 예약 build는 같은 DB lock을 공유하며,
payload와 artifact는 서로 다른 경로를 사용해 예약 audit 중 교체될 수 없습니다.
장시간 child process는 2분 heartbeat를 기록하고 `caffeinate -s`와 Standard QoS로
AC 연결 중 system sleep을 막습니다. `automation status`는 plist drift, unload,
failed health, stale heartbeat를 운영 정상으로 오인하지 않습니다. 최초 실패·실패
유형 변경·복구에는 credential이나 원문 예외 없이 중복 억제된 macOS 알림을 보냅니다.
자동화 해제는 다음과 같습니다.

```bash
.venv/bin/regime-lab automation uninstall
```

## 실데이터 실행

기본 실행은 macOS Keychain의 `regime-fred-api-key`,
`regime-alpha-vantage-api-key`를 프로세스 환경으로만 잠시 주입합니다. 키
값을 채팅, `.env`, 명령 출력, SQLite, JSON, Git에 남기지 마세요.

```bash
.venv/bin/regime-lab build \
  --config config/series.json \
  --profile standard \
  --alfred-rights-confirmed
.venv/bin/regime-lab validate web/data/regime-results.json
```

이미 안전한 프로세스 환경에 값을 주입한 경우에만 `--from-env`를 함께
사용합니다. `--alfred-rights-confirmed`는 어느 경로에서도 생략할 수 없습니다.

`ALFRED_ML_RIGHTS_ACK=1`은 단순 약관 동의가 아니라, 이 프로젝트의 로컬
저장·ML 학습 사용 범위를 포함하는 권리를 사용자가 확인했다는 fail-closed
표시입니다. 값이 없으면 ALFRED client는 네트워크 요청 전에
`license_blocked`로 종료합니다. 공식 약관은 실행 직전에 다시 확인하세요:
[FRED Terms](https://fred.stlouisfed.org/legal/terms/),
[FRED API Terms](https://fred.stlouisfed.org/docs/api/terms_of_use.html),
[Alpha Vantage Terms](https://www.alphavantage.co/terms_of_service/),
[Federal Reserve H.10](https://www.federalreserve.gov/releases/h10/).

현재 공개 범위는 사용자가 확인한 개인·비상업 목적의 대시보드 파생 결과입니다.
Alpha Vantage 원시 시계열은 공개하지 않으며, ALFRED는 사용자가 확인한 저장·ML·
파생결과 범위와 각 series의 표시 조건을 전제로 합니다. 이 확인은 상업 이용이나
원자료 재배포 권리로 확대되지 않습니다. Federal Reserve H.10은 공식 공개 자료로
출처 링크를 유지하되 XML·관측행은 공개하지 않습니다. 공개 앱에는 다음 FRED 고지와
공식 약관 링크를 유지합니다.

> This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.

Alpha Vantage 무료 tier는 일일 호출 수가 작으므로 기본 universe는 23개
ETF로 제한해 25회 한도에 두 번의 여유를 둡니다. refresh가 quota를 초과할
가능성이 있으면 유료 fallback을 시도하지 않고 `quota_exhausted`로 멈춥니다.
standard-free `daily_request_cap` 계약은 형변환 없는 정수 `25`로 고정하며,
다른 값은 provider 호출과 quota event 기록 전에 fail-closed 처리합니다.
공식 FAQ는 reset 시간대를 명시하지 않으므로 내부 한도는 calendar-day reset이
아닌 **rolling 24시간** 요청 event ledger로 집행합니다. 이전 UTC/뉴욕 날짜
counter가 발견되면 각 날짜 bucket을 성공·실패 batch provenance와 대조해
입증 가능한 호출 시각으로 이관하고, 시각을 확인할 수 없는 잔여 호출은 이관
시점부터 24시간 동안 사용량으로 유지합니다. 일반 무료 key가 여러 개여도 프로젝트
전체 한도는 합산 25회로 유지하며 key rotation으로 한도를 늘리지 않습니다.
Premium 또는 Alpha Vantage가 확인한 open-source/educational entitlement는
검증 가능한 plan·분당 한도·만료 정보가 있을 때만 별도 credential로 지원합니다.
각 수집은 첫 network 요청 전에 필요한 symbol 수 전체를 SQLite transaction으로
원자 예약합니다. 예약 후 실패하거나 process가 종료되어 쓰지 못한 credit도
24시간 동안 보수적으로 사용량에 남으며, 무료 plan 호출은 자동 retry하지 않습니다.
예약 뒤 실제 호출된 credit의 24시간은 해당 transport 시각부터 다시 계산됩니다.
마이그레이션 중에는 구버전 collector를 먼저 종료해야 합니다. 새 collector가
예약을 끝낸 뒤 구버전 process가 별도 날짜 counter로 호출하는 경우까지 두 binary가
상호 배제할 수는 없으며, 다음 새 collector 접근 시에는 이를 즉시 합산·차단합니다.

최초 실데이터 실행은 ALFRED 수정 이력을 보존하는 full base 때문에 로컬
SQLite가 수백 MB가 될 수 있습니다. 이후 정상 주간 실행은 ALFRED
`output_type=3`의 신규·수정 event와 Alpha Vantage의 신규·변경 row만 delta로
저장합니다. 실패 응답은 health·요청 provenance만 남기고 부분 observation은
저장하지 않습니다. full base와 성공 delta를 합쳐 읽으므로 과거 정보는
유지하면서 매주 전체 history를 중복 저장하지 않습니다.

ALFRED delta 수집은 주간 overlap의 달력일 범위를 먼저
`/series/vintagedates`로 조회하고, 해당 series에 대해 실제로 반환된 vintage만
`output_type=3` 요청에 전달합니다. 2026-08-12 live JSON 점검에서 UNRATE
요청에 discovery가 반환하지 않은 날짜를 함께 넣으면 HTTP 400이 발생했기
때문에 적용한 보수적 정규화이며, 모든 series와 미래 API 동작을 일반화한
공식 보장은 아닙니다. FRED가 실제 vintage가 없는 좁은 구간에 HTTP 500을
반환하는 경우에는 이 오류를 곧바로 빈 결과로 간주하지 않습니다. bounded
observation history로 discovery를 한 번 넓혀 schema-valid 응답을 받은 뒤 원래
주간 구간과의 교집합이 비어 있음이 확인될 때만 정상 empty delta로 처리하며,
넓은 조회도 실패하면 last-good을 유지하고 분석·배포를 중단합니다.

Alpha Vantage가 과거 adjusted 값을 나중에 바꾸면 변경값은 탐지 시점부터만
유효한 새 revision으로 추가됩니다. 이전 row가 응답에서 사라진 경우에는
삭제로 추정하지 않고 해당 수집을 `degraded`로 표시한 뒤 last-good history를
유지합니다. 아직 끝나지 않은 현재 주 row는 snapshot 당시의 완료 주 cutoff와
대조해 영구 제외하므로 다음 주에 삭제·수정으로 오인하지 않습니다.

단, 최초 Alpha Vantage baseline은 첫 수집 시점에 공급자가 반환한 adjusted
history입니다. 공급자가 과거에 어떤 adjusted 값을 보여 주었는지에 대한 원래
vintage는 이 endpoint만으로 복원할 수 없으므로, baseline 이전 가격 이력은
엄밀한 historical vintage가 아니라 초기 backfill입니다. 프로젝트는 첫 수집
이후 탐지한 변경만 prospective revision으로 보존하고 이 한계를 결과에 명시합니다.

## 정보 시점 계약

- 기준 시각: 완료된 미국 거래 주의 금요일 16:00 `America/New_York`
- 사용 조건: `available_at <= cutoff`
- 발표시각을 모르는 당일 거시자료: 보수적으로 다음 주부터 사용
- 월·분기 자료: 실제 발표 뒤 forward-fill하며 `age_days`, `is_filled`,
  `release_lag_days`를 함께 보존
- 금지: backward-fill, centered smoothing, full-sample scaling/PCA,
  smoothed HMM/Viterbi 상태를 실시간 label로 사용
- 가격 label: 시장 자료만 사용; 거시·금리·FX·유동성은 예측 input으로만 사용

자세한 정의는 [methodology](docs/methodology.md)를 참고하세요.

## 입력 자료와 구조 실험

Alpha Vantage의 23개 ETF 조정종가와 거래량을 이용해 주가 추세·변동성·낙폭,
시장 대비 상대수익률뿐 아니라 주가 상승 breadth, 추세 참여율, 수익률 dispersion,
방향 동조화와 13·26주 평균 pairwise correlation을 만듭니다. XLY/XLP,
XLK/XLU, HYG/LQD, HYG/IEF, TLT/SHY의 4·13주 상대수익률은 경기민감/방어,
성장/방어, 신용위험과 금리곡선 proxy로 사용합니다. 전체 ETF의 거래량 증가
breadth와 가격-거래량 confirmation도 포함합니다.

SPY·IWM·RSP·HYG·TLT는 공급자의 조정종가/원종가 비율을 같은 주의
open·high·low에 적용해 조정 OHLC를 만들고, high-low range, close location,
전주 조정종가 대비 opening gap을 계산합니다. 필드가 일부만 존재하면 값을
합성하지 않고 해당 수집을 실패 처리합니다.

ALFRED는 기존 금리·물가·고용·성장·달러·유동성 series에 Chicago Fed NFCI와
`NFCIRISK`, `NFCICREDIT`, `NFCILEVERAGE`, `NFCINONFINLEVERAGE`를 포함합니다.
수준·변화량 외에 과거 52주만으로 표준화한 4주 변화량을 사용하며, 월·분기
자료는 발표된 이후에만 주별 행으로 forward-fill합니다.

구조 v4 실험은 기존 label과 v3 기준선을 그대로 두고, 11개 GICS sector
breadth, H.15 Treasury curve level/slope/curvature, H.8 bank credit·funding,
ANFCI, 과거 release만 이용한 macro innovation을 추가합니다. 모델 쪽에서는
1주 이탈 hazard와 조건부 destination을 결합하고, 완료된 과거 OOS loss만으로
Markov·XGBoost·hazard-destination 구조의 비중을 갱신하는 causal ensemble을
비교합니다. 한 주 hazard를 1·4·13주로 누적한 monotone survival 값은
geometric-duration coherence shadow일 뿐 선정 후보나 운영 예측을 대체하지
않습니다. 사전 계약과 v3 SHA 기준선은
[structural v4 preregistration](docs/structural-v4-preregistration.md)에 고정했습니다.

## 모델 비교

v3 기준 next-state 비교군은 14개입니다. Persistence·smoothed Markov·majority 기준선에
elastic-net 및 ridge multinomial logistic, 현재 국면 dummy를 추가한
transition logistic, duration-aware TVTP stay/switch/destination hurdle,
shrinkage LDA, calibrated linear SVM, PCA-spline logistic, random forest,
extra trees, histogram gradient boosting, shallow regularized XGBoost를 같은
feature와 expanding walk-forward split으로 비교합니다. Gaussian HMM은 `full`
프로필에서만 추가합니다. v4는 사전등록한 `xgb_hazard_destination`과
`causal_dynamic_ensemble`을 더해 standard 16개, full 17개를 같은
walk-forward gate에서 비교합니다. V5 standard는
`causal_multiscale_ensemble`을 추가한 17개이며, 실제 검토에서는 Markov가
champion을 유지했습니다. V5 full은 Gaussian HMM을 더한 18개입니다. Deep learning은 현재 범위에서 제외하며,
HMM의 full-sample smoothed path는 label이나 실시간 판정에 쓰지 않습니다.

2023년 이전의 이용 가능한 OOS origin 전체만으로 provisional champion을
고정합니다. 각 challenger는 가장 좋은 probabilistic baseline보다 selection
Log loss가 최소 0.05 낮고, fallback이 없고, Brier가 0.01보다 더 악화되지
않아야 합니다. 주별 paired loss에는 13주 circular moving-block bootstrap
(seed 17, 1,999회)과 one-sided Holm 다중비교 보정을 적용합니다. 통과 모델이
없으면 Markov 등 가장 좋은 probabilistic baseline을 유지합니다.

이 프로젝트에서 2023년 이후 결과는 이전 후보군을 대상으로 이미 확인한
적이 있으므로, v3 후보군에 대해 더 이상 `untouched holdout`이라고
부르지 않습니다. 화면과 산출물에서는 `retrospective external-period
diagnostic`으로 명시하며, 이 구간의 결과로 champion을 교체하지 않습니다.
진단 구간에서 champion의 Log loss가 최우수 모델보다 0.05 초과 뒤처지면
`weak_generalization`과 `degraded` 상태는 계속 표시합니다. 후보·고정
hyperparameter·계산 budget은 `candidate-manifest.json`과 SHA-256으로 남깁니다.

next-state champion과 별도로 1·4·13주 전환위험 연구군은 empirical hazard,
homogeneous Markov hazard, duration-aware TVTP hurdle, regularized binary
logistic을 비교하고, 표준·full 프로필에서는 가용한 경우 shallow binary
XGBoost도 포함합니다. 여기서 사건은 **기준 국면과 다른 상태가 `t+1..t+h`
중 한 번이라도 등장하는 것**입니다. `t+h`의 마지막 상태나 특정 destination을
맞히는 문제가 아니므로, 떠났다가 돌아온 경로도 사건으로 셉니다.

각 horizon `h`에서 학습 label의 `target_end`는 예측 origin보다 반드시 앞서며
`h`개 origin을 purge합니다. 모델 family, Platt-logit calibration, 경보 threshold는
2023년 이전에 `target_end`까지 끝난 selection OOS만으로 정합니다. threshold는
그 과거 행의 balanced accuracy를 최대화하고 근거가 부족하면 0.5를 사용합니다.
진단 origin은 2023-01-01 이후부터 시작하며, horizon이 cutoff를 가로지르는 중간
`h-1`개 origin은 embargo로 양쪽 평가에서 제외합니다. 이후에는 family,
calibrator, threshold를 고정하되 각 estimator 자체는 해당 origin 전에 결과가
확정된 행까지 expanding 방식으로 다시 적합합니다. 이 구간은 계속
transition 산출물에서 `retrospective_diagnostic`으로 표시합니다. 전체 dashboard의
next-state 기간 역할은 `retrospective_external_period_diagnostic`으로 유지됩니다.

화면의 1주 `transition_probability`는 권위 있는 next-state 확률에서 계산한
`1 - P(S[t+1] = S[t])`이며 `transition_risk.1w`와 같은 값입니다. 4·13주 값은
별도 전환위험 benchmark가 제공합니다. 또한 고정 parameter explicit-duration
filter를 **shadow nowcast**로 병기하지만, 이는 민감도 점검일 뿐 canonical 3국면
label이나 다음 주 정답을 바꾸지 않습니다.

각 build는 payload와 모든 supporting artifact를 비공개 sibling transaction
directory에 먼저 완성합니다. 기존 payload와 artifact를 함께 recovery 위치로
옮긴 뒤 새 generation을 승격하며, 실패하면 둘 다 이전 generation으로
rollback합니다. 핵심 추가 artifact는 다음과 같습니다.

- `transition-oos-predictions.csv`
- `transition-model-leaderboard.csv`
- `transition-walk-forward-splits.csv`
- `nested-selection.csv`
- `transition-forecasts.csv`
- `transition-candidate-forecasts.csv`
- `transition-candidate-status.csv`
- `feature-manifest.json`
- `feature-ablation-manifest.json`
- `feature-ablation-oos-predictions.csv`
- `feature-ablation-leaderboard.csv`
- `stacking-weights.csv`
- `structural-forecasts.csv`
- `joint-survival-forecasts.csv`
- `state-label-history.csv`
- `weekly-state-forecasts.csv`
- `multiscale-ensemble-scales.csv`
- `directional-oos-predictions.csv`
- `directional-model-leaderboard.csv`
- `directional-walk-forward-splits.csv`
- `directional-selection-diagnostics.csv`
- `directional-forecasts.csv`
- `conditional-asset-outcomes.csv`
- `conditional-asset-statistics.csv`
- `fx-features.csv`
- `fx-coverage.csv`
- `fx-ablation-oos.csv`

v2 비교 기준은 로컬 ignored snapshot으로 고정했습니다. payload SHA-256은
`50ab693b15f5100b1e39d98356c88455b76a4a2c4a4c335e5882509568c5fe98`,
snapshot inventory SHA-256은
`09603aca14244fc00ee56f0d75a45192fc29a77c8f1a47b9927aef32d4fcbf0f`입니다.
v4의 직접 비교 기준인 frozen v3 payload SHA-256은
`de93c585117b2784750f586a4f84ad99964c63081b252ad7affd7a75bd797095`, inventory
SHA-256은 `8ef3778cc8c36faff0c80e2bf094f1f11bd6966ab3b7b2d6edb84ba292aff6b9`입니다.
독립 audit는 상수뿐 아니라 snapshot 전 파일과 preregistration 원본을 직접
해시하므로 이후 결과가 좋아지거나 나빠져도 기준선을 소급 변경할 수 없습니다.

## 주요 명령

```bash
.venv/bin/regime-lab demo --profile quick
.venv/bin/regime-lab build --config config/series.json --profile full --alfred-rights-confirmed
.venv/bin/regime-lab validate web/data/regime-results.json
.venv/bin/python scripts/audit_outputs.py
.venv/bin/pytest
```

명령과 화면이 `demo`, `degraded`, `live`를 명시적으로 구분합니다. 핵심
source가 stale·차단·누락이면 마지막 값을 현재 값처럼 조용히 연장하지 않고
결과를 `unavailable` 또는 `degraded`로 표시합니다.

실데이터 날짜 선택 범위는 2023+ 사후 진단 구간부터입니다.
2006–2022 자료는 label threshold, feature history, 모델 선정에 쓰이지만 당시
날짜에 오늘 고른 champion을 소급 표시하지 않습니다. 최초 `standard` 실행은
이 MacBook Air 환경에서 1시간 이상 걸릴 수 있으며, CLI는 약 5% 간격으로
walk-forward 진행률을 출력합니다. `quick`은 개발·UI 점검용입니다.

## 디렉터리

```text
config/                 source universe와 cutoff/model 기본값
src/regime_lab/data/    provider, PIT contract, snapshot store, as-of join
src/regime_lab/analysis feature, label, walk-forward models, evaluation
src/regime_lab/         CLI, payload contract, orchestration, local server
web/                    정적 동적 dashboard
tests/                  누수·계약·모델·UI 회귀 테스트
```

## 연구 한계

Regime은 관측 가능한 정답이 아니라 잠재적 시장 요약입니다. 높은 accuracy는
지속성 baseline만 반복해도 나올 수 있으므로 확률 score와 전환 성능을 함께
봐야 합니다. 현재 공개 산출물은 개인·비상업 연구용 파생 결과입니다.

개별 기업 재무·실적은 현재 무료 source에서 장기간의 발표 당시 값과 수정 이력을
일관되게 재현하기 어려워 직접 feature로 넣지 않았습니다. 대신 규모·동일가중·섹터
ETF와 회사채 ETF를 기업·breadth proxy로 사용합니다. 신뢰할 수 있는
point-in-time fundamentals source가 확보되기 전까지 오늘의 재무 이력을 과거
fold에 소급하는 방식은 허용하지 않습니다.
