# 종합 개선 최종 검증 — 2026-09-07

**2026-09-04 관측·공식 예측 192주를 보존한 발행 후보의 패키징과 계산 재현을 완료했다. 최종 브라우저 검증과 전체 pytest 1,417개를 통과했다. 실패는 없고, 별도 옵션이 필요한 장시간 V4 검사 1개는 건너뛰었다.**

작업본은 `/Users/changgison/projects/regime-improvements`이며, 확인 대상은 `build/release-20260904/pages-package/`다. 심사된 발행 파일 네 개를 작업본의 `publication/live`에 반영했다. 원본은 `build/release-20260904/source/`에 보존했다. **이 배포 전 검증 단계에서는 커밋·push·실제 배포를 수행하지 않았다.**

## 확인된 결과

| 항목 | 결과 | 근거 파일 (`build/release-20260904/` 기준) |
|---|---|---|
| 공식 내용 보존 | 192주 공식 예측, 현재 예측·모델 선정, 발행 원장 불변 | `weekly-research-replay-verification.json` |
| 방향성 전체 평가 | 과거 6,492행 + 최신 18행, fallback 0건 | `model-economics/directional-replay-verification.json` |
| 방향성 캐시 재생 | 학습 호출 0회, 0.647초, 원 실행과 정확히 일치; CSV 5개 해시 일치 | 같은 파일 |
| 확률·기간 계약 | 공식 1주 확률 불일치 0주, 기간별 누적확률·범위 위반 0건; 지속기간 지원 부족 4주 명시 | `model-economics/publication-preview-verification.json` |
| 정상 주간 연구 조립 | 캐시와 실제 입력으로 104.594초, 네트워크 호출 0회 | `weekly-research-replay-verification.json` |
| 연구 계산 재현 | 배분 v2·경보·하방·ablation JSON 정확히 일치; 운영 진단은 계산 시각만 다름 | 같은 파일 |
| 발행 준비 | 원본·입력 해시, frozen V4 비교, 선정 근거, 심사 및 정적 패키지 검증 통과 | `release-candidate/release-preparation.json` |
| 최종 Pages 패키지 | 15개 파일 해시 확인, core·연구·이력 2개 조각 포함 | `pages-package-verification.json` |
| 실제 브라우저 | 11개 모델 지표 대조, 과거 날짜·예측 기준 링크 복원, 기간·평균·경보·성과 조작, 1280px·390px 화면 통과; 콘솔 오류·경고 0건 | `browser-verification.json` |
| HTTP와 원본 보존 | 루트 및 `/regime/`의 30개 요청 정상·해시 일치, 동결 47개 파일 불변, 원본 프로젝트 clean | `http-and-preservation-verification.json` |
| 실행 환경 | Python 3.13.13·Node 24.7.0은 CI와 동일. macOS 공통 의존성 28개 버전 일치, `pip check` 통과 | `environment-verification.json` |
| 최종 전체 검사 | 1,417개 통과, 실패 0개, 별도 V4 검사 1개 건너뜀; 298.53초 | `final-ci-runtime-tests.log` |
| 주간 연결 영향 검사 | 신규 단위 검사와 CLI·발행·원장·연구 검사 총 46개 통과 | 프로젝트 루트의 `tests/test_publication_research.py` 및 관련 검사 |

주간 빌드는 같은 generation의 입력으로 새 연구를 완성한 뒤 기존 원자적 발행 절차에 넘긴다. 같은 주 재시도는 추가 자료 스냅샷을 재사용하며, 연구 실패 시 새 발행으로 넘어가지 않는다. 운영 진단의 범위는 이번 발행의 원장 추가 전까지이고, 계산 출처·해시·시각은 `research.extensions.build`에 기록한다.

## 전체 검사와 실행 범위

- **전체 pytest: 1,417개 통과, 실패 0개.** 첫 실행에서 확인한 기존 감사기의 연구 필드 누락을 수정했다. 새 필드만 명시적으로 허용하고 공유 연구 계약 검증을 연결했으며, 원 스키마·선정·비용·원장 검사는 유지했다. 잘못된 스키마·격리 플래그·확률·NaN·원 spec 변조를 거부하는 회귀 검사를 추가했다. 최종 로그: `build/release-20260904/final-ci-runtime-tests.log`.
- 건너뛴 1개는 `REGIME_RUN_V4_E2E=1`에서만 실행하는 별도 장시간 V4 번들 검사다. 이번 발행 후보의 frozen V4 비교와 기존 발행 심사는 통과했다. pandas/NumPy의 기존 timedelta deprecation 경고 6건이 남아 있으며 실행 실패는 없다.
- 실제 Pages와 같은 패키징 명령을 실행했으며, 결과 15개 파일은 독립 발행 준비 패키지와 바이트까지 같다. 로컬 미리보기는 이 완성 패키지를 제공한다.
- 이 문서는 배포 전 로컬 검증 기록이다. 당시 GitHub CI 실행, 커밋·push·실제 배포는 수행하지 않았다. 이후 배포 상태는 [GitHub Actions](https://github.com/SonChangGi/regime/actions/workflows/pages.yml)에서 확인한다. Linux 전용 의존성은 로컬 macOS 검사에서 제외했다.

이번 변경은 해석·비교·실행 정의·주간 유지·화면 사용성을 개선한 것이다. 예측력이나 실현 수익률 향상이 확정됐다는 의미는 아니다. 방향 확률의 진단 Log loss는 1·4주에서 낮아졌고 13주에서 소폭 높아졌다. 연구 후보의 자동 승격은 없다.

[현재 로컬 미리보기](http://127.0.0.1:8766/regime/) · [개선 내용과 로컬 실행](comprehensive-improvements.md) · [현재 발행 스냅샷](current-snapshot.md)
