# 운영·재현성·복구 계약

이 문서는 주간 V5 작업의 운영 상태, 재현 가능한 실행 환경, 체크포인트 재사용,
게시 추적, 비파괴 백업 복구 절차를 정의한다. API와 데이터 제공자 권한 계약은 별도
검토 대상이며 여기서는 변경하지 않는다.

## 상태는 세 축으로 읽는다

`automation status`의 단일 `operational` 값만으로 게시 성공을 판단하지 않는다.

| 필드 | 의미 | 성공으로 간주하지 않는 경우 |
|---|---|---|
| `scheduler_healthy` | LaunchAgent 설치·로드·설정, heartbeat, 권한·runtime authorization이 유효함 | Pages 결과가 최신이라는 뜻은 아님 |
| `publication_current` | 현재 목표 주차의 공개 payload를 실제 readback으로 확인함 | 수집·학습·패키징을 이번 실행에서 모두 수행했다는 뜻은 아님 |
| `end_to_end_proven` | 수집부터 공개 readback까지 완료한 적이 있음 | `already_current` 확인만으로 새로 설정되지 않음 |

학습 중 health에는 `completed_origins`, `total_origins`, `last_progress_at`,
`current_origin`, `current_model`을 기록한다. 설정된 no-progress 시간 동안 origin
체크포인트가 늘지 않으면 자식 process group을 종료하고 실패로 기록한다. 단순히
process가 살아 있거나 heartbeat가 갱신된다는 이유만으로 계산 진행을 추정하지 않는다.

## 실행·게시 provenance

`run-registry.jsonl`은 다음 상태를 순서대로 보존한다.

```text
started -> collecting -> analyzing -> completed
          -> publication_reviewed -> published
```

`publication_reviewed`는 검증된 candidate package를 cache한 뒤, `published`는 Pages
배포 후 public byte readback을 통과한 뒤에만 추가한다. 새 행은 file lock 아래 append와
`fsync`를 수행하고 행 자체 SHA-256을 포함한다. 기존 checksum 없는 event/1 행은 읽을 수
있다. 마지막에 newline 없는 부분 쓰기만 복구 대상으로 보존·격리하며, 완결된 행의
checksum·schema 손상은 조용히 건너뛰지 않는다.

## Runtime과 generation 결속

로컬 authorization에는 Python, OS/platform, 모델 관련 설치 package 버전,
`requirements-ci.lock` SHA-256의 canonical fingerprint가 들어간다. 현재 fingerprint가
authorization과 다르면 provider 호출이나 SQLite 변경 전에 실패한다. 같은 fingerprint는
automation health와 V5 `generation-manifest.json`에도 기록된다. 환경 또는 lock을 의도적으로
바꾼 뒤에는 automation을 다시 검토·설치해야 한다.

SQLite snapshot store의 schema 변경은 `PRAGMA user_version`으로 한 번만 실행한다.
지원 버전보다 새로운 DB나 버전과 실제 column이 모순되는 DB는 쓰지 않는다. source와
dataset은 snapshot 선택 SQL에, series 목록과 as-of는 observation SQL에 push down한다.
collection은 last-good 레코드를 한 번 읽고 source 및 `(source, series_id)`로 한 번만
grouping한다.

## Walk-forward 재사용 경계

체크포인트 v2는 origin마다 아래 요소를 묶은 `training_slice_sha256`을 저장한다.

- 해당 origin까지의 causal training feature/state prefix
- 그 origin의 prediction feature row와 target identity
- benchmark parameter/model manifest와 source fingerprint
- Python 구현이 요구하는 checkpoint implementation version

새 주차 append처럼 기존 origin의 위 identity가 그대로면 완료 record를 새 namespace로
검증 후 재직렬화해 재사용한다. 역사 revision, model/runtime/parameter 또는 source
fingerprint가 영향을 준 origin은 재학습한다. v1 체크포인트는 계속 읽을 수 있지만
origin-level 재사용 증거로 승격하지 않는다.

## 공개 payload 분리

V5 package는 호환·감사용 `data/regime-results.json`을 유지하면서 다음 projection을 함께
생성한다.

- `data/regime-core.json`: `research`를 제외한 payload와 generation/source-payload SHA,
  research sidecar path/SHA
- `data/regime-research.json`: 같은 generation/source-payload SHA와 research 객체

두 파일은 `publication-manifest.json`의 byte 수와 SHA-256 inventory에 포함된다. packaging
staging과 upload 전 verifier는 원본 payload projection, generation, source byte hash,
sidecar hash를 다시 확인한다. core는 full payload보다 작고 3.5 MB 이하이며 source의
85% 이하라는 초기 전송 budget도 통과해야 한다. research가 느리거나 unavailable이면 core 화면은 유지할 수
있지만 다른 generation의 research를 합치지는 않는다.

## 비파괴 backup과 restore drill

정책 source of truth는 `config/recovery-policy.json`이다. `automatic_deletion`은 반드시
`false`이며 `retention_valid_generations`는 정리 실행 값이 아니라 검토 기준이다. routine
backup은 기준을 초과한 valid generation, legacy 또는 corrupt generation을 삭제하지 않는다.

백업 전에 다음을 계산해 하나라도 한도를 넘으면 SQLite online backup 전에 실패한다.

- source DB/WAL/SHM의 예상 byte 수와 단일 source ceiling
- 기존 backup inventory + 예상 세대의 total byte ceiling
- 완료 후 남아야 하는 filesystem free bytes
- 보수적 최소 처리량으로 계산한 예상 시간과 실제 backup/restore deadline

인벤토리와 최신 valid-current restore drill은 다음처럼 실행한다.

```bash
.venv/bin/python scripts/recovery_inventory.py \
  --backup-directory build/weekly-automation/database-backups \
  --checkpoint build/weekly-automation/generation-v5/.private-checkpoints \
  --preview build/v5-next-preview \
  --restore-drill
```

inventory는 `valid-current`, `legacy`, `corrupt`, `checkpoint`, `preview`의 건수·byte·mtime을
분리하고 retention 초과 수를 `review_only_no_automatic_deletion`으로 표시한다. restore drill은
격리된 임시 DB로 복사한 뒤 source hash/byte, `quick_check`, `integrity_check`,
`foreign_key_check`, schema와 core table count를 검증한다. drill 실패나 기한 초과 시 기존
백업은 그대로 보존한다.
