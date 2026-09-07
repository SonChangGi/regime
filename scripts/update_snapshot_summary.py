#!/usr/bin/env python3
"""Generate the current snapshot summary from the preserved publication."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    payload = json.loads((ROOT / "publication/live/regime-results.json").read_text())
    meta = payload["meta"]
    latest = payload["weekly"][-1]
    path = ROOT / "docs/current-snapshot.md"
    path.write_text(
        f"# 배포 데이터 스냅샷\n\n이 문서는 `scripts/update_snapshot_summary.py`로 현재 publication에서 생성합니다.\n\n| 항목 | 값 |\n|---|---|\n| 데이터 기준 | {meta['data_as_of']} |\n| 생성 시각 | {meta['generated_at']} |\n| 운영 모델 | {payload['selection']['operating_champion']} |\n| 관측 이력 | {len(payload['weekly'])}주 |\n| 관측 국면 | {latest['current']['state']} |\n| 예측 대상 | {payload['forecast']['target_at']} |\n\n[발행 원본](../publication/live/regime-results.json) · [개선 로컬 실행](comprehensive-improvements.md)\n"
    )
    print(path)


if __name__ == "__main__":
    main()
