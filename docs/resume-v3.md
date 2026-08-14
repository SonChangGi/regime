# v3 resume checkpoint — 2026-08-12

The v3 implementation is complete, but the standard live walk-forward build
was intentionally interrupted at the user's request before it produced any v3
dashboard/artifact output.

## Safe current state

- `web/data/regime-results.json` remains the validated v2 live payload.
- `artifacts/latest/` remains the matching validated v2 artifact generation.
- The frozen v2 comparison is in `artifacts/baselines/v2-20260812/`.
- `data/regime.sqlite3` contains successful last-good expanded Alpha OHLC/volume
  and all 25 ALFRED series, so the next build should reuse the same cutoff
  without additional provider calls.
- No deployment has occurred.

## Resume

```bash
.venv/bin/regime-lab build \
  --config config/series.json \
  --profile standard \
  --alfred-rights-confirmed

.venv/bin/regime-lab validate web/data/regime-results.json
.venv/bin/python scripts/audit_outputs.py --mode live
.venv/bin/pytest -q
node --check web/app.js
```

The standard build restarts model computation from origin 1; model checkpoints
are progress messages, not serialized fit checkpoints. Supporting artifacts are
staged as a sibling directory and swapped as one directory generation. Payload
and artifacts carry the same `generation_id`, so the audit fails closed if an
interruption occurs between their two atomic replacements.

After a successful audit, start the local server and complete desktop/mobile,
light/dark, keyboard and console browser QA before asking for deployment
approval:

```bash
.venv/bin/regime-lab serve --port 8765
```
