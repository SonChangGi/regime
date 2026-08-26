#!/usr/bin/env python3
"""Create a fail-closed static dashboard package.

Only the three static application assets, one explicitly selected dashboard
payload, and the required derived-only V5 comparison sidecar are copied. Live
publication is limited to the user's personal,
non-commercial *derived result* snapshot: raw observations, databases, model
artifacts, credentials, and arbitrary files are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from regime_lab.schema import ContractError, validate_dashboard_payload
from regime_lab.config import project_root
from regime_lab.frozen_v4 import (
    FROZEN_V4_BASELINE,
    FROZEN_V4_INVENTORY_FILE_COUNT,
    FROZEN_V4_OOS_PREDICTIONS,
)
from regime_lab.integrity import (
    GENERATION_MANIFEST_SCHEMA_VERSION,
    IntegrityError,
    validate_generation_manifest,
    validate_lifecycle_consistency,
    validate_reviewed_candidate_hash,
)
from regime_lab.selection_family_audit import validate_selection_family_audit
from regime_lab.path_safety import UnsafeMutablePath, confined_mutable_path
from regime_lab.provider_rights import (
    ProviderRightsError,
    verify_provider_rights,
)
from regime_lab.publication_contract import (
    PublicContractError as PackagingError,
    V5_COMPARISON_SCHEMA_VERSION,
    V5_RESULT_VERSION,
    reject_raw_provider_material as _reject_raw_provider_material,
    rewrite_index_asset_versions,
    validate_index_asset_versions,
    validate_v5_comparison_sidecar,
)
from regime_lab.contract_v5 import V5_PUBLICATION_STATUS


STATIC_ALLOWLIST = ("index.html", "styles.css", "app.js")
PAYLOAD_DESTINATION = "data/regime-results.json"
V5_COMPARISON_FILENAME = "v5-vs-v4-comparison.json"
V5_COMPARISON_DESTINATION = f"data/{V5_COMPARISON_FILENAME}"
SELECTION_FAMILY_FILENAME = "selection-family-audit.json"
SELECTION_FAMILY_DESTINATION = f"data/{SELECTION_FAMILY_FILENAME}"
GENERATION_MANIFEST_FILENAME = "generation-manifest.json"
GENERATION_MANIFEST_DESTINATION = f"data/{GENERATION_MANIFEST_FILENAME}"
MANIFEST_DESTINATION = "publication-manifest.json"
PUBLICATION_MODE_DEMO = "demo"
PUBLICATION_MODE_LIVE_DERIVED = "live-derived"
V4_RESULT_VERSION = "weekly-regime-result-v4"
LIVE_SOURCE_LICENSES_BY_RESULT_VERSION = {
    V4_RESULT_VERSION: {
        "alpha_vantage": "private_noncommercial",
        "alfred": "user_confirmed_ml_storage_derived",
    },
    V5_RESULT_VERSION: {
        "alpha_vantage": "private_noncommercial",
        "alfred": "user_confirmed_ml_storage_derived",
        "frb_h10": "federal_reserve_board_public_domain_citation_requested",
    },
}
RIGHTS_PROVIDER_BY_SOURCE_ID = {
    "alpha_vantage": "alpha_vantage",
    "alfred": "fred_alfred",
    "frb_h10": "frb_h10",
}


def _read_regular_file(path: Path, *, label: str) -> bytes:
    if path.is_symlink():
        raise PackagingError(f"{label} must not be a symbolic link: {path}")
    if not path.is_file():
        raise PackagingError(f"{label} is missing or is not a regular file: {path}")
    return path.read_bytes()


def _decode_payload(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackagingError("payload must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise PackagingError("payload root must be a JSON object")
    return payload


def _decode_v5_comparison(raw: bytes) -> dict[str, Any]:
    try:
        report = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackagingError("V5 comparison must be valid UTF-8 JSON") from exc
    if not isinstance(report, dict):
        raise PackagingError("V5 comparison root must be a JSON object")
    return report


def validate_public_demo_payload(payload: dict[str, Any]) -> None:
    """Fail closed unless the payload is an explicitly synthetic demo."""

    # Rights metadata is necessary but not sufficient for publication.  The
    # packaged document must also satisfy the exact dashboard contract; this
    # prevents a synthetic-looking but malformed payload from becoming a
    # public page that only fails after browser load.
    try:
        validate_dashboard_payload(payload)
    except ContractError as exc:
        raise PackagingError(f"dashboard payload contract is invalid: {exc}") from exc

    meta = payload.get("meta")
    if not isinstance(meta, dict) or meta.get("mode") != "demo":
        mode = meta.get("mode") if isinstance(meta, dict) else None
        raise PackagingError(f"only meta.mode=demo may be packaged (got {mode!r})")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PackagingError("demo payload must contain a non-empty sources array")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise PackagingError(f"sources[{index}] must be an object")
        source_id = source.get("id")
        license_class = source.get("license_class")
        if not isinstance(source_id, str) or not source_id.startswith("synthetic_"):
            raise PackagingError(
                f"sources[{index}].id must identify a synthetic fixture"
            )
        if license_class != "synthetic_fixture":
            raise PackagingError(
                f"sources[{index}].license_class must be synthetic_fixture"
            )


def validate_public_live_derived_payload(payload: dict[str, Any]) -> None:
    """Validate a personal, non-commercial snapshot containing derived results only."""

    _reject_raw_provider_material(payload)
    try:
        validate_dashboard_payload(payload)
    except ContractError as exc:
        raise PackagingError(f"dashboard payload contract is invalid: {exc}") from exc

    meta = payload.get("meta")
    if not isinstance(meta, dict) or meta.get("mode") != "live":
        mode = meta.get("mode") if isinstance(meta, dict) else None
        raise PackagingError(f"live-derived publication requires meta.mode=live (got {mode!r})")
    result_version = meta.get("result_version")
    source_licenses = LIVE_SOURCE_LICENSES_BY_RESULT_VERSION.get(result_version)
    if source_licenses is None:
        raise PackagingError(
            "live-derived publication requires a reviewed V4 or V5 result"
        )
    if (
        result_version == V5_RESULT_VERSION
        and meta.get("publication_status") != V5_PUBLICATION_STATUS
    ):
        raise PackagingError(
            "live-derived V5 publication requires publication_status=reviewed_publication"
        )
    if result_version == V5_RESULT_VERSION:
        try:
            lifecycle = validate_lifecycle_consistency(payload)
        except IntegrityError as exc:
            raise PackagingError(f"V5 lifecycle is invalid: {exc}") from exc
        if lifecycle["publication"] != V5_PUBLICATION_STATUS:
            raise PackagingError(
                "live-derived V5 lifecycle is not reviewed for publication"
            )
        _validate_reviewed_candidate_hash(payload)

    weekly = payload.get("weekly")
    if not isinstance(weekly, list) or len(weekly) < 52:
        count = len(weekly) if isinstance(weekly, list) else 0
        raise PackagingError(
            f"live-derived publication requires at least 52 weekly results (got {count})"
        )

    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise PackagingError("live-derived payload must contain sources")
    by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            raise PackagingError(f"sources[{index}] must contain an id")
        source_id = source["id"]
        if source_id in by_id:
            raise PackagingError(f"duplicate live-derived source: {source_id}")
        by_id[source_id] = source
    if set(by_id) != set(source_licenses):
        raise PackagingError(
            f"live-derived sources do not match the reviewed {result_version} contract"
        )
    for source_id, expected_license in source_licenses.items():
        actual = by_id[source_id].get("license_class")
        if actual != expected_license:
            raise PackagingError(
                f"{source_id}.license_class must be {expected_license}"
            )
    try:
        verify_provider_rights(
            (RIGHTS_PROVIDER_BY_SOURCE_ID[source_id] for source_id in by_id),
            policy_path=project_root() / "config/provider_rights.json",
            capabilities=("derived_publication",),
        )
    except (KeyError, ProviderRightsError) as exc:
        raise PackagingError(
            f"live-derived publication blocked by provider-rights policy: {exc}"
        ) from exc


def validate_public_payload(
    payload: dict[str, Any],
    *,
    publication_mode: str,
    rights_acknowledged: bool,
) -> None:
    if publication_mode == PUBLICATION_MODE_DEMO:
        validate_public_demo_payload(payload)
        return
    if publication_mode == PUBLICATION_MODE_LIVE_DERIVED:
        if rights_acknowledged is not True:
            raise PackagingError(
                "live-derived publication requires explicit personal/non-commercial rights acknowledgement"
            )
        validate_public_live_derived_payload(payload)
        return
    raise PackagingError(f"unsupported publication mode: {publication_mode!r}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_reviewed_candidate_hash(payload: dict[str, Any]) -> None:
    try:
        validate_reviewed_candidate_hash(payload)
    except IntegrityError as exc:
        raise PackagingError(
            f"V5 publication review candidate canonical hash is invalid: {exc}"
        ) from exc


def _write_public_file(root: Path, relative_path: str, value: bytes) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value)
    target.chmod(0o644)


def package_public_dashboard(
    *,
    web_root: str | Path,
    payload_path: str | Path,
    output_directory: str | Path,
    publication_mode: str = PUBLICATION_MODE_DEMO,
    rights_acknowledged: bool = False,
    comparison_path: str | Path | None = None,
    generation_manifest_path: str | Path | None = None,
    selection_family_path: str | Path | None = None,
    staged_generation_contract_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Build a new static directory from an explicit, minimal allowlist."""

    web_root = Path(web_root)
    payload_path = Path(payload_path)
    output_directory = confined_mutable_path(
        output_directory,
        project_directory=project_root(),
        label="public dashboard output",
    )

    if output_directory.exists() or output_directory.is_symlink():
        raise PackagingError(
            f"output already exists; refusing to overwrite it: {output_directory}"
        )

    files: dict[str, bytes] = {}
    for relative_path in STATIC_ALLOWLIST:
        files[relative_path] = _read_regular_file(
            web_root / relative_path,
            label=f"allowlisted web asset {relative_path}",
        )
    files["index.html"] = rewrite_index_asset_versions(
        files["index.html"],
        styles_raw=files["styles.css"],
        app_raw=files["app.js"],
    )

    payload_raw = _read_regular_file(payload_path, label="dashboard payload")
    payload = _decode_payload(payload_raw)
    validate_public_payload(
        payload,
        publication_mode=publication_mode,
        rights_acknowledged=rights_acknowledged,
    )
    files[PAYLOAD_DESTINATION] = payload_raw

    result_version = payload.get("meta", {}).get("result_version")
    if result_version == V5_RESULT_VERSION:
        resolved_comparison_path = (
            Path(comparison_path)
            if comparison_path is not None
            else payload_path.with_name(V5_COMPARISON_FILENAME)
        )
        comparison_raw = _read_regular_file(
            resolved_comparison_path,
            label="reviewed V5/V4 comparison sidecar",
        )
        comparison = _decode_v5_comparison(comparison_raw)
        validate_v5_comparison_sidecar(
            comparison,
            payload=payload,
            payload_raw=payload_raw,
        )
        manifest_binding = payload.get("meta", {}).get(
            "generation_manifest_sha256"
        )
        if manifest_binding is not None or generation_manifest_path is not None:
            resolved_manifest_path = (
                Path(generation_manifest_path)
                if generation_manifest_path is not None
                else payload_path.with_name("generation-manifest.json")
            )
            try:
                manifest_document = _decode_payload(
                    _read_regular_file(
                        resolved_manifest_path,
                        label="reviewed V5 generation manifest",
                    )
                )
            except PackagingError as exc:
                raise PackagingError(
                    f"V5 generation manifest is invalid: {exc}"
                ) from exc
            has_selection_family = (
                manifest_document.get("selection_family_sidecar") is not None
            )
            resolved_selection_family_path = (
                Path(selection_family_path)
                if selection_family_path is not None
                else payload_path.with_name(SELECTION_FAMILY_FILENAME)
            )
            contract_directory = None
            if staged_generation_contract_directory is not None:
                contract_directory = Path(staged_generation_contract_directory)
                if not contract_directory.is_absolute():
                    contract_directory = project_root() / contract_directory
            try:
                generation = validate_generation_manifest(
                    resolved_manifest_path,
                    require_comparison=True,
                    require_artifacts=False,
                    payload_path_override=(
                        payload_path if contract_directory is not None else None
                    ),
                    comparison_path_override=(
                        resolved_comparison_path
                        if contract_directory is not None
                        else None
                    ),
                    selection_family_path_override=(
                        resolved_selection_family_path
                        if contract_directory is not None and has_selection_family
                        else None
                    ),
                )
            except IntegrityError as exc:
                raise PackagingError(
                    f"V5 generation manifest is invalid: {exc}"
                ) from exc
            if contract_directory is not None:
                expected_declared = {
                    "payload": contract_directory / "regime-results.json",
                    "comparison": (
                        contract_directory / "v5-vs-v4-comparison.json"
                    ),
                    "selection": (
                        contract_directory / SELECTION_FAMILY_FILENAME
                    ),
                }
                if generation["declared_payload_path"].resolve() != (
                    expected_declared["payload"].resolve()
                ):
                    raise PackagingError(
                        "staged generation payload contract path is invalid"
                    )
                if generation["declared_comparison_path"].resolve() != (
                    expected_declared["comparison"].resolve()
                ):
                    raise PackagingError(
                        "staged generation comparison contract path is invalid"
                    )
                declared_selection = generation.get(
                    "declared_selection_family_path"
                )
                if has_selection_family and (
                    declared_selection is None
                    or declared_selection.resolve()
                    != expected_declared["selection"].resolve()
                ):
                    raise PackagingError(
                        "staged generation selection-family contract path is invalid"
                    )
            if generation["payload_path"].resolve() != payload_path.resolve():
                raise PackagingError(
                    "V5 generation manifest points to a different payload"
                )
            if generation["comparison_path"].resolve() != (
                resolved_comparison_path.resolve()
            ):
                raise PackagingError(
                    "V5 generation manifest points to a different comparison"
                )
            manifest_selection_path = generation.get("selection_family_path")
            if (
                generation.get("schema_version")
                == GENERATION_MANIFEST_SCHEMA_VERSION
                and manifest_selection_path is None
            ):
                raise PackagingError(
                    "reviewed V5 generation manifest is missing selection-family audit"
                )
            if manifest_selection_path is not None:
                if (
                    contract_directory is None
                    and manifest_selection_path.resolve()
                    != resolved_selection_family_path.resolve()
                ):
                    raise PackagingError(
                        "V5 generation manifest points to a different selection-family audit"
                    )
                selection_family_raw = _read_regular_file(
                    resolved_selection_family_path,
                    label="reviewed V5 selection-family audit",
                )
                selection_family = _decode_payload(selection_family_raw)
                try:
                    validate_selection_family_audit(
                        selection_family,
                        expected_generation_id=str(payload["meta"]["generation_id"]),
                    )
                except (TypeError, ValueError) as exc:
                    raise PackagingError(
                        f"V5 selection-family audit is invalid: {exc}"
                    ) from exc
                if generation.get("selection_family") != selection_family:
                    raise PackagingError(
                        "V5 selection-family audit differs from generation manifest"
                    )
                files[SELECTION_FAMILY_DESTINATION] = selection_family_raw
            elif selection_family_path is not None:
                raise PackagingError(
                    "selection-family path is not bound by the generation manifest"
                )
            files[GENERATION_MANIFEST_DESTINATION] = _read_regular_file(
                resolved_manifest_path,
                label="reviewed V5 generation manifest",
            )
        elif selection_family_path is not None:
            raise PackagingError(
                "selection-family path requires a bound generation manifest"
            )
        files[V5_COMPARISON_DESTINATION] = comparison_raw

    is_live_derived = publication_mode == PUBLICATION_MODE_LIVE_DERIVED
    source_ids = sorted(
        source["id"]
        for source in payload.get("sources", [])
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    )

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "package_kind": (
            "personal_noncommercial_live_derived"
            if is_live_derived
            else "synthetic_demo_only"
        ),
        "payload_mode": "live" if is_live_derived else "demo",
        "publication_scope": (
            "personal_noncommercial_derived_results"
            if is_live_derived
            else "synthetic_fixture"
        ),
        "contains_raw_observations": False,
        "source_ids": source_ids,
        "payload_data_as_of": payload["meta"].get("data_as_of"),
        "files": {
            path: {"bytes": len(value), "sha256": _sha256(value)}
            for path, value in sorted(files.items())
        },
    }
    manifest_raw = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    staging.chmod(0o755)
    try:
        for relative_path, value in files.items():
            _write_public_file(staging, relative_path, value)
        _write_public_file(staging, MANIFEST_DESTINATION, manifest_raw)
        # Read back and revalidate the exact staged package immediately before
        # the atomic directory cutover.  Any serialization or write drift fails
        # while the last-good destination remains untouched.
        staged_payload_raw = _read_regular_file(
            staging / PAYLOAD_DESTINATION,
            label="staged dashboard payload",
        )
        staged_payload = _decode_payload(staged_payload_raw)
        validate_public_payload(
            staged_payload,
            publication_mode=publication_mode,
            rights_acknowledged=rights_acknowledged,
        )
        if result_version == V5_RESULT_VERSION:
            staged_comparison_raw = _read_regular_file(
                staging / V5_COMPARISON_DESTINATION,
                label="staged V5/V4 comparison sidecar",
            )
            validate_v5_comparison_sidecar(
                _decode_v5_comparison(staged_comparison_raw),
                payload=staged_payload,
                payload_raw=staged_payload_raw,
            )
            if SELECTION_FAMILY_DESTINATION in files:
                staged_selection_family = _decode_payload(
                    _read_regular_file(
                        staging / SELECTION_FAMILY_DESTINATION,
                        label="staged V5 selection-family audit",
                    )
                )
                try:
                    validate_selection_family_audit(
                        staged_selection_family,
                        expected_generation_id=str(
                            staged_payload["meta"]["generation_id"]
                        ),
                    )
                except (TypeError, ValueError) as exc:
                    raise PackagingError(
                        f"staged V5 selection-family audit is invalid: {exc}"
                    ) from exc
        staged_manifest = _decode_payload(
            _read_regular_file(
                staging / MANIFEST_DESTINATION,
                label="staged publication manifest",
            )
        )
        if staged_manifest != manifest:
            raise PackagingError("staged publication manifest differs from memory")
        for relative_path, expected in manifest["files"].items():
            staged_raw = _read_regular_file(
                staging / relative_path,
                label=f"staged publication member {relative_path}",
            )
            if expected != {"bytes": len(staged_raw), "sha256": _sha256(staged_raw)}:
                raise PackagingError(
                    f"staged publication member hash differs: {relative_path}"
                )
        validate_index_asset_versions(
            _read_regular_file(staging / "index.html", label="staged index.html"),
            styles_raw=_read_regular_file(
                staging / "styles.css",
                label="staged styles.css",
            ),
            app_raw=_read_regular_file(
                staging / "app.js",
                label="staged app.js",
            ),
        )
        if output_directory.exists() or output_directory.is_symlink():
            raise PackagingError(
                f"output appeared during packaging; refusing overwrite: {output_directory}"
            )
        os.replace(staging, output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def package_public_demo(
    *,
    web_root: str | Path,
    payload_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Backward-compatible synthetic demo wrapper."""

    return package_public_dashboard(
        web_root=web_root,
        payload_path=payload_path,
        output_directory=output_directory,
        publication_mode=PUBLICATION_MODE_DEMO,
        rights_acknowledged=False,
        comparison_path=None,
        generation_manifest_path=None,
        selection_family_path=None,
        staged_generation_contract_directory=None,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package an allowlisted Regime dashboard for static publication"
    )
    parser.add_argument("--web-root", type=Path, default=Path("web"))
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("dist/public-dashboard")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Required final generation-manifest.json for live-derived V5",
    )
    parser.add_argument(
        "--selection-family",
        type=Path,
        help="Reviewed selection-family-audit.json bound by the generation manifest",
    )
    parser.add_argument(
        "--staged-generation-contract-directory",
        type=Path,
        help=(
            "Validate staging files against the exact final project-relative "
            "publication directory recorded in their generation manifest"
        ),
    )
    parser.add_argument(
        "--publication-mode",
        choices=(PUBLICATION_MODE_DEMO, PUBLICATION_MODE_LIVE_DERIVED),
        default=PUBLICATION_MODE_DEMO,
    )
    parser.add_argument(
        "--acknowledge-personal-noncommercial-publication",
        action="store_true",
        help="Required for live-derived publication; never expands provider rights",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=None,
        help=(
            "Required reviewed derived-only V5/V4 comparison sidecar for a V5 "
            "payload; defaults to the payload directory"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = package_public_dashboard(
            web_root=args.web_root,
            payload_path=args.payload,
            output_directory=args.output,
            publication_mode=args.publication_mode,
            rights_acknowledged=args.acknowledge_personal_noncommercial_publication,
            comparison_path=args.comparison,
            generation_manifest_path=args.manifest,
            selection_family_path=args.selection_family,
            staged_generation_contract_directory=(
                args.staged_generation_contract_directory
            ),
        )
    except (PackagingError, UnsafeMutablePath, OSError) as exc:
        print(f"public dashboard package refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "package_kind": manifest["package_kind"],
                "files": sorted(
                    [*manifest["files"], MANIFEST_DESTINATION]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
