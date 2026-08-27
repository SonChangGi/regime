"""Versioned JSON hashes and cross-file publication integrity contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from regime_lab.artifact_inventory import (
    ARTIFACT_INVENTORY_FILENAME,
    ArtifactInventoryError,
    verify_artifact_inventory,
)
from regime_lab.config import project_root
from regime_lab.runtime_fingerprint import (
    RuntimeFingerprintError,
    build_runtime_fingerprint,
    validate_runtime_fingerprint,
)


CANONICAL_JSON_SHA256_V1 = "canonical_json_sha256_v1"
LEGACY_GENERATION_MANIFEST_SCHEMA_VERSION = "regime-generation-manifest/1"
GENERATION_MANIFEST_SCHEMA_VERSION = "regime-generation-manifest/2"
GENERATION_MANIFEST_FILENAME = "generation-manifest.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SELECTION_STATUS = "selected_by_gate"
_DEPLOYMENT_STATUSES = {"candidate", "reviewed", "operating"}
_PUBLICATION_STATUSES = {"unpublished", "reviewed_publication"}


class IntegrityError(RuntimeError):
    """Raised when hashes, lifecycle state, or generation identity diverge."""


def canonical_json_bytes_v1(value: object) -> bytes:
    """Return the stable UTF-8 representation used by the v1 semantic hash."""

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise IntegrityError("value is not canonical JSON") from exc
    return serialized.encode("utf-8")


def canonical_json_sha256_v1(value: object) -> str:
    """Hash JSON semantics independently of whitespace and object key order."""

    return hashlib.sha256(canonical_json_bytes_v1(value)).hexdigest()


def _object(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{context} must be an object")
    return value


def _exact_object(
    value: object,
    *,
    context: str,
    fields: set[str],
) -> Mapping[str, Any]:
    document = _object(value, context=context)
    if set(document) != fields:
        raise IntegrityError(f"{context} fields must be exactly {sorted(fields)}")
    return document


def _sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IntegrityError(f"{context} must be a lowercase SHA-256")
    return value


def _nonempty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError(f"{context} must be a non-empty string")
    return value


def payload_without_generation_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a payload and remove only the manifest back-reference.

    ``payload_contract_sha256`` is defined over this value.  Removing exactly
    this field breaks the payload/manifest hash cycle without weakening any
    other payload field.
    """

    candidate = deepcopy(dict(payload))
    meta = _object(candidate.get("meta"), context="payload.meta")
    mutable_meta = dict(meta)
    mutable_meta.pop("generation_manifest_sha256", None)
    candidate["meta"] = mutable_meta
    return candidate


def canonical_json_sha256_v1_without_generation_binding(
    payload: Mapping[str, Any],
) -> str:
    """Hash a payload contract without its manifest back-reference."""

    return canonical_json_sha256_v1(payload_without_generation_binding(payload))


def reviewed_candidate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic reviewed-but-unpublished candidate identity."""

    candidate = payload_without_generation_binding(payload)
    meta = dict(_object(candidate.get("meta"), context="payload.meta"))
    model = dict(_object(candidate.get("model"), context="payload.model"))
    lifecycle = dict(
        _object(model.get("lifecycle"), context="payload.model.lifecycle")
    )
    lifecycle["selection"] = {"status": _SELECTION_STATUS}
    lifecycle["deployment"] = {"status": "reviewed"}
    lifecycle["publication"] = {"status": "unpublished"}
    model["lifecycle"] = lifecycle
    model["selection_status"] = _SELECTION_STATUS
    meta["publication_status"] = "unpublished"
    meta.pop("publication_review", None)
    candidate["meta"] = meta
    candidate["model"] = model
    validate_lifecycle_consistency(candidate)
    return candidate


def reviewed_candidate_sha256_v1(payload: Mapping[str, Any]) -> str:
    """Hash the normalized pre-publication review identity."""

    return canonical_json_sha256_v1(reviewed_candidate_payload(payload))


def validate_reviewed_candidate_hash(payload: Mapping[str, Any]) -> str:
    """Validate the reviewed candidate identity embedded in a publication."""

    meta = _object(payload.get("meta"), context="payload.meta")
    review = _object(
        meta.get("publication_review"),
        context="payload.meta.publication_review",
    )
    expected = _sha256(
        review.get("reviewed_candidate_sha256"),
        context="payload.meta.publication_review.reviewed_candidate_sha256",
    )
    actual = reviewed_candidate_sha256_v1(payload)
    if actual != expected:
        raise IntegrityError("reviewed candidate canonical JSON hash mismatch")
    return actual


def comparison_without_payload_binding(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only the sidecar's raw-payload hash to avoid a second cycle."""

    candidate = deepcopy(dict(comparison))
    try:
        inputs = _object(candidate["inputs"], context="comparison.inputs")
        v5 = _object(inputs["v5"], context="comparison.inputs.v5")
        result = _object(
            v5["regime_results"],
            context="comparison.inputs.v5.regime_results",
        )
    except KeyError as exc:
        raise IntegrityError(
            "comparison payload binding is missing"
        ) from exc
    mutable_inputs = dict(inputs)
    mutable_v5 = dict(v5)
    mutable_result = dict(result)
    mutable_result.pop("sha256", None)
    mutable_v5["regime_results"] = mutable_result
    mutable_inputs["v5"] = mutable_v5
    candidate["inputs"] = mutable_inputs
    return candidate


def canonical_comparison_contract_sha256_v1(
    comparison: Mapping[str, Any],
) -> str:
    """Hash the sidecar while excluding only its raw-payload back-reference."""

    return canonical_json_sha256_v1(comparison_without_payload_binding(comparison))


def validate_lifecycle_consistency(payload: Mapping[str, Any]) -> dict[str, str]:
    """Validate the single allowed selection/deployment/publication state machine."""

    meta = _object(payload.get("meta"), context="payload.meta")
    model = _object(payload.get("model"), context="payload.model")
    lifecycle = _exact_object(
        model.get("lifecycle"),
        context="payload.model.lifecycle",
        fields={"selection", "deployment", "publication"},
    )
    selection = _exact_object(
        lifecycle.get("selection"),
        context="payload.model.lifecycle.selection",
        fields={"status"},
    )
    deployment = _exact_object(
        lifecycle.get("deployment"),
        context="payload.model.lifecycle.deployment",
        fields={"status"},
    )
    publication = _exact_object(
        lifecycle.get("publication"),
        context="payload.model.lifecycle.publication",
        fields={"status"},
    )

    selection_status = selection.get("status")
    deployment_status = deployment.get("status")
    publication_status = publication.get("status")
    if selection_status != _SELECTION_STATUS:
        raise IntegrityError(
            "payload.model.lifecycle.selection.status must be selected_by_gate"
        )
    if model.get("selection_status") != selection_status:
        raise IntegrityError(
            "payload.model.selection_status must alias lifecycle.selection.status"
        )
    if deployment_status not in _DEPLOYMENT_STATUSES:
        raise IntegrityError("payload.model.lifecycle.deployment.status is invalid")
    if publication_status not in _PUBLICATION_STATUSES:
        raise IntegrityError("payload.model.lifecycle.publication.status is invalid")
    if meta.get("publication_status") != publication_status:
        raise IntegrityError(
            "payload.meta.publication_status must alias lifecycle.publication.status"
        )

    allowed = (
        publication_status == "unpublished"
        and deployment_status in {"candidate", "reviewed"}
    ) or (
        publication_status == "reviewed_publication"
        and deployment_status == "operating"
    )
    if not allowed:
        raise IntegrityError(
            "lifecycle combination is invalid: only unpublished+candidate/reviewed "
            "or reviewed_publication+operating is allowed"
        )
    return {
        "selection": str(selection_status),
        "deployment": str(deployment_status),
        "publication": str(publication_status),
    }


def _read_json_object(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"{context} must be a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"{context} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"{context} must be a JSON object")
    return value, raw


def _resolve_project_target(value: object, *, context: str) -> Path:
    text = _nonempty_string(value, context=context)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or text != pure.as_posix():
        raise IntegrityError(f"{context} must be a confined project-relative path")
    base = project_root().resolve()
    target = (base / Path(*pure.parts)).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise IntegrityError(f"{context} escapes the project directory") from exc
    if target.is_symlink():
        raise IntegrityError(f"{context} must not resolve to a symbolic link")
    return target


def _resolve_project_member(value: object, *, context: str) -> Path:
    target = _resolve_project_target(value, context=context)
    if not target.is_file():
        raise IntegrityError(f"{context} must resolve to a regular project file")
    return target


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_relative_regular(path: Path, *, context: str) -> str:
    base = project_root().resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError as exc:
        raise IntegrityError(f"{context} must be inside the project directory") from exc
    if path.is_symlink() or not resolved.is_file():
        raise IntegrityError(f"{context} must be a regular file")
    return relative.as_posix()


def _project_relative_target(path: Path, *, context: str) -> str:
    """Resolve a confined logical output path that may not exist yet."""

    base = project_root().resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError as exc:
        raise IntegrityError(f"{context} must be inside the project directory") from exc
    if path.is_symlink():
        raise IntegrityError(f"{context} must not be a symlink")
    return relative.as_posix()


def build_generation_manifest(
    *,
    payload: Mapping[str, Any],
    payload_path: str | Path,
    artifact_directory: str | Path,
    input_snapshot: Mapping[str, Any],
    label_spec_path: str | Path,
    comparison: Mapping[str, Any] | None = None,
    comparison_path: str | Path | None = None,
    selection_family: Mapping[str, Any] | None = None,
    selection_family_path: str | Path | None = None,
    payload_contract_path: str | Path | None = None,
    comparison_contract_path: str | Path | None = None,
    selection_family_contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the canonical manifest body before adding its payload back-reference.

    The comparison may still contain a placeholder raw-payload hash because its
    contract digest intentionally excludes that one field.  Callers can bind
    the returned manifest into the payload, serialize the payload, then replace
    the sidecar placeholder with the final payload byte hash.
    """

    meta = _object(payload.get("meta"), context="payload.meta")
    model = _object(payload.get("model"), context="payload.model")
    generation_id = _nonempty_string(
        meta.get("generation_id"),
        context="payload.meta.generation_id",
    )
    validate_lifecycle_consistency(payload)
    payload_file = Path(payload_path)
    payload_relative = _project_relative_target(
        Path(payload_contract_path or payload_file),
        context="generation payload",
    )
    artifacts = Path(artifact_directory)
    inventory_path = artifacts / ARTIFACT_INVENTORY_FILENAME
    try:
        inventory_summary = verify_artifact_inventory(artifacts)
    except (ArtifactInventoryError, OSError) as exc:
        raise IntegrityError(f"generation artifact inventory is invalid: {exc}") from exc
    generation_file, _ = _read_json_object(
        artifacts / "build-generation.json",
        context="artifact build-generation",
    )
    if generation_file != {"generation_id": generation_id}:
        raise IntegrityError("artifact and payload generation_id mismatch")

    snapshot = _exact_object(
        input_snapshot,
        context="input snapshot",
        fields={"data_as_of", "sha256"},
    )
    if snapshot.get("data_as_of") != meta.get("data_as_of"):
        raise IntegrityError("input snapshot data_as_of differs from payload")
    _sha256(snapshot.get("sha256"), context="input snapshot.sha256")

    label_path = Path(label_spec_path)
    label_relative = _project_relative_regular(label_path, context="label spec")
    label_contract = _object(payload.get("label"), context="payload.label")
    label_spec_id = _nonempty_string(
        label_contract.get("spec_id"),
        context="payload.label.spec_id",
    )
    label_version = _nonempty_string(
        label_contract.get("spec_version"),
        context="payload.label.spec_version",
    )
    if model.get("label_version") != label_version:
        raise IntegrityError("payload label version differs from model")
    label_spec_sha256 = _sha256(
        label_contract.get("spec_sha256"),
        context="payload.label.spec_sha256",
    )
    execution_parameters = _object(
        model.get("execution_parameters"),
        context="payload.model.execution_parameters",
    )
    execution_sha256 = _sha256(
        execution_parameters.get("sha256"),
        context="payload.model.execution_parameters.sha256",
    )

    comparison_record: dict[str, str] | None = None
    if comparison is None and comparison_path is not None:
        raise IntegrityError("comparison_path requires a comparison document")
    if comparison is None and comparison_contract_path is not None:
        raise IntegrityError(
            "comparison_contract_path requires a comparison document"
        )
    if comparison is not None:
        if comparison_path is None:
            raise IntegrityError("comparison document requires comparison_path")
        comparison_relative = _project_relative_target(
            Path(comparison_contract_path or comparison_path),
            context="generation comparison sidecar",
        )
        comparison_record = {
            "path": comparison_relative,
            "comparison_contract_sha256": (
                canonical_comparison_contract_sha256_v1(comparison)
            ),
        }

    selection_family_record: dict[str, str] | None = None
    if selection_family is None and selection_family_path is not None:
        raise IntegrityError(
            "selection_family_path requires a selection-family document"
        )
    if selection_family is None and selection_family_contract_path is not None:
        raise IntegrityError(
            "selection_family_contract_path requires a selection-family document"
        )
    if selection_family is not None:
        if selection_family_path is None:
            raise IntegrityError(
                "selection-family document requires selection_family_path"
            )
        # Import lazily: selection_family_audit uses canonical hashing from this
        # module and must remain import-cycle free.
        from regime_lab.selection_family_audit import (
            validate_selection_family_payload_binding,
        )

        try:
            validate_selection_family_payload_binding(
                selection_family,
                payload,
            )
        except (TypeError, ValueError) as exc:
            raise IntegrityError(
                f"selection-family sidecar contract is invalid: {exc}"
            ) from exc
        if (
            selection_family.get("candidate_manifest_sha256")
            != model.get("candidate_manifest_sha256")
        ):
            raise IntegrityError(
                "selection-family candidate manifest differs from payload"
            )
        selection_family_relative = _project_relative_target(
            Path(selection_family_contract_path or selection_family_path),
            context="generation selection-family sidecar",
        )
        selection_family_record = {
            "path": selection_family_relative,
            "selection_family_contract_sha256": canonical_json_sha256_v1(
                selection_family
            ),
        }

    return {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "generation_id": generation_id,
        "payload": {
            "path": payload_relative,
            "payload_contract_sha256": (
                canonical_json_sha256_v1_without_generation_binding(payload)
            ),
        },
        "comparison_sidecar": comparison_record,
        "selection_family_sidecar": selection_family_record,
        "artifact_inventory": {
            "sha256": _raw_sha256(inventory_path),
            "file_count": inventory_summary["file_count"],
        },
        "input_snapshot": dict(snapshot),
        "label_spec": {
            "path": label_relative,
            "registry_sha256": _raw_sha256(label_path),
            "spec_id": label_spec_id,
            "version": label_version,
            "spec_sha256": label_spec_sha256,
        },
        "execution_spec": {"sha256": execution_sha256},
        "runtime_fingerprint": build_runtime_fingerprint(project_root()),
    }


def bind_payload_to_generation_manifest(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy carrying the canonical manifest hash back-reference."""

    bound = deepcopy(dict(payload))
    meta = dict(_object(bound.get("meta"), context="payload.meta"))
    meta["generation_manifest_sha256"] = canonical_json_sha256_v1(manifest)
    bound["meta"] = meta
    return bound


def validate_generation_manifest(
    manifest_path: str | Path,
    *,
    require_comparison: bool | None = None,
    require_selection_family: bool | None = None,
    require_artifacts: bool = True,
    artifact_directory: str | Path | None = None,
    payload_path_override: str | Path | None = None,
    comparison_path_override: str | Path | None = None,
    selection_family_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one complete generation and return its resolved members.

    The manifest is semantic JSON.  ``payload_contract_sha256`` excludes only
    ``meta.generation_manifest_sha256``.  The final payload must carry the
    canonical manifest hash.  The sidecar contract hash excludes only its
    payload-byte hash; that raw hash is checked separately against the final
    payload bytes.  These two narrowly scoped exclusions make the graph
    acyclic while keeping every final edge independently verifiable.  The
    artifact inventory digest/count is always bound; its private members are
    required locally and opportunistically verified in publication checkouts.
    """

    path = Path(manifest_path)
    manifest, _ = _read_json_object(path, context="generation manifest")
    legacy_fields = {
        "schema_version",
        "generation_id",
        "payload",
        "comparison_sidecar",
        "artifact_inventory",
        "input_snapshot",
        "label_spec",
        "execution_spec",
    }
    current_fields = {*legacy_fields, "selection_family_sidecar"}
    current_runtime_fields = {*current_fields, "runtime_fingerprint"}
    schema_version = manifest.get("schema_version")
    allowed_fields = (
        (legacy_fields,)
        if schema_version == LEGACY_GENERATION_MANIFEST_SCHEMA_VERSION
        else (current_fields, current_runtime_fields)
    )
    if set(manifest) not in allowed_fields:
        raise IntegrityError(
            "generation manifest fields do not match a supported schema: "
            f"{sorted(manifest)}"
        )
    if schema_version not in {
        LEGACY_GENERATION_MANIFEST_SCHEMA_VERSION,
        GENERATION_MANIFEST_SCHEMA_VERSION,
    }:
        raise IntegrityError("generation manifest schema_version is invalid")
    generation_id = _nonempty_string(
        manifest.get("generation_id"),
        context="generation manifest.generation_id",
    )
    payload_record = _exact_object(
        manifest.get("payload"),
        context="generation manifest.payload",
        fields={"path", "payload_contract_sha256"},
    )
    declared_payload_path = (
        _resolve_project_target
        if payload_path_override is not None
        else _resolve_project_member
    )(
        payload_record.get("path"),
        context="generation manifest.payload.path",
    )
    payload_path = (
        Path(payload_path_override)
        if payload_path_override is not None
        else declared_payload_path
    )
    payload, payload_raw = _read_json_object(
        payload_path,
        context="generation payload",
    )
    payload_contract_sha256 = _sha256(
        payload_record.get("payload_contract_sha256"),
        context="generation manifest.payload.payload_contract_sha256",
    )
    if (
        canonical_json_sha256_v1_without_generation_binding(payload)
        != payload_contract_sha256
    ):
        raise IntegrityError("generation payload contract hash mismatch")
    meta = _object(payload.get("meta"), context="generation payload.meta")
    if meta.get("generation_id") != generation_id:
        raise IntegrityError("generation payload generation_id mismatch")
    manifest_sha256 = canonical_json_sha256_v1(manifest)
    if meta.get("generation_manifest_sha256") != manifest_sha256:
        raise IntegrityError("generation payload manifest hash back-reference mismatch")

    comparison_record = manifest.get("comparison_sidecar")
    if require_comparison is True and comparison_record is None:
        raise IntegrityError("generation comparison sidecar is required")
    if require_comparison is False and comparison_record is not None:
        raise IntegrityError("generation comparison sidecar is not allowed")
    if comparison_record is None and comparison_path_override is not None:
        raise IntegrityError(
            "comparison path override requires a manifest comparison sidecar"
        )
    comparison_path: Path | None = None
    declared_comparison_path: Path | None = None
    comparison: dict[str, Any] | None = None
    if comparison_record is not None:
        record = _exact_object(
            comparison_record,
            context="generation manifest.comparison_sidecar",
            fields={"path", "comparison_contract_sha256"},
        )
        declared_comparison_path = (
            _resolve_project_target
            if comparison_path_override is not None
            else _resolve_project_member
        )(
            record.get("path"),
            context="generation manifest.comparison_sidecar.path",
        )
        comparison_path = (
            Path(comparison_path_override)
            if comparison_path_override is not None
            else declared_comparison_path
        )
        comparison, _ = _read_json_object(
            comparison_path,
            context="generation comparison sidecar",
        )
        expected_comparison = _sha256(
            record.get("comparison_contract_sha256"),
            context=(
                "generation manifest.comparison_sidecar."
                "comparison_contract_sha256"
            ),
        )
        if (
            canonical_comparison_contract_sha256_v1(comparison)
            != expected_comparison
        ):
            raise IntegrityError("generation comparison sidecar contract hash mismatch")
        try:
            raw_payload_binding = comparison["inputs"]["v5"]["regime_results"][
                "sha256"
            ]
        except (KeyError, TypeError) as exc:
            raise IntegrityError(
                "generation comparison payload binding is missing"
            ) from exc
        if raw_payload_binding != hashlib.sha256(payload_raw).hexdigest():
            raise IntegrityError(
                "generation comparison is bound to different payload bytes"
            )

    selection_family_record = manifest.get("selection_family_sidecar")
    if require_selection_family is True and selection_family_record is None:
        raise IntegrityError("generation selection-family sidecar is required")
    if require_selection_family is False and selection_family_record is not None:
        raise IntegrityError("generation selection-family sidecar is not allowed")
    if (
        selection_family_record is None
        and selection_family_path_override is not None
    ):
        raise IntegrityError(
            "selection-family path override requires a manifest sidecar"
        )
    selection_family_path: Path | None = None
    declared_selection_family_path: Path | None = None
    selection_family: dict[str, Any] | None = None
    if selection_family_record is not None:
        record = _exact_object(
            selection_family_record,
            context="generation manifest.selection_family_sidecar",
            fields={"path", "selection_family_contract_sha256"},
        )
        declared_selection_family_path = (
            _resolve_project_target
            if selection_family_path_override is not None
            else _resolve_project_member
        )(
            record.get("path"),
            context="generation manifest.selection_family_sidecar.path",
        )
        selection_family_path = (
            Path(selection_family_path_override)
            if selection_family_path_override is not None
            else declared_selection_family_path
        )
        selection_family, _ = _read_json_object(
            selection_family_path,
            context="generation selection-family sidecar",
        )
        expected_selection_family = _sha256(
            record.get("selection_family_contract_sha256"),
            context=(
                "generation manifest.selection_family_sidecar."
                "selection_family_contract_sha256"
            ),
        )
        if canonical_json_sha256_v1(selection_family) != expected_selection_family:
            raise IntegrityError(
                "generation selection-family sidecar contract hash mismatch"
            )
        from regime_lab.selection_family_audit import (
            validate_selection_family_payload_binding,
        )

        try:
            validate_selection_family_payload_binding(
                selection_family,
                payload,
            )
        except (TypeError, ValueError) as exc:
            raise IntegrityError(
                f"generation selection-family sidecar is invalid: {exc}"
            ) from exc

    inventory_record = _exact_object(
        manifest.get("artifact_inventory"),
        context="generation manifest.artifact_inventory",
        fields={"file_count", "sha256"},
    )
    expected_inventory_sha256 = _sha256(
        inventory_record.get("sha256"),
        context="generation manifest.artifact_inventory.sha256",
    )
    expected_file_count = inventory_record.get("file_count")
    if type(expected_file_count) is not int or expected_file_count < 1:
        raise IntegrityError(
            "generation manifest.artifact_inventory.file_count must be positive"
        )
    artifacts = (
        Path(artifact_directory)
        if artifact_directory is not None
        else path.parent / "artifacts"
    )
    artifacts_available = artifacts.exists() or artifacts.is_symlink()
    if require_artifacts and not artifacts_available:
        raise IntegrityError("generation artifact directory is unavailable")
    if artifacts_available:
        inventory_path = artifacts / ARTIFACT_INVENTORY_FILENAME
        if inventory_path.is_symlink() or not inventory_path.is_file():
            raise IntegrityError("generation artifact inventory is missing/non-regular")
        if _raw_sha256(inventory_path) != expected_inventory_sha256:
            raise IntegrityError("generation artifact inventory hash mismatch")
        try:
            inventory_summary = verify_artifact_inventory(artifacts)
        except (ArtifactInventoryError, OSError) as exc:
            raise IntegrityError(
                f"generation artifact inventory is invalid: {exc}"
            ) from exc
        if inventory_summary["file_count"] != expected_file_count:
            raise IntegrityError("generation artifact inventory file count mismatch")
        generation_file, _ = _read_json_object(
            artifacts / "build-generation.json",
            context="artifact build-generation",
        )
        if generation_file != {"generation_id": generation_id}:
            raise IntegrityError("artifact and manifest generation_id mismatch")
        inventory_summary = {**inventory_summary, "verified": True}
        if selection_family is not None:
            from regime_lab.selection_family_audit import (
                build_selection_family_audit_from_artifacts,
            )

            try:
                rebuilt_selection_family = (
                    build_selection_family_audit_from_artifacts(payload, artifacts)
                )
            except (OSError, TypeError, ValueError) as exc:
                raise IntegrityError(
                    f"generation selection-family source evidence is invalid: {exc}"
                ) from exc
            if rebuilt_selection_family != selection_family:
                raise IntegrityError(
                    "generation selection-family sidecar differs from source evidence"
                )
    else:
        inventory_summary = {
            "path": ARTIFACT_INVENTORY_FILENAME,
            "file_count": expected_file_count,
            "sha256": expected_inventory_sha256,
            "verified": False,
        }

    input_snapshot = _exact_object(
        manifest.get("input_snapshot"),
        context="generation manifest.input_snapshot",
        fields={"data_as_of", "sha256"},
    )
    if input_snapshot.get("data_as_of") != meta.get("data_as_of"):
        raise IntegrityError("input snapshot data_as_of differs from payload")
    _sha256(
        input_snapshot.get("sha256"),
        context="generation manifest.input_snapshot.sha256",
    )

    model = _object(payload.get("model"), context="generation payload.model")
    label_spec = _exact_object(
        manifest.get("label_spec"),
        context="generation manifest.label_spec",
        fields={
            "path",
            "registry_sha256",
            "spec_id",
            "version",
            "spec_sha256",
        },
    )
    label_contract = _object(
        payload.get("label"),
        context="generation payload.label",
    )
    if (
        label_spec.get("spec_id") != label_contract.get("spec_id")
        or label_spec.get("version") != label_contract.get("spec_version")
        or label_spec.get("version") != model.get("label_version")
    ):
        raise IntegrityError("label spec version differs from payload")
    label_spec_sha256 = _sha256(
        label_spec.get("spec_sha256"),
        context="generation manifest.label_spec.spec_sha256",
    )
    if label_contract.get("spec_sha256") != label_spec_sha256:
        raise IntegrityError("label spec hash differs from payload")
    label_registry_sha256 = _sha256(
        label_spec.get("registry_sha256"),
        context="generation manifest.label_spec.registry_sha256",
    )
    label_spec_path = _resolve_project_member(
        label_spec.get("path"),
        context="generation manifest.label_spec.path",
    )
    label_registry_verified = _raw_sha256(label_spec_path) == label_registry_sha256
    if (require_artifacts or artifacts_available) and not label_registry_verified:
        raise IntegrityError("label registry hash differs from the project source")

    execution_spec = _exact_object(
        manifest.get("execution_spec"),
        context="generation manifest.execution_spec",
        fields={"sha256"},
    )
    execution_parameters = _object(
        model.get("execution_parameters"),
        context="generation payload.model.execution_parameters",
    )
    execution_sha256 = _sha256(
        execution_spec.get("sha256"),
        context="generation manifest.execution_spec.sha256",
    )
    if execution_parameters.get("sha256") != execution_sha256:
        raise IntegrityError("execution spec hash differs from payload")

    runtime_fingerprint: dict[str, Any] | None = None
    raw_runtime_fingerprint = manifest.get("runtime_fingerprint")
    if raw_runtime_fingerprint is not None:
        runtime_fingerprint = dict(
            _object(
                raw_runtime_fingerprint,
                context="generation manifest.runtime_fingerprint",
            )
        )
        try:
            validate_runtime_fingerprint(runtime_fingerprint)
        except RuntimeFingerprintError as exc:
            raise IntegrityError(
                f"generation runtime fingerprint is invalid: {exc}"
            ) from exc

    lifecycle = validate_lifecycle_consistency(payload)
    return {
        "ok": True,
        "schema_version": str(schema_version),
        "generation_id": generation_id,
        "manifest_sha256": manifest_sha256,
        "payload_contract_sha256": payload_contract_sha256,
        "payload_path": payload_path,
        "declared_payload_path": declared_payload_path,
        "payload": payload,
        "payload_raw": payload_raw,
        "comparison_path": comparison_path,
        "declared_comparison_path": declared_comparison_path,
        "comparison": comparison,
        "selection_family_path": selection_family_path,
        "declared_selection_family_path": declared_selection_family_path,
        "selection_family": selection_family,
        "artifact_directory": artifacts,
        "artifact_inventory": inventory_summary,
        "input_snapshot": dict(input_snapshot),
        "label_spec": {
            **dict(label_spec),
            "registry_verified": label_registry_verified,
        },
        "execution_spec": dict(execution_spec),
        "runtime_fingerprint": runtime_fingerprint,
        "lifecycle": lifecycle,
    }


__all__ = [
    "CANONICAL_JSON_SHA256_V1",
    "GENERATION_MANIFEST_FILENAME",
    "GENERATION_MANIFEST_SCHEMA_VERSION",
    "LEGACY_GENERATION_MANIFEST_SCHEMA_VERSION",
    "IntegrityError",
    "bind_payload_to_generation_manifest",
    "build_generation_manifest",
    "canonical_comparison_contract_sha256_v1",
    "canonical_json_bytes_v1",
    "canonical_json_sha256_v1",
    "canonical_json_sha256_v1_without_generation_binding",
    "comparison_without_payload_binding",
    "payload_without_generation_binding",
    "reviewed_candidate_payload",
    "reviewed_candidate_sha256_v1",
    "validate_generation_manifest",
    "validate_lifecycle_consistency",
    "validate_reviewed_candidate_hash",
]
