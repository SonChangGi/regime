"""Machine-enforced provider-rights policy for live Regime research.

User acknowledgement records intent; it cannot grant a permission that the
provider has not granted.  Live collection and model fitting therefore require
an affirmative, current policy entry for every provider used by the config.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


UTC = timezone.utc
LIVE_CAPABILITIES = (
    "collection",
    "local_storage",
    "model_training",
    "derived_publication",
)
_PROVIDER_ALIASES = {
    "alfred": "fred_alfred",
    "alpha_vantage": "alpha_vantage",
}


class ProviderRightsError(RuntimeError):
    """Raised before transport or training when rights are not affirmative."""


def providers_for_live_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return rights-policy provider ids required by a live series config."""

    declared = config.get("provider_rights_providers", ())
    if not isinstance(declared, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in declared
    ):
        raise ProviderRightsError(
            "provider_rights_providers must be a list of non-empty ids"
        )
    inferred = [
        policy_id
        for config_key, policy_id in _PROVIDER_ALIASES.items()
        if isinstance(config.get(config_key), Mapping)
    ]
    return tuple(dict.fromkeys([*inferred, *(item.strip() for item in declared)]))


def _aware_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ProviderRightsError(f"provider rights {label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProviderRightsError(f"provider rights {label} is invalid") from exc
    if parsed.tzinfo is None:
        raise ProviderRightsError(f"provider rights {label} must include timezone")
    return parsed.astimezone(UTC)


def load_provider_rights(path: str | Path) -> dict[str, Any]:
    selected = Path(path)
    try:
        document = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderRightsError(
            f"provider rights policy is unavailable: {selected}"
        ) from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ProviderRightsError("provider rights policy schema is invalid")
    if not isinstance(document.get("providers"), dict):
        raise ProviderRightsError("provider rights policy has no providers")
    return document


def verify_provider_rights(
    provider_ids: Iterable[str],
    *,
    policy_path: str | Path,
    now: datetime | None = None,
    capabilities: Iterable[str] = LIVE_CAPABILITIES,
) -> None:
    """Require current affirmative permission for each requested capability."""

    requested = tuple(dict.fromkeys(str(item) for item in provider_ids))
    if not requested:
        return
    required_capabilities = tuple(dict.fromkeys(str(item) for item in capabilities))
    unknown_capabilities = set(required_capabilities) - set(LIVE_CAPABILITIES)
    if unknown_capabilities:
        raise ValueError(
            "unknown provider-rights capabilities: "
            + ", ".join(sorted(unknown_capabilities))
        )

    document = load_provider_rights(policy_path)
    providers = document["providers"]
    current = (now or datetime.now(UTC)).astimezone(UTC)
    failures: list[str] = []

    for provider_id in requested:
        entry = providers.get(provider_id)
        if not isinstance(entry, dict):
            failures.append(f"{provider_id}: policy entry missing")
            continue
        status = entry.get("status")
        reason = str(entry.get("reason_code") or status or "not_allowed")
        if status != "allowed":
            failures.append(f"{provider_id}: {reason}")
            continue
        try:
            review_after = _aware_timestamp(
                entry.get("review_after"),
                label=f"{provider_id}.review_after",
            )
        except ProviderRightsError as exc:
            failures.append(str(exc))
            continue
        if review_after <= current:
            failures.append(f"{provider_id}: rights review expired")
            continue
        missing = [
            capability
            for capability in required_capabilities
            if entry.get("capabilities", {}).get(capability) is not True
        ]
        if missing:
            failures.append(
                f"{provider_id}: permission missing for {', '.join(missing)}"
            )
            continue
        evidence = entry.get("evidence")
        if (
            not isinstance(evidence, dict)
            or not evidence.get("source_url")
            or not isinstance(evidence.get("basis"), str)
            or not str(evidence["basis"]).strip()
        ):
            failures.append(f"{provider_id}: permission evidence missing")
            continue
        if evidence.get("basis") == "user_attested_direct_provider_approval":
            reference = evidence.get("approval_reference")
            conditions = entry.get("conditions")
            try:
                attested_at = _aware_timestamp(
                    evidence.get("attested_at"),
                    label=f"{provider_id}.evidence.attested_at",
                )
            except ProviderRightsError as exc:
                failures.append(str(exc))
                continue
            if attested_at > current:
                failures.append(f"{provider_id}: approval attestation is in the future")
            elif not isinstance(reference, str) or not reference.strip():
                failures.append(f"{provider_id}: approval reference missing")
            elif not isinstance(conditions, dict) or conditions.get("project") != "regime":
                failures.append(f"{provider_id}: project approval scope missing")
            elif "derived_publication" in required_capabilities:
                scope_failures: list[str] = []
                if evidence.get("approval_scope") != (
                    "personal_noncommercial_derived_results_only"
                ):
                    scope_failures.append("approval scope")
                if conditions.get("raw_publication") is not False:
                    scope_failures.append("raw-publication prohibition")
                if conditions.get("derived_outputs_only") is not True:
                    scope_failures.append("derived-output restriction")
                if conditions.get("personal_noncommercial_only") is not True:
                    scope_failures.append("personal/noncommercial restriction")
                if conditions.get("commercial_publication") is not False:
                    scope_failures.append("commercial-publication prohibition")
                if conditions.get("derived_publication_scope_document_pending") is not False:
                    scope_failures.append("scope confirmation")
                if scope_failures:
                    failures.append(
                        f"{provider_id}: derived-publication scope incomplete: "
                        + ", ".join(scope_failures)
                    )

    if failures:
        raise ProviderRightsError(
            "live provider-rights gate blocked execution: " + "; ".join(failures)
        )


__all__ = [
    "LIVE_CAPABILITIES",
    "ProviderRightsError",
    "load_provider_rights",
    "providers_for_live_config",
    "verify_provider_rights",
]
