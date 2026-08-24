from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REVIEWED_NODE24_ACTIONS = {
    "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1": "v7.0.1",
    "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97": "v7.0.0",
    "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020": "v7.0.0",
    "actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d": "v6.0.0",
    "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128": "v5.0.0",
}
REVIEWED_NODE24_COMPOSITE_ACTIONS = {
    # v5.0.0 pins actions/upload-artifact v7.0.0, whose action runtime is node24.
    "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9": (
        "v5.0.0"
    ),
}


def test_ci_lock_hashes_every_exact_requirement_and_build_tool() -> None:
    lines = (ROOT / "requirements-ci.lock").read_text(encoding="utf-8").splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(r"[a-z0-9_.-]+==[^ \\]+ \\", line)
    ]

    assert starts
    names = {lines[index].split("==", 1)[0] for index in starts}
    assert {"pip", "setuptools", "wheel"}.issubset(names)
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        assert any("--hash=sha256:" in line for line in lines[start + 1 : end])


def test_workflows_pin_runner_runtimes_actions_and_hash_install() -> None:
    reviewed_actions = {
        **REVIEWED_NODE24_ACTIONS,
        **REVIEWED_NODE24_COMPOSITE_ACTIONS,
    }
    for relative in (".github/workflows/ci.yml", ".github/workflows/pages.yml"):
        workflow = (ROOT / relative).read_text(encoding="utf-8")
        assert "ubuntu-latest" not in workflow
        assert "runs-on: ubuntu-24.04" in workflow
        assert 'python-version: "3.13.13"' in workflow
        assert 'node-version: "24.7.0"' in workflow
        assert "pip install --upgrade" not in workflow
        assert "pip install --require-hashes -r requirements-ci.lock" in workflow
        assert "pip install --no-build-isolation --no-deps -e ." in workflow
        uses = re.findall(r"^\s*uses:\s*([^\s#]+)", workflow, flags=re.MULTILINE)
        assert uses
        assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", item) for item in uses)
        assert set(uses).issubset(reviewed_actions)
        for action in uses:
            assert f"uses: {action} # {reviewed_actions[action]}" in workflow

    workflows = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (".github/workflows/ci.yml", ".github/workflows/pages.yml")
    )
    for action, release in reviewed_actions.items():
        assert f"uses: {action} # {release}" in workflows
