"""Confinement checks for paths that local commands may create or replace."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


class UnsafeMutablePath(ValueError):
    """Raised before a write target can escape an approved local root."""


def _canonical_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise UnsafeMutablePath(f"approved mutable root is not a directory: {root}")
    return root


def _existing_symlink_component(path: Path, *, root: Path) -> Path | None:
    """Return the first symlink at or below ``root`` in the lexical path."""

    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    current = root
    for part in relative.parts:
        current = current / part
        # is_symlink also detects a dangling destination symlink.
        if current.is_symlink():
            return current
        if not current.exists():
            # Descendants cannot exist once an ordinary component is absent.
            break
    return None


def confined_mutable_path(
    value: str | Path,
    *,
    project_directory: str | Path,
    label: str = "write target",
    temporary_directory: str | Path | None = None,
) -> Path:
    """Resolve one mutable target under the project or the process temp root.

    Relative paths are project-relative.  Absolute paths are accepted only
    beneath the canonical project directory or ``tempfile.gettempdir()`` (the
    latter keeps isolated tests and local preview builds usable).  Existing or
    dangling symlinks at the destination or in its in-root parent chain are
    rejected, even when they resolve back inside an allowed root.
    """

    project_root = _canonical_root(project_directory)
    temp_root = _canonical_root(
        tempfile.gettempdir() if temporary_directory is None else temporary_directory
    )
    raw = Path(value).expanduser()
    lexical = raw if raw.is_absolute() else project_root / raw
    lexical = Path(os.path.abspath(lexical))
    resolved = lexical.resolve(strict=False)

    allowed_root: Path | None = None
    for root in (project_root, temp_root):
        if resolved != root and resolved.is_relative_to(root):
            allowed_root = root
            break
    if allowed_root is None:
        raise UnsafeMutablePath(
            f"{label} must stay below the project or system temporary directory: "
            f"{lexical}"
        )

    # Require the lexical spelling to use the canonical root as well.  This
    # rejects aliases such as an in-project symlink that happens to resolve to
    # another approved location.
    if not lexical.is_relative_to(allowed_root):
        raise UnsafeMutablePath(
            f"{label} must not traverse a symbolic-link path: {lexical}"
        )
    symlink = _existing_symlink_component(lexical, root=allowed_root)
    if symlink is not None:
        raise UnsafeMutablePath(
            f"{label} must not use a symbolic-link destination or parent: {symlink}"
        )
    return lexical


__all__ = ["UnsafeMutablePath", "confined_mutable_path"]
