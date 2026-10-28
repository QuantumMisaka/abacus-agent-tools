"""Shared note-based work-directory naming and allocation."""

import os
import traceback
from itertools import count
from pathlib import Path
from typing import Iterator


def slugify_work_note(value) -> str:
    """Clean a note to the at-most-64-character directory-name slug.

    An empty return value means the note contains no usable characters.
    """
    text = "" if value is None else str(value).strip()
    parts = []
    previous_dash = False
    for char in text:
        if char.isalnum() or char in "_-":
            parts.append(char)
            previous_dash = False
        elif not previous_dash:
            parts.append("-")
            previous_dash = True
    slug = "".join(parts).strip("-")
    if len(slug) > 64:
        slug = slug[:64].rstrip("-")
    return slug


# Existing private import path remains available for library callers.
_slugify_work_note = slugify_work_note


def iter_work_path_names(note) -> Iterator[str]:
    """Yield names for one complete note, without probing or changing the filesystem.

    Names start with the cleaned note and then use the first available ordinal
    from ``-002`` onward. A numeric tail in the note remains part of the note.
    """
    slug = slugify_work_note(note)
    if not slug:
        raise ValueError("note slug is empty; provide a note containing letters, numbers, '_' or '-'.")
    return _iter_names(slug)


def _iter_names(slug: str) -> Iterator[str]:
    yield slug
    for ordinal in count(2):
        yield f"{slug}-{ordinal:03d}"


def generate_work_path(
    note=None, fallback_name=None, create: bool = True, *, parent=None
) -> str:
    """Return the first currently free note-based work directory path.

    ``note=None`` keeps the legacy internal fallback to ``fallback_name`` or
    the direct caller's function name. Existing positional arguments and the
    default cwd-relative return path remain unchanged. With ``parent`` set,
    the returned path is joined to that explicit parent; an absolute parent
    therefore produces an absolute result.

    When ``create=True``, allocation uses atomic ``mkdir`` and retries only
    name collisions. When ``create=False``, this is a side-effect-free query:
    the returned candidate is not reserved and may be taken before use.
    """
    if note is None:
        base = fallback_name or traceback.extract_stack(limit=2)[-2].name
    else:
        base = note

    for name in iter_work_path_names(base):
        path = name if parent is None else str(Path(parent) / name)
        if create:
            try:
                os.mkdir(path)
            except FileExistsError:
                continue
        else:
            try:
                os.lstat(path)
            except FileNotFoundError:
                pass
            else:
                continue
        return path

    raise AssertionError("unreachable: work-path names have no ordinal limit")


def generate_stage_path(
    stage: str,
    *,
    parent=None,
    fallback_note=None,
) -> str:
    """Allocate a purpose-named stage under an explicit work parent.

    With ``parent`` set, the library owns the stage name and creates it below
    the caller-selected directory.  Without a parent, the historical
    ``fallback_note`` allocation is preserved for backward-compatible callers.
    """

    if parent is None:
        return generate_work_path(note=fallback_note)
    if not stage or not str(stage).strip():
        raise ValueError("stage name must be non-empty when parent is provided")
    parent_path = Path(parent).expanduser().resolve()
    parent_path.mkdir(parents=True, exist_ok=True)
    return generate_work_path(note=stage, parent=parent_path)
