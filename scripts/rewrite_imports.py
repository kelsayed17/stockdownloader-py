#!/usr/bin/env python3
"""Rewrite Python imports across the codebase using a JSON mapping file.

Reads a JSON mapping of old module paths to new module paths and rewrites
all matching imports in ``src/`` and ``tests/``.

Handles:
    1. ``from stockdownloader.old.path import Name``
    2. ``from stockdownloader.old.path import (Name1, Name2)`` (multiline)
    3. ``import stockdownloader.old.path``
    4. String references in lazy imports: ``importlib.import_module("old.path")``
    5. TYPE_CHECKING-guarded imports (same syntax, just under ``if TYPE_CHECKING:``)
    6. String values in ``.json`` files (e.g. config ``"module": "old.path"``)

Usage::

    python3 scripts/rewrite_imports.py mappings.json [--dry-run]
    python3 scripts/rewrite_imports.py mappings.json --dry-run --include-json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def load_mappings(mapping_path: str) -> list[tuple[str, str]]:
    """Load old->new module path mappings from a JSON file.

    Returns mappings sorted by key length descending (longest first)
    so that more specific paths are replaced before shorter prefixes.
    """
    with open(mapping_path) as f:
        raw: dict[str, str] = json.load(f)

    # Sort by key length descending to avoid partial matches.
    # e.g. "stockdownloader.model.financial_models" must match
    # before "stockdownloader.model".
    pairs = sorted(raw.items(), key=lambda kv: len(kv[0]), reverse=True)
    return pairs


def rewrite_python_file(
    filepath: Path,
    mappings: list[tuple[str, str]],
    *,
    dry_run: bool = False,
) -> tuple[int, list[str]]:
    """Rewrite imports in a single Python file.

    Returns (replacement_count, list_of_change_descriptions).
    """
    try:
        original = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return 0, []

    content = original
    total_replacements = 0
    changes: list[str] = []

    for old_path, new_path in mappings:
        old_escaped = re.escape(old_path)

        # Pattern 1 & 2 & 5: "from <old_path> import ..." (single-line and
        # multiline, including under TYPE_CHECKING guards).
        # The "from" line contains the module path; we only replace there.
        pattern_from = re.compile(
            r"(from\s+)" + old_escaped + r"(\s+import\b)"
        )
        count = len(pattern_from.findall(content))
        if count:
            content = pattern_from.sub(
                lambda m: m.group(1) + new_path + m.group(2), content
            )
            total_replacements += count
            changes.append(
                f"  from ... import: {old_path} -> {new_path} ({count}x)"
            )

        # Pattern 3: "import <old_path>" (bare import, possibly with "as" alias
        # or inline comment / noqa).
        # Must NOT match "from X import Y" (already handled above).
        # Uses a negative lookbehind for "from " to avoid double-matching.
        pattern_import = re.compile(
            r"(?<!from\s)"            # not preceded by "from "
            r"(^[ \t]*import\s+)"     # "import " at start of line
            + old_escaped
            + r"(\s|$|;|#)",          # followed by whitespace, EOL, ;, or comment
            re.MULTILINE,
        )
        count = len(pattern_import.findall(content))
        if count:
            content = pattern_import.sub(
                lambda m: m.group(1) + new_path + m.group(2), content
            )
            total_replacements += count
            changes.append(
                f"  import ...: {old_path} -> {new_path} ({count}x)"
            )

        # Pattern 4: String references in lazy imports.
        # Matches import_module("old.path") or import_module('old.path').
        pattern_lazy = re.compile(
            r"(import_module\(\s*[\"'])" + old_escaped + r"([\"']\s*\))"
        )
        count = len(pattern_lazy.findall(content))
        if count:
            content = pattern_lazy.sub(
                lambda m: m.group(1) + new_path + m.group(2), content
            )
            total_replacements += count
            changes.append(
                f"  import_module(): {old_path} -> {new_path} ({count}x)"
            )

    if content == original:
        return 0, []

    if not dry_run:
        filepath.write_text(content, encoding="utf-8")

    return total_replacements, changes


def rewrite_json_file(
    filepath: Path,
    mappings: list[tuple[str, str]],
    *,
    dry_run: bool = False,
) -> tuple[int, list[str]]:
    """Rewrite module path strings in a JSON file.

    Targets string values that exactly match or contain an old module path
    (e.g. ``"module": "stockdownloader.strategy.daily.simple_strategies"``).

    Returns (replacement_count, list_of_change_descriptions).
    """
    try:
        original = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return 0, []

    content = original
    total_replacements = 0
    changes: list[str] = []

    for old_path, new_path in mappings:
        old_escaped = re.escape(old_path)

        # Match JSON string values containing the old module path.
        # Captures: "...old_path..." where old_path sits between quotes
        # with possible surrounding text. Uses a negative lookbehind to
        # avoid matching when preceded by alphanumeric/underscore chars
        # (prevents partial substring matches inside longer paths).
        pattern = re.compile(
            r'("(?:[^"\\]|\\.)*?)(?<![a-zA-Z0-9_])'
            + old_escaped
            + r'((?:[^"\\]|\\.)*?")'
        )

        count = len(pattern.findall(content))
        if count:
            content = pattern.sub(
                lambda m: m.group(1) + new_path + m.group(2), content
            )
            total_replacements += count
            changes.append(
                f"  json string: {old_path} -> {new_path} ({count}x)"
            )

    if not dry_run:
        filepath.write_text(content, encoding="utf-8")

    return total_replacements, changes


def find_files(
    root: Path,
    directories: list[str],
    *,
    extensions: tuple[str, ...] = (".py",),
) -> list[Path]:
    """Recursively find files with given extensions in specified directories."""
    files: list[Path] = []
    for dirname in directories:
        dirpath = root / dirname
        if not dirpath.is_dir():
            continue
        for ext in extensions:
            files.extend(sorted(dirpath.rglob(f"*{ext}")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite Python imports using a JSON mapping file.",
        epilog=(
            "Example:\n"
            "  python3 scripts/rewrite_imports.py mappings.json --dry-run\n"
            "  python3 scripts/rewrite_imports.py mappings.json --include-json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mapping_file",
        help="JSON file mapping old module paths to new module paths.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files.",
    )
    parser.add_argument(
        "--include-json",
        action="store_true",
        help="Also rewrite module paths in .json config files.",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["src", "tests"],
        help="Directories to search (default: src tests).",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root (parent of scripts/).
    project_root = Path(__file__).resolve().parent.parent
    mapping_file = Path(args.mapping_file)
    if not mapping_file.is_absolute():
        mapping_file = project_root / mapping_file

    if not mapping_file.exists():
        print(f"Error: mapping file not found: {mapping_file}", file=sys.stderr)
        sys.exit(1)

    mappings = load_mappings(str(mapping_file))
    if not mappings:
        print("Warning: mapping file is empty, nothing to do.", file=sys.stderr)
        sys.exit(0)

    mode_label = "[DRY RUN] " if args.dry_run else ""
    print(f"{mode_label}Loaded {len(mappings)} mapping(s) from {mapping_file.name}")
    print(f"{mode_label}Mappings (longest first):")
    for old, new in mappings:
        print(f"  {old}  ->  {new}")
    print()

    # --- Python files ---
    py_files = find_files(project_root, args.dirs, extensions=(".py",))
    total_files_changed = 0
    total_replacements = 0

    for filepath in py_files:
        count, changes = rewrite_python_file(
            filepath, mappings, dry_run=args.dry_run
        )
        if count:
            rel = filepath.relative_to(project_root)
            total_files_changed += 1
            total_replacements += count
            print(f"{mode_label}{rel}  ({count} replacement(s))")
            for ch in changes:
                print(f"  {ch}")

    # --- JSON files (optional) ---
    json_files_changed = 0
    json_replacements = 0

    if args.include_json:
        json_dirs = args.dirs + ["config"]
        json_files = find_files(
            project_root, json_dirs, extensions=(".json",)
        )
        for filepath in json_files:
            count, changes = rewrite_json_file(
                filepath, mappings, dry_run=args.dry_run
            )
            if count:
                rel = filepath.relative_to(project_root)
                json_files_changed += 1
                json_replacements += count
                print(f"{mode_label}{rel}  ({count} replacement(s))")
                for ch in changes:
                    print(f"  {ch}")

    # --- Summary ---
    print()
    print(f"{mode_label}Summary:")
    print(f"  Python files changed: {total_files_changed}")
    print(f"  Python replacements:  {total_replacements}")
    if args.include_json:
        print(f"  JSON files changed:   {json_files_changed}")
        print(f"  JSON replacements:    {json_replacements}")
    grand_total = total_replacements + json_replacements
    print(f"  Total replacements:   {grand_total}")

    if args.dry_run and grand_total > 0:
        print(f"\n{mode_label}No files were modified. "
              "Remove --dry-run to apply changes.")


if __name__ == "__main__":
    main()
