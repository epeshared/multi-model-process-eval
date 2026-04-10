"""wiki_read — read a wiki page or list pages in a wiki directory."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..util import REPO_ROOT, truncate

WIKI_ROOT = REPO_ROOT / "wiki"


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    page = str(args.get("page") or "").strip()

    # No page specified → list all wiki files as a directory tree
    if not page:
        return _list_tree()

    # Resolve path (relative to wiki/ or repo root)
    if page.startswith("wiki/"):
        target = REPO_ROOT / page
    else:
        target = WIKI_ROOT / page

    target = target.resolve()

    # Security: ensure path stays within the repo
    if not str(target).startswith(str(REPO_ROOT)):
        raise ValueError("path escapes repository root")

    if target.is_dir():
        files = sorted(target.rglob("*.md"))
        entries: List[Dict[str, str]] = []
        for f in files:
            entries.append({
                "path": str(f.relative_to(REPO_ROOT)),
                "title": _extract_title(f),
            })
        return {"type": "directory", "path": str(target.relative_to(REPO_ROOT)), "entries": entries}

    if not target.exists():
        return {"type": "error", "error": f"not found: {page}"}

    content = target.read_text(encoding="utf-8", errors="replace")
    return {
        "type": "page",
        "path": str(target.relative_to(REPO_ROOT)),
        "content": truncate(content, limit=60000),
    }


def _extract_title(p: Path) -> str:
    """Extract title from YAML frontmatter or first heading."""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception:
        return p.stem
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("title:"):
            return stripped[len("title:"):].strip().strip('"').strip("'")
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return p.stem


def _list_tree() -> Dict[str, Any]:
    """Return full wiki directory tree."""
    entries: List[Dict[str, str]] = []
    for f in sorted(WIKI_ROOT.rglob("*.md")):
        entries.append({
            "path": str(f.relative_to(REPO_ROOT)),
            "title": _extract_title(f),
        })
    return {"type": "tree", "total": len(entries), "entries": entries}


SPEC = {
    "type": "object",
    "properties": {
        "page": {
            "type": "string",
            "description": (
                "Wiki page path relative to wiki/ (e.g. 'index.md', "
                "'entities/models/clip.md', 'concepts/amx.md'). "
                "Omit to list all pages. Can also be a directory to list its contents."
            ),
            "default": "",
        },
    },
    "required": [],
}
