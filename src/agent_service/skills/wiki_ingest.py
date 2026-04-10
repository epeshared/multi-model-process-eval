"""wiki_ingest — create or update a wiki page and maintain index/log."""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict

from ..util import REPO_ROOT

WIKI_ROOT = REPO_ROOT / "wiki"
TODAY = lambda: datetime.date.today().isoformat()  # noqa: E731


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    page = str(args.get("page") or "").strip()
    content = str(args.get("content") or "").strip()
    log_entry = str(args.get("log_entry") or "").strip()
    index_row = str(args.get("index_row") or "").strip()

    if not page:
        raise ValueError("page is required (e.g. 'sources/my-source.md')")
    if not content:
        raise ValueError("content is required")

    # Resolve and validate path
    if page.startswith("wiki/"):
        target = REPO_ROOT / page
    else:
        target = WIKI_ROOT / page
    target = target.resolve()

    if not str(target).startswith(str(WIKI_ROOT.resolve())):
        raise ValueError("page must be inside wiki/")

    # Create parent dirs and write page
    target.parent.mkdir(parents=True, exist_ok=True)
    is_new = not target.exists()
    target.write_text(content, encoding="utf-8")

    touched = [str(target.relative_to(REPO_ROOT))]

    # Append to log
    if log_entry:
        log_path = WIKI_ROOT / "log.md"
        if log_path.exists():
            existing = log_path.read_text(encoding="utf-8")
            log_path.write_text(existing.rstrip() + "\n\n" + log_entry + "\n", encoding="utf-8")
            touched.append("wiki/log.md")

    # Append to index
    if index_row:
        index_path = WIKI_ROOT / "index.md"
        if index_path.exists():
            existing = index_path.read_text(encoding="utf-8")
            index_path.write_text(existing.rstrip() + "\n" + index_row + "\n", encoding="utf-8")
            touched.append("wiki/index.md")

    return {
        "ok": True,
        "action": "created" if is_new else "updated",
        "page": str(target.relative_to(REPO_ROOT)),
        "touched": touched,
    }


SPEC = {
    "type": "object",
    "properties": {
        "page": {
            "type": "string",
            "description": (
                "Target page path relative to wiki/ "
                "(e.g. 'sources/my-article.md', 'concepts/new-concept.md'). "
                "Parent directories are created automatically."
            ),
        },
        "content": {
            "type": "string",
            "description": (
                "Full markdown content of the page including YAML frontmatter. "
                "Must follow wiki conventions (title, created, updated, tags, sources)."
            ),
        },
        "log_entry": {
            "type": "string",
            "description": (
                "Optional append-only entry for wiki/log.md. "
                "Format: '## [YYYY-MM-DD] ingest | Title\\n- Summary: ...\\n- Pages touched: ...'"
            ),
            "default": "",
        },
        "index_row": {
            "type": "string",
            "description": (
                "Optional row to append to wiki/index.md. "
                "Format: '| [Page](path) | Category | Summary | YYYY-MM-DD |'"
            ),
            "default": "",
        },
    },
    "required": ["page", "content"],
}
