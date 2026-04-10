"""wiki_search — search the wiki index and pages by keyword."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from ..util import REPO_ROOT, truncate

WIKI_ROOT = REPO_ROOT / "wiki"


def _search_file(path: Path, pattern: re.Pattern[str], max_bytes: int = 500_000) -> List[Dict[str, Any]]:
    """Return matching lines with context from a single file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    except Exception:
        return []
    hits: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), 1):
        if pattern.search(line):
            hits.append({"line": i, "text": line.strip()})
    return hits


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    scope = str(args.get("scope") or "all").strip()

    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc

    # Determine which directories to search
    if scope == "index":
        search_paths = [WIKI_ROOT / "index.md"]
    elif scope in ("entities", "concepts", "guides", "comparisons", "sources"):
        search_dir = WIKI_ROOT / scope
        search_paths = sorted(search_dir.rglob("*.md")) if search_dir.is_dir() else []
    else:
        search_paths = sorted(WIKI_ROOT.rglob("*.md"))

    results: List[Dict[str, Any]] = []
    for p in search_paths:
        hits = _search_file(p, pattern)
        if hits:
            results.append({
                "file": str(p.relative_to(REPO_ROOT)),
                "matches": hits[:20],  # cap per file
            })

    return {
        "query": query,
        "scope": scope,
        "total_files_matched": len(results),
        "results": results[:50],  # cap total
    }


SPEC = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Keyword or regex pattern to search for in the wiki.",
        },
        "scope": {
            "type": "string",
            "enum": ["all", "index", "entities", "concepts", "guides", "comparisons", "sources"],
            "default": "all",
            "description": "Limit search to a specific wiki section. Default: all.",
        },
    },
    "required": ["query"],
}
