"""wiki_lint — health-check the wiki for common issues."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Set

from ..util import REPO_ROOT

WIKI_ROOT = REPO_ROOT / "wiki"

_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def handler(args: Dict[str, Any]) -> Dict[str, Any]:
    checks = args.get("checks")
    if not checks:
        checks = ["orphans", "broken_links", "missing_frontmatter", "empty_pages"]

    all_pages: List[Path] = sorted(WIKI_ROOT.rglob("*.md"))
    # Skip .gitkeep and non-md
    all_pages = [p for p in all_pages if p.suffix == ".md"]

    rel_paths: Set[str] = set()
    for p in all_pages:
        rel_paths.add(str(p.relative_to(WIKI_ROOT)))

    issues: List[Dict[str, Any]] = []

    # Collect all inbound links for orphan detection
    inbound: Dict[str, Set[str]] = {rp: set() for rp in rel_paths}
    page_contents: Dict[str, str] = {}

    for p in all_pages:
        rp = str(p.relative_to(WIKI_ROOT))
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        page_contents[rp] = text

        # Parse markdown links
        for _label, href in _MD_LINK.findall(text):
            if href.startswith("http") or href.startswith("#"):
                continue
            # Resolve relative link
            target = (p.parent / href.split("#")[0]).resolve()
            if target.suffix == ".md" and str(target).startswith(str(WIKI_ROOT)):
                target_rp = str(target.relative_to(WIKI_ROOT))
                if target_rp in inbound:
                    inbound[target_rp].add(rp)

    # Check: orphan pages (no inbound links, excluding index/log/overview)
    if "orphans" in checks:
        skip = {"index.md", "log.md", "overview.md"}
        for rp, sources in inbound.items():
            if rp in skip:
                continue
            if not sources:
                issues.append({
                    "check": "orphan",
                    "page": f"wiki/{rp}",
                    "detail": "No inbound links from other wiki pages",
                })

    # Check: broken internal links
    if "broken_links" in checks:
        for p in all_pages:
            rp = str(p.relative_to(WIKI_ROOT))
            text = page_contents.get(rp, "")
            for label, href in _MD_LINK.findall(text):
                if href.startswith("http") or href.startswith("#"):
                    continue
                target = (p.parent / href.split("#")[0]).resolve()
                if not target.exists():
                    issues.append({
                        "check": "broken_link",
                        "page": f"wiki/{rp}",
                        "detail": f"Link target not found: {href}",
                    })

    # Check: missing YAML frontmatter
    if "missing_frontmatter" in checks:
        for rp, text in page_contents.items():
            if rp in ("log.md",):
                continue
            if not text.startswith("---"):
                issues.append({
                    "check": "missing_frontmatter",
                    "page": f"wiki/{rp}",
                    "detail": "Page does not start with YAML frontmatter (---)",
                })

    # Check: empty or near-empty pages
    if "empty_pages" in checks:
        for rp, text in page_contents.items():
            # Strip frontmatter
            body = text
            if text.startswith("---"):
                end = text.find("---", 3)
                if end > 0:
                    body = text[end + 3:]
            if len(body.strip()) < 50:
                issues.append({
                    "check": "empty_page",
                    "page": f"wiki/{rp}",
                    "detail": f"Page body is very short ({len(body.strip())} chars)",
                })

    return {
        "total_pages": len(all_pages),
        "total_issues": len(issues),
        "issues": issues,
    }


SPEC = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["orphans", "broken_links", "missing_frontmatter", "empty_pages"],
            },
            "description": (
                "Which checks to run. Default: all. "
                "orphans = pages with no inbound links, "
                "broken_links = internal links to non-existent pages, "
                "missing_frontmatter = pages without YAML frontmatter, "
                "empty_pages = pages with very little content."
            ),
            "default": ["orphans", "broken_links", "missing_frontmatter", "empty_pages"],
        },
    },
    "required": [],
}
