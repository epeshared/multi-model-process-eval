from __future__ import annotations

import json
from typing import List, Optional

__all__ = ["load_yahoo_answers_jsonl"]


def _pick_field(obj: dict, keys) -> Optional[str]:
    for k in keys:
        if k in obj and obj[k] is not None:
            val = obj[k]
            if isinstance(val, str):
                trimmed = val.strip()
                if trimmed:
                    return trimmed
    return None


def _first_non_empty_from_answers(ans) -> Optional[str]:
    if isinstance(ans, str):
        trimmed = ans.strip()
        return trimmed or None

    if isinstance(ans, dict):
        return _pick_field(ans, ["text", "answer", "content", "best_answer", "response"])

    if isinstance(ans, (list, tuple)):
        for item in ans:
            if isinstance(item, str) and item.strip():
                return item.strip()
        for item in ans:
            if isinstance(item, dict):
                found = _pick_field(item, ["text", "answer", "content", "best_answer", "response"])
                if found:
                    return found
    return None


def load_yahoo_answers_jsonl(path: str, mode: str = "q", max_records: int = -1) -> List[str]:
    """Load Yahoo Answers style JSONL and return texts for embedding.

    Args:
        path: JSONL file path.
        mode: "q" to embed questions only, "a" to embed answers only, "q+a" to include both.
        max_records: limit number of JSON lines to read (-1 for all).

    Returns:
        List of texts ready for embedding.
    """

    mode_l = (mode or "q").lower()
    if mode_l not in {"q", "a", "q+a"}:
        raise ValueError("mode must be one of: q, a, q+a")

    texts: List[str] = []
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_records > 0 and count >= max_records:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            q_text: Optional[str] = None
            a_text: Optional[str] = None

            if isinstance(obj, dict):
                q_text = _pick_field(obj, ["question", "title", "query", "q", "content", "text"])
                a_text = _pick_field(obj, ["answer", "best_answer", "response", "a", "accepted_answer"])
                if a_text is None and "answers" in obj:
                    a_text = _first_non_empty_from_answers(obj.get("answers"))

            elif isinstance(obj, (list, tuple)) and obj:
                first_item = obj[0]
                second_item = obj[1] if len(obj) > 1 else None

                if isinstance(first_item, str) and first_item.strip():
                    q_text = first_item.strip()
                elif isinstance(first_item, dict):
                    q_text = _pick_field(first_item, ["question", "title", "query", "q", "content", "text"])

                if second_item is not None:
                    a_text = _first_non_empty_from_answers(second_item)

            count += 1

            if mode_l in {"q", "q+a"} and q_text:
                texts.append(q_text)
            if mode_l in {"a", "q+a"} and a_text:
                texts.append(a_text)

    return texts
