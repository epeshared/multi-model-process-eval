from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any

# Allow running as a script without requiring `scripts/` to be a Python package.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from mteb_container_encoder import ContainerMTEBConfig, ContainerMTEBEncoder


def _parse_list(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or None


def _maybe_json_dict(path_or_json: str | None) -> dict[str, Any] | None:
    if not path_or_json:
        return None
    p = Path(path_or_json)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(path_or_json)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".tmp.",
        delete=False,
    ) as tf:
        tf.write(payload)
        tmp_path = Path(tf.name)
    os.replace(tmp_path, path)
    # When running as root with a restrictive umask, files can become unreadable
    # from the editor/user session. Results are not sensitive; make them readable.
    try:
        path.chmod(0o644)
    except Exception:
        pass


def _is_valid_json_file(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        if path.stat().st_size == 0:
            return False
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def _load_json_if_valid(path: Path) -> Any | None:
    if not _is_valid_json_file(path):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _merge_run_by_model_backend(
    runs: list[Any],
    *,
    model_name: str,
    backend: str,
    run_entry: dict[str, Any],
) -> list[Any]:
    """Merge/overwrite a run in `runs` by matching (model_name, backend).

    Legacy behavior:
      If an existing entry has matching model_name but missing backend, we treat it
      as a match (only if there isn't already an exact (model_name, backend) match).
    """
    out = list(runs)

    # Prefer exact match.
    for i, r in enumerate(list(out)):
        if not isinstance(r, dict):
            continue
        if r.get("model_name") == model_name and r.get("backend") == backend:
            out[i] = run_entry
            return out

    # Fallback: upgrade a legacy entry that has model_name but no backend.
    legacy_index: int | None = None
    for i, r in enumerate(list(out)):
        if not isinstance(r, dict):
            continue
        if r.get("model_name") == model_name and ("backend" not in r or r.get("backend") in (None, "")):
            legacy_index = i
            break
    if legacy_index is not None:
        out[legacy_index] = run_entry
        return out

    out.append(run_entry)
    return out


def _task_names(tasks: Any) -> list[str]:
    names: list[str] = []
    for task in tasks:
        name = getattr(getattr(task, "metadata", None), "name", None) or getattr(task, "task_name", None)
        if name:
            names.append(str(name))
    # Preserve order but de-dupe
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _prune_output_folder(output_folder: Path, *, keep_task_names: list[str]) -> None:
    """Remove MTEB cache/view artifacts and keep only per-task summary JSONs.

    Keeps:
      output_folder/results/<task_name>.json for task_name in keep_task_names
    Removes:
      output_folder/view/**
      output_folder/results/<model_name>/** (per-model cache dirs)
      output_folder/results/*.json that are not in keep_task_names
    """

    view_root = output_folder / "view"
    if view_root.exists():
        shutil.rmtree(view_root, ignore_errors=True)

    results_root = output_folder / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    keep_files = {f"{t}.json" for t in keep_task_names}

    for p in list(results_root.iterdir()):
        # Keep only the summary files for tasks run.
        if p.is_file() and p.name.endswith(".json"):
            if p.name not in keep_files:
                try:
                    p.unlink()
                except Exception:
                    pass
            continue

        # Remove any per-model result directories (e.g. results/<safe_model_name>/...)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MTEB using existing container embedding backends (vLLM HTTP / SGLang HTTP)."
    )

    # Backend / server
    parser.add_argument("--backend", required=True, choices=["vllm-http", "sglang"], help="Embedding backend")
    parser.add_argument("--base-url", required=True, help="Embedding server base URL, e.g. http://127.0.0.1:9090")
    parser.add_argument("--model-id", required=True, help="Model id (must match server served-model-name)")
    parser.add_argument("--api", default="v1", choices=["native", "v1", "openai"], help="SGLang API mode")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""), help="Optional API key")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--encoding-format",
        default=None,
        choices=[None, "base64", "float"],
        help="vLLM embeddings encoding_format preference",
    )

    # Encoding behavior
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--tokenizer-id",
        default="",
        help="Optional HuggingFace repo id or local path for tokenizer; used only to compute seq_len/token_len stats.",
    )
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")

    # Optional profiling (only used by sglang-http client)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-kwargs",
        default=None,
        help="JSON dict or path to JSON file passed to sglang /start_profile and /stop_profile",
    )

    # Task selection
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names (e.g. 'STSBenchmark,MSMARCO')",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark name (mutually exclusive with --tasks).",
    )
    parser.add_argument("--task-types", default=None, help="Comma-separated task types")
    parser.add_argument("--languages", default=None, help="Comma-separated languages, e.g. 'eng,zho'")
    parser.add_argument("--domains", default=None, help="Comma-separated domains")

    # Output
    parser.add_argument(
        "--output-folder",
        default=str(Path("scripts/embedding/mteb").resolve()),
        help="Folder where MTEB cache/results are stored",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run by deleting existing result JSONs before evaluation.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete this model's cached result folder under output_folder/results (and output_folder/view) before evaluation.",
    )

    parser.add_argument(
        "--prune-output",
        action="store_true",
        help="After evaluation, delete generated cache/view files and keep only output_folder/results/<task>.json summary files for tasks run.",
    )

    args = parser.parse_args()

    try:
        import mteb  # type: ignore
    except Exception as e:
        raise SystemExit(
            "MTEB is not installed. Install with: pip install -r requirements.txt -r requirements-mteb.txt\n"
            f"Original error: {e}"
        )

    if args.tasks and args.benchmark:
        raise SystemExit("Use only one of --tasks or --benchmark")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    profile_kwargs = _maybe_json_dict(args.profile_kwargs)

    cfg = ContainerMTEBConfig(
        backend=args.backend,
        base_url=args.base_url,
        model_id=args.model_id,
        api=args.api,
        api_key=args.api_key,
        timeout=args.timeout,
        encoding_format=args.encoding_format,
        normalize=not args.no_normalize,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tokenizer_id=str(args.tokenizer_id or ""),
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        profile=args.profile,
        profile_kwargs=profile_kwargs,
    )

    model = ContainerMTEBEncoder(cfg)

    if args.benchmark:
        benchmark = mteb.get_benchmark(args.benchmark)
        tasks = benchmark
    else:
        tasks = mteb.get_tasks(
            tasks=_parse_list(args.tasks),
            task_types=_parse_list(args.task_types),
            languages=_parse_list(args.languages),
            domains=_parse_list(args.domains),
        )

    cache = mteb.ResultCache(cache_path=output_folder)

    safe_model_name = model.mteb_model_meta.name.replace("/", "__")
    revision = getattr(model, "revision", "container")

    # If user asks to clear cache, remove the whole per-model folder so MTEB cannot reuse.
    if args.clear_cache:
        results_dir = output_folder / "results" / safe_model_name / revision
        view_dir = output_folder / "view" / safe_model_name / revision
        for d in (results_dir, view_dir):
            if d.exists():
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    print(f"[mteb] warning: failed to remove cache dir {d}: {e}")

    # If user asks to overwrite, delete existing result JSON files so MTEB cannot skip.
    if args.overwrite:
        for task in tasks:
            task_name = getattr(getattr(task, "metadata", None), "name", None) or getattr(task, "task_name", None)
            if not task_name:
                continue
            result_path = output_folder / "results" / safe_model_name / revision / f"{task_name}.json"
            if result_path.exists():
                try:
                    result_path.unlink()
                except Exception as e:
                    print(f"[mteb] warning: failed to remove {result_path}: {e}")
            view_path = output_folder / "view" / safe_model_name / revision / f"{task_name}.json"
            if view_path.exists():
                try:
                    view_path.unlink()
                except Exception as e:
                    print(f"[mteb] warning: failed to remove {view_path}: {e}")

    # If cached result JSON exists but is corrupt/empty, MTEB will crash while loading it.
    # Preflight and remove invalid cache files (safe: they are rebuildable).
    safe_model_name = model.mteb_model_meta.name.replace("/", "__")
    revision = getattr(model, "revision", "container")
    evaluated_task_names = _task_names(tasks)
    for task in tasks:
        task_name = getattr(getattr(task, "metadata", None), "name", None) or getattr(task, "task_name", None)
        if not task_name:
            continue
        result_path = output_folder / "results" / safe_model_name / revision / f"{task_name}.json"
        if result_path.exists() and not _is_valid_json_file(result_path):
            print(f"[mteb] warning: found invalid cached JSON, removing: {result_path}")
            try:
                result_path.unlink()
            except Exception as e:
                print(f"[mteb] warning: failed to remove invalid cache {result_path}: {e}")

    print(f"[mteb] backend={args.backend} base_url={args.base_url} model_id={args.model_id}")
    print(f"[mteb] output_folder={output_folder}")

    mteb.evaluate(
        model,
        tasks,
        cache=cache,
    )

    # Print workload stats (how many texts/samples were encoded and how many batches were issued).
    stats_after = model.get_encoding_stats() if hasattr(model, "get_encoding_stats") else {}
    if isinstance(stats_after, dict):
        try:
            total_texts = int(stats_after.get("total_texts_encoded") or 0)
            total_batches = int(stats_after.get("total_batches") or 0)
            batch_size = int(stats_after.get("batch_size") or int(args.batch_size))
            encode_time_s = float(stats_after.get("encode_time_s") or 0.0)
            print(
                "[mteb] "
                f"total_samples={total_texts} total_batches={total_batches} "
                f"batch_size={batch_size} encode_time_s={encode_time_s:.6f}"
            )
        except Exception:
            pass

    # ----
    # Post-process results:
    # - Keep the canonical MTEB cache JSON under output_folder/results intact.
    #   (MTEB loads it on cached runs; removing fields like `scores` would break.)
    # - Write a separate human-friendly view JSON under output_folder/view that contains only:
    #     {dataset_revision, task_name, mteb_version, runs:[...per-model...]}
    # ----
    safe_model_name = model.mteb_model_meta.name.replace("/", "__")
    revision = getattr(model, "revision", "container")
    stats = model.get_encoding_stats() if hasattr(model, "get_encoding_stats") else {}
    stats_texts = int(stats.get("total_texts_encoded") or 0) if isinstance(stats, dict) else 0
    recorded_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    for task in tasks:
        task_name = getattr(getattr(task, "metadata", None), "name", None) or getattr(task, "task_name", None)
        if not task_name:
            continue
        result_path = output_folder / "results" / safe_model_name / revision / f"{task_name}.json"
        if not result_path.exists():
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Keep canonical MTEB cache JSON compatible: only add non-breaking fields.
                if "evaluation_time" in data and "evaluation_time_unit" not in data:
                    data["evaluation_time_unit"] = "seconds"
                if stats_texts > 0:
                    data.setdefault("embedding_stats", {})
                    if isinstance(data["embedding_stats"], dict):
                        data["embedding_stats"].update(stats)
                _atomic_write_json(result_path, data)

                # Build/merge view JSON that only contains runs.
                view_path = output_folder / "view" / safe_model_name / revision / f"{task_name}.json"
                view_data = _load_json_if_valid(view_path)
                runs: list[Any]
                if isinstance(view_data, dict) and isinstance(view_data.get("runs"), list):
                    runs = list(view_data["runs"])
                else:
                    runs = []

                run_entry: dict[str, Any] = {
                    "scores": data.get("scores"),
                    "evaluation_time": data.get("evaluation_time"),
                    "kg_co2_emissions": data.get("kg_co2_emissions"),
                    "model_id": args.model_id,
                    "evaluation_time_unit": data.get("evaluation_time_unit", "seconds"),
                    "embedding_stats": data.get("embedding_stats"),
                }

                replaced = False
                for i, r in enumerate(list(runs)):
                    if isinstance(r, dict) and r.get("model_id") == args.model_id:
                        runs[i] = run_entry
                        replaced = True
                        break
                if not replaced:
                    runs.append(run_entry)

                # User-facing view file: keep ONLY runs.
                _atomic_write_json(view_path, {"runs": runs})

                # Build/merge a per-task summary JSON under output_folder/results/<task_name>.json
                # This aggregates runs across different models.
                summary_path = output_folder / "results" / f"{task_name}.json"
                summary_data = _load_json_if_valid(summary_path)

                existing_runs: list[Any]
                if isinstance(summary_data, dict) and isinstance(summary_data.get("runs"), list):
                    existing_runs = list(summary_data["runs"])
                else:
                    existing_runs = []

                model_name = str(args.model_id)
                backend_name = str(args.backend)
                summary_run_entry: dict[str, Any] = {
                    "model_name": model_name,
                    "backend": backend_name,
                    "scores": data.get("scores"),
                    "evaluation_time": data.get("evaluation_time"),
                    "kg_co2_emissions": data.get("kg_co2_emissions"),
                    "evaluation_time_unit": data.get("evaluation_time_unit", "seconds"),
                    "embedding_stats": data.get("embedding_stats"),
                }

                merged_runs = _merge_run_by_model_backend(
                    existing_runs,
                    model_name=model_name,
                    backend=backend_name,
                    run_entry=summary_run_entry,
                )

                merged_summary = {
                    "dataset_revision": data.get("dataset_revision"),
                    "task_name": data.get("task_name") or task_name,
                    "mteb_version": data.get("mteb_version"),
                    "runs": merged_runs,
                }
                _atomic_write_json(summary_path, merged_summary)
        except Exception as e:
            print(f"[mteb] warning: failed to post-process {result_path}: {e}")

    if args.prune_output:
        try:
            _prune_output_folder(output_folder, keep_task_names=evaluated_task_names)
            print(f"[mteb] pruned output_folder; kept task summaries: {evaluated_task_names}")
        except Exception as e:
            print(f"[mteb] warning: failed to prune output_folder={output_folder}: {e}")


if __name__ == "__main__":
    main()
