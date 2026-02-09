# Scale-test Web UI

A tiny dependency-free web server to browse scale-test run outputs under:

- `scripts/scale-test/*/result/**/analysis/`

It is designed to be **future-proof**: the home page auto-discovers all tasks (e.g. `embedding`, `vl`, `omni`) and all suites under their `result/` folder.

## Start

From repo root:

- `python3 scripts/scale-test/web/server.py --port 8080`

Then open:

- `http://<host>:8080/`

## Notes

- CSVs are previewed (first 200 rows) and can be downloaded.
- PNG plots are rendered as a responsive gallery.
- No external dependencies (uses Python stdlib only).

## Env

- `SCALE_TEST_WEB_LOG=0` disables request logs.
