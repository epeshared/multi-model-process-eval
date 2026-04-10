# Raw Sources

Immutable source documents. The LLM reads from here but never modifies.

## Structure

- `sources/` — articles, papers, vendor documentation, release notes
- `results/` — raw benchmark output data (CSV, JSON, logs)
- `assets/` — images, charts, diagrams

## How to Add Sources

1. Drop files into the appropriate subdirectory
2. Ask the LLM to ingest: "Ingest raw/sources/my-article.md"
3. The LLM will:
   - Create a summary page in `wiki/sources/`
   - Update entity/concept pages across the wiki
   - Update `wiki/index.md`
   - Append to `wiki/log.md`
