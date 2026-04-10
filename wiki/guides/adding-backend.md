---
title: Adding a New Backend
created: 2026-04-10
updated: 2026-04-10
tags: [guide, development, backend]
---

# Adding a New Backend

How to add a new inference backend to the framework.

## Architecture

Each task has a `*_backends/` directory with one file per backend:

```
src/tasks/embedding_backends/
├── sglang_http.py
├── sglang_offline.py
├── vllm_http.py
└── vllm_offline.py
```

## Steps

### 1. Create the Backend File

Create `src/tasks/<task>_backends/<new_backend>.py`.

### 2. Implement the Session Pattern

Every backend must provide:

```python
def load_session(model_id, **kwargs):
    """Load model/connect to server. Return a session object."""
    session = ...
    setattr(session, "_backend_tag", "my-new-backend")
    return session

def embed(session, inputs, **kwargs):
    """Run inference. Return results."""
    ...
```

The `_backend_tag` attribute lets upper layers identify the backend.

### 3. Register in the Task Entry

In `src/tasks/<task>.py`, add your backend to the dispatch logic:

```python
if backend == "my-new-backend":
    from .embedding_backends.my_backend import load_session, embed
```

### 4. Add Server Scripts (if HTTP)

Create `scripts/<task>/<backend_name>/start_<backend>_server.sh` if the backend requires a server process.

### 5. Update Documentation

- Add to the [Backend Feature Matrix](../comparisons/backend-feature-matrix.md)
- Update the task entity page
- Update `wiki/index.md`

## Conventions

- Use env vars for configuration (no hardcoded paths)
- Support `BATCH_SIZE`, `MAX_SAMPLES`, `WARMUP_SAMPLES` where applicable
- Return timing metadata for benchmark aggregation
- Image transport: support data-url and/or base64 for multimodal backends

## Related

- [Backend Feature Matrix](../comparisons/backend-feature-matrix.md)
- [SGLang Backend](../entities/backends/sglang.md) — reference implementation
- [vLLM Backend](../entities/backends/vllm.md) — reference implementation
