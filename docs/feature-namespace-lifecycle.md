# Feature Request: Namespace Lifecycle Management

## Background

Chonk's data model has two namespace types:

- `global` — system-wide shared knowledge, built and owned by the application
- `user:{user_id}` — per-user experimental domains, built and owned by individual users

Each namespace contains a hierarchy of domains. Each domain contains data sources and optional child domains. Chunks carry both `namespace` and `domain_id`, and search filtering operates on `domain_id`.

This model supports a promotion workflow: users build experimental domains in their own namespace, validate them, and promote useful ones to `global` for all users.

## Required Features

### 1. Namespace cache validation

A single call to check whether a namespace index is up-to-date:

```python
store.namespace_cache_valid(namespace_id: str) -> bool
```

Returns `True` if the namespace index exists, is fully built (NER, SVO, community), and all source freshness checks pass. Returns `False` if the namespace has never been built, a source has been modified since last index, or secondary indexes are stale.

**Trigger points in the application:**
- System startup → check `global`; build if invalid
- User login / session start → check `user:{user_id}`; build if invalid
- User modifies their namespace → invalidate `user:{user_id}`; trigger lazy rebuild

### 2. Async namespace builder

A non-blocking build pipeline for a single namespace:

```python
handle = store.build_namespace_async(
    namespace_id: str,
    on_progress: Callable[[str, int, int], None] | None = None,
    on_complete: Callable[[int], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
    force: bool = False,
) -> IndexHandle
```

Runs the full pipeline — crawl, chunk, embed, NER, SVO, community — for all sources registered under `namespace_id`. Returns an `IndexHandle` (same interface as `Indexer`). Safe to call concurrently for different namespaces. Idempotent if `force=False` and cache is valid.

### 3. Background freshness refresh

A periodic background job that re-indexes stale namespaces without manual intervention:

```python
refresher = NamespaceRefresher(
    store: Store,
    interval_seconds: int = 3600,
    staleness_threshold_seconds: int = 86400,
    on_rebuild: Callable[[str], None] | None = None,
)
refresher.start()   # non-blocking
refresher.stop()
```

Each interval, the refresher checks all registered namespaces against their source freshness timestamps. Namespaces whose sources have changed since last index (beyond `staleness_threshold_seconds`) are queued for rebuild via `build_namespace_async`.

### 4. Domain-scoped search (clarification / confirmation)

Search already filters on `domain_id` — this is correct. The caller constructs the full `domain_ids` list spanning both namespaces:

```python
domain_ids = store.resolve_domain_ids(
    [("global", name) for name in active_global_domains] +
    [("user:{user_id}", name) for name in active_user_domains],
    include_global=False,
)
results = store.search(query_vec, domain_ids=domain_ids)
```

No changes needed to the search API. What is needed is a helper to resolve which domain names belong to a given namespace, so callers can split their active domain list correctly:

```python
store.list_domains(namespace_id: str) -> list[str]
# Returns domain names registered under namespace_id
```

### 5. Namespace promotion

Move a domain from a user namespace to global:

```python
store.promote_domain(
    domain_name: str,
    from_namespace: str,   # e.g. "user:alice"
    to_namespace: str,     # e.g. "global"
) -> None
```

Re-registers the domain, its sources, and all associated chunks under the target namespace. Invalidates community cache entries for both namespaces.

## Summary

| Feature | Purpose |
|---|---|
| `namespace_cache_valid()` | Startup and login checks — build only when stale |
| `build_namespace_async()` | Non-blocking full-pipeline build for one namespace |
| `NamespaceRefresher` | Periodic background re-index based on source freshness |
| `list_domains()` | Caller can split active domains by namespace for `resolve_domain_ids` |
| `promote_domain()` | Promote user experiment to global |
