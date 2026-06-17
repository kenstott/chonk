# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 922f7d92-d696-4926-ae36-c311108071fa
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""ImportCrawler — BFS import-graph traversal for Python, TypeScript/JS, and Java."""

from __future__ import annotations

import ast
import re
import sys
from collections import deque
from pathlib import Path

_TS_IMPORT_RE = re.compile(
    r"""(?:import|from)\s+(?:[\w{},\s*]+\s+from\s+)?['"](\.{1,2}/[^'"]+)['"]"""
)
_JS_REQUIRE_RE = re.compile(r"""require\s*\(\s*['"](\.{1,2}/[^'"]+)['"]\s*\)""")
_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+([\w.]+)\s*;")

_TS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs")
_CODE_EXTENSIONS = {".py", ".pyw"} | set(_TS_EXTENSIONS) | {".java"}

# Python 3.10+ has sys.stdlib_module_names; build a fallback for 3.11-
try:
    _STDLIB_NAMES: frozenset[str] = frozenset(sys.stdlib_module_names)  # type: ignore[attr-defined]
except AttributeError:
    _STDLIB_NAMES = frozenset(
        {
            "abc",
            "ast",
            "asyncio",
            "builtins",
            "collections",
            "contextlib",
            "copy",
            "dataclasses",
            "datetime",
            "enum",
            "functools",
            "hashlib",
            "html",
            "http",
            "importlib",
            "inspect",
            "io",
            "itertools",
            "json",
            "logging",
            "math",
            "os",
            "pathlib",
            "pickle",
            "platform",
            "queue",
            "random",
            "re",
            "shutil",
            "signal",
            "socket",
            "sqlite3",
            "string",
            "struct",
            "subprocess",
            "sys",
            "tempfile",
            "threading",
            "time",
            "traceback",
            "typing",
            "unittest",
            "urllib",
            "uuid",
            "warnings",
            "weakref",
            "xml",
            "zipfile",
            "zlib",
        }
    )

_JAVA_DEFAULT_SKIP = frozenset(
    {
        "java.",
        "javax.",
        "org.springframework.",
        "com.google.",
        "org.apache.",
        "com.fasterxml.",
        "org.slf4j.",
        "ch.qos.",
    }
)


class ImportCrawler:
    """BFS crawler that follows import statements across source files.

    Args:
        root_path:     Resolved paths outside this directory are excluded.
        max_depth:     Maximum import depth from the seed (depth 0 = seed itself).
        skip_prefixes: Java package prefixes to skip (extends default list).
    """

    def __init__(
        self,
        root_path: str | None = None,
        max_depth: int = 3,
        skip_prefixes: list[str] | None = None,
    ) -> None:
        self._root = Path(root_path).resolve() if root_path else None
        self._max_depth = max_depth
        extra = frozenset(f"{p}." if not p.endswith(".") else p for p in (skip_prefixes or []))
        self._java_skip = _JAVA_DEFAULT_SKIP | extra

    def can_handle(self, uri: str) -> bool:
        return Path(uri).suffix.lower() in _CODE_EXTENSIONS

    def crawl(self, uri: str, **kwargs: object) -> list[str]:
        seed = Path(uri).resolve()
        seen: set[str] = set()
        queue: deque[tuple[Path, int]] = deque([(seed, 0)])

        while queue:
            current, depth = queue.popleft()
            key = str(current)
            if key in seen:
                continue
            seen.add(key)

            if depth >= self._max_depth:
                continue

            ext = current.suffix.lower()
            if ext in {".py", ".pyw"}:
                imports = self._resolve_python(current)
            elif ext in set(_TS_EXTENSIONS):
                imports = self._resolve_typescript(current)
            elif ext == ".java":
                imports = self._resolve_java(current)
            else:
                imports = []

            for resolved in imports:
                rkey = str(resolved)
                if rkey not in seen:
                    queue.append((resolved, depth + 1))

        return list(seen)

    # ── Python ───────────────────────────────────────────────────────────────

    def _resolve_python(self, path: Path) -> list[Path]:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        results: list[Path] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = self._resolve_python_module(alias.name, path.parent)
                    if resolved:
                        results.append(resolved)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level or 0
                if level > 0:
                    # Relative import
                    base = path.parent
                    for _ in range(level - 1):
                        base = base.parent
                    if module:
                        module_path = base / module.replace(".", "/")
                    else:
                        module_path = base
                    for name in node.names:
                        candidates = [
                            module_path / f"{name.name}.py",
                            module_path / name.name / "__init__.py",
                            Path(str(module_path) + ".py"),
                            module_path / "__init__.py",
                        ]
                        for c in candidates:
                            if c.exists() and self._within_root(c):
                                results.append(c.resolve())
                                break
                else:
                    resolved = self._resolve_python_module(module, path.parent)
                    if resolved:
                        results.append(resolved)

        return results

    def _resolve_python_module(self, module: str, base_dir: Path) -> Path | None:
        if not module:
            return None
        top = module.split(".")[0]
        if top in _STDLIB_NAMES:
            return None
        root = self._root or base_dir

        parts = module.replace(".", "/")
        candidates = [
            root / f"{parts}.py",
            root / parts / "__init__.py",
        ]
        for c in candidates:
            if c.exists() and self._within_root(c):
                return c.resolve()
        return None

    # ── TypeScript / JavaScript ───────────────────────────────────────────────

    def _resolve_typescript(self, path: Path) -> list[Path]:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        results: list[Path] = []
        for line in source.splitlines():
            for m in _TS_IMPORT_RE.finditer(line):
                spec = m.group(1)
                resolved = self._resolve_ts_specifier(spec, path.parent)
                if resolved:
                    results.append(resolved)
            for m in _JS_REQUIRE_RE.finditer(line):
                spec = m.group(1)
                resolved = self._resolve_ts_specifier(spec, path.parent)
                if resolved:
                    results.append(resolved)
        return results

    def _resolve_ts_specifier(self, spec: str, base_dir: Path) -> Path | None:
        candidate = (base_dir / spec).resolve()
        if candidate.exists() and candidate.suffix.lower() in set(_TS_EXTENSIONS):
            if self._within_root(candidate):
                return candidate
        # Try adding extensions
        for ext in _TS_EXTENSIONS:
            with_ext = Path(str(candidate) + ext)
            if with_ext.exists() and self._within_root(with_ext):
                return with_ext.resolve()
            # Also try without existing suffix replaced
            base_no_ext = candidate.with_suffix("")
            alt = Path(str(base_no_ext) + ext)
            if alt.exists() and self._within_root(alt):
                return alt.resolve()
        return None

    # ── Java ─────────────────────────────────────────────────────────────────

    def _resolve_java(self, path: Path) -> list[Path]:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        results: list[Path] = []
        for line in source.splitlines():
            m = _JAVA_IMPORT_RE.match(line)
            if not m:
                continue
            fqcn = m.group(1)
            if any(fqcn.startswith(prefix) for prefix in self._java_skip):
                continue
            resolved = self._resolve_java_class(fqcn)
            if resolved:
                results.append(resolved)
        return results

    def _resolve_java_class(self, fqcn: str) -> Path | None:
        parts = fqcn.replace(".", "/") + ".java"
        root = self._root or Path(".")
        candidates = [
            root / "src" / "main" / "java" / parts,
            root / parts,
        ]
        for c in candidates:
            if c.exists() and self._within_root(c):
                return c.resolve()
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _within_root(self, path: Path) -> bool:
        if self._root is None:
            return True
        try:
            path.resolve().relative_to(self._root)
            return True
        except ValueError:
            return False
