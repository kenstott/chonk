# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for ImportCrawler."""

from chonk.transports._import_crawler import ImportCrawler

# =============================================================================
# Python import crawling
# =============================================================================


class TestPythonImportCrawler:
    def test_python_crawl_follows_relative_import(self, tmp_path):
        utils = tmp_path / "utils.py"
        utils.write_text("def helper(): pass\n")
        seed = tmp_path / "main.py"
        seed.write_text("from utils import helper\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=1)
        result = crawler.crawl(str(seed))
        assert str(utils) in result
        assert str(seed) in result

    def test_python_crawl_depth_limit(self, tmp_path):
        deep = tmp_path / "deep.py"
        deep.write_text("x = 1\n")
        mid = tmp_path / "mid.py"
        mid.write_text("from deep import x\n")
        seed = tmp_path / "seed.py"
        seed.write_text("from mid import x\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=1)
        result = crawler.crawl(str(seed))
        assert str(seed) in result
        assert str(mid) in result
        assert str(deep) not in result

    def test_python_crawl_root_path_boundary(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        outside = tmp_path / "outside.py"
        outside.write_text("x = 1\n")
        seed = subdir / "seed.py"
        seed.write_text("import sys\n")

        crawler = ImportCrawler(root_path=str(subdir), max_depth=3)
        result = crawler.crawl(str(seed))
        assert str(outside) not in result

    def test_python_skips_stdlib(self, tmp_path):
        seed = tmp_path / "seed.py"
        seed.write_text("import os\nimport sys\nimport json\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=1)
        result = crawler.crawl(str(seed))
        assert len(result) == 1
        assert str(seed) in result

    def test_seed_always_included(self, tmp_path):
        seed = tmp_path / "seed.py"
        seed.write_text("")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=0)
        result = crawler.crawl(str(seed))
        assert str(seed) in result

    def test_deduplication(self, tmp_path):
        shared = tmp_path / "shared.py"
        shared.write_text("x = 1\n")
        a = tmp_path / "a.py"
        a.write_text("from shared import x\n")
        b = tmp_path / "b.py"
        b.write_text("from shared import x\n")
        seed = tmp_path / "seed.py"
        seed.write_text("from a import x\nfrom b import x\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=2)
        result = crawler.crawl(str(seed))
        assert result.count(str(shared)) == 1

    def test_python_package_import(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        init = pkg / "__init__.py"
        init.write_text("from .utils import helper\n")
        utils = pkg / "utils.py"
        utils.write_text("def helper(): pass\n")
        seed = tmp_path / "main.py"
        seed.write_text("from mypkg import helper\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=1)
        result = crawler.crawl(str(seed))
        assert str(init) in result or str(utils) in result


# =============================================================================
# TypeScript/JS import crawling
# =============================================================================


class TestTypeScriptImportCrawler:
    def test_ts_crawl_relative_import(self, tmp_path):
        helper = tmp_path / "helper.ts"
        helper.write_text("export function help() {}\n")
        seed = tmp_path / "main.ts"
        seed.write_text("import { help } from './helper';\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=1)
        result = crawler.crawl(str(seed))
        assert str(helper) in result
        assert str(seed) in result

    def test_ts_skips_bare_specifier(self, tmp_path):
        seed = tmp_path / "main.ts"
        seed.write_text("import React from 'react';\nimport _ from 'lodash';\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=1)
        result = crawler.crawl(str(seed))
        assert len(result) == 1
        assert str(seed) in result

    def test_ts_resolves_tsx_extension(self, tmp_path):
        comp = tmp_path / "Comp.tsx"
        comp.write_text("export default function Comp() { return null; }\n")
        seed = tmp_path / "App.tsx"
        seed.write_text("import Comp from './Comp';\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=1)
        result = crawler.crawl(str(seed))
        assert str(comp) in result

    def test_js_crawl_relative_import(self, tmp_path):
        util = tmp_path / "util.js"
        util.write_text("module.exports = {};\n")
        seed = tmp_path / "index.js"
        seed.write_text("const util = require('./util');\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=1)
        result = crawler.crawl(str(seed))
        assert str(util) in result


# =============================================================================
# Java import crawling
# =============================================================================


class TestJavaImportCrawler:
    def test_java_crawl_package_resolution(self, tmp_path):
        pkg_dir = tmp_path / "com" / "example"
        pkg_dir.mkdir(parents=True)
        foo = pkg_dir / "Foo.java"
        foo.write_text("package com.example; public class Foo {}\n")
        seed = tmp_path / "Main.java"
        seed.write_text("import com.example.Foo;\npublic class Main {}\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=1)
        result = crawler.crawl(str(seed))
        assert str(foo) in result

    def test_java_skips_java_stdlib(self, tmp_path):
        seed = tmp_path / "Main.java"
        seed.write_text("import java.util.List;\nimport java.io.File;\npublic class Main {}\n")

        crawler = ImportCrawler(root_path=str(tmp_path), max_depth=1)
        result = crawler.crawl(str(seed))
        assert len(result) == 1
        assert str(seed) in result

    def test_java_configurable_skip_prefixes(self, tmp_path):
        pkg_dir = tmp_path / "com" / "example"
        pkg_dir.mkdir(parents=True)
        foo = pkg_dir / "Foo.java"
        foo.write_text("package com.example; public class Foo {}\n")
        seed = tmp_path / "Main.java"
        seed.write_text("import com.example.Foo;\npublic class Main {}\n")

        # Skip com.example explicitly
        crawler = ImportCrawler(
            root_path=str(tmp_path),
            max_depth=1,
            skip_prefixes=["com.example"],
        )
        result = crawler.crawl(str(seed))
        assert str(foo) not in result
