# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for PythonExtractor, TypeScriptExtractor, and JavaExtractor."""

import pytest
from chonk.extractors._java import JavaExtractor
from chonk.extractors._python import PythonExtractor
from chonk.extractors._typescript import TypeScriptExtractor

# =============================================================================
# PythonExtractor
# =============================================================================

class TestPythonExtractor:
    def test_class_becomes_h1(self):
        src = b"class MyClass:\n    pass\n"
        result = PythonExtractor().extract(src)
        assert "# MyClass" in result

    def test_method_becomes_h2(self):
        src = b"class MyClass:\n    def my_method(self):\n        pass\n"
        result = PythonExtractor().extract(src)
        assert "## my_method" in result

    def test_docstring_emitted_as_prose(self):
        src = b'class Foo:\n    """This is Foo."""\n    pass\n'
        result = PythonExtractor().extract(src)
        assert "This is Foo." in result
        assert "```python" in result

    def test_imports_grouped(self):
        src = b"import os\nimport sys\n\nclass Foo:\n    pass\n"
        result = PythonExtractor().extract(src)
        assert "## Imports" in result
        assert "import os" in result
        assert "import sys" in result

    def test_syntax_error_raises_value_error(self):
        src = b"def broken(\n"
        with pytest.raises(ValueError):
            PythonExtractor().extract(src)

    def test_module_level_function_becomes_h1(self):
        src = b"def top_level():\n    pass\n"
        result = PythonExtractor().extract(src)
        assert "# top_level" in result

    def test_async_function_handled(self):
        src = b"async def fetch_data():\n    pass\n"
        result = PythonExtractor().extract(src)
        assert "# fetch_data" in result

    def test_empty_input_returns_empty_string(self):
        result = PythonExtractor().extract(b"")
        assert result == ""

    def test_can_handle_python(self):
        assert PythonExtractor().can_handle("python")

    def test_cannot_handle_text(self):
        assert not PythonExtractor().can_handle("text")

    def test_method_docstring_as_prose(self):
        src = b'class Foo:\n    def bar(self):\n        """Bar does bar."""\n        pass\n'
        result = PythonExtractor().extract(src)
        assert "Bar does bar." in result

    def test_unicode_decode_error_raises_value_error(self):
        src = bytes([0xFF, 0xFE, 0x00])
        with pytest.raises(ValueError):
            PythonExtractor().extract(src)


# =============================================================================
# TypeScriptExtractor
# =============================================================================

class TestTypeScriptExtractor:
    def test_class_becomes_h1(self):
        src = b"class MyService {\n  run() {}\n}\n"
        result = TypeScriptExtractor().extract(src)
        assert "# MyService" in result

    def test_interface_becomes_h1(self):
        src = b"interface IUser {\n  name: string;\n}\n"
        result = TypeScriptExtractor().extract(src)
        assert "# IUser" in result

    def test_method_becomes_h2(self):
        src = b"class Svc {\n  doWork() {\n    return 1;\n  }\n}\n"
        result = TypeScriptExtractor().extract(src)
        assert "## doWork" in result

    def test_jsdoc_emitted_as_prose(self):
        src = b"/** Does the thing. */\nfunction doThing() {}\n"
        result = TypeScriptExtractor().extract(src)
        assert "Does the thing." in result

    def test_arrow_function_becomes_h1(self):
        src = b"const greet = (name: string) => {\n  return name;\n};\n"
        result = TypeScriptExtractor().extract(src)
        assert "# greet" in result

    def test_can_handle_javascript(self):
        assert TypeScriptExtractor().can_handle("javascript")

    def test_can_handle_typescript(self):
        assert TypeScriptExtractor().can_handle("typescript")

    def test_cannot_handle_python(self):
        assert not TypeScriptExtractor().can_handle("python")

    def test_imports_grouped(self):
        src = b"import { foo } from './foo';\nimport bar from 'bar';\n\nclass A {}\n"
        result = TypeScriptExtractor().extract(src)
        assert "## Imports" in result

    def test_empty_input_returns_empty_string(self):
        result = TypeScriptExtractor().extract(b"")
        assert result == ""

    def test_abstract_class_becomes_h1(self):
        src = b"abstract class Base {\n  abstract run(): void;\n}\n"
        result = TypeScriptExtractor().extract(src)
        assert "# Base" in result

    def test_export_function_becomes_h1(self):
        src = b"export function helper() {\n  return 1;\n}\n"
        result = TypeScriptExtractor().extract(src)
        assert "# helper" in result


# =============================================================================
# JavaExtractor
# =============================================================================

class TestJavaExtractor:
    def test_class_becomes_h1(self):
        src = b"public class Foo {\n  void bar() {}\n}\n"
        result = JavaExtractor().extract(src)
        assert "# Foo" in result

    def test_interface_becomes_h1(self):
        src = b"public interface IService {\n  void run();\n}\n"
        result = JavaExtractor().extract(src)
        assert "# IService" in result

    def test_method_becomes_h2(self):
        src = b"public class Foo {\n  public void doStuff() {\n    int x = 1;\n  }\n}\n"
        result = JavaExtractor().extract(src)
        assert "## doStuff" in result

    def test_javadoc_emitted_as_prose(self):
        src = b"/** Main entry point. */\npublic class App {\n  public static void main(String[] args) {}\n}\n"
        result = JavaExtractor().extract(src)
        assert "Main entry point." in result

    def test_enum_becomes_h1(self):
        src = b"public enum Color {\n  RED, GREEN, BLUE\n}\n"
        result = JavaExtractor().extract(src)
        assert "# Color" in result

    def test_can_handle_java(self):
        assert JavaExtractor().can_handle("java")

    def test_cannot_handle_python(self):
        assert not JavaExtractor().can_handle("python")

    def test_imports_grouped(self):
        src = b"import java.util.List;\nimport com.example.Foo;\n\npublic class Bar {}\n"
        result = JavaExtractor().extract(src)
        assert "## Imports" in result

    def test_empty_input_returns_empty_string(self):
        result = JavaExtractor().extract(b"")
        assert result == ""

    def test_record_becomes_h1(self):
        src = b"public record Point(int x, int y) {}\n"
        result = JavaExtractor().extract(src)
        assert "# Point" in result
