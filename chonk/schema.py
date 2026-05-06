# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: b4f1a2e3-8c9d-4e5f-a6b7-c8d9e0f1a2b3
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Structured metadata models for schema and API document loading."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnMeta:
    name: str
    data_type: str
    description: str | None = None
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_ref: str | None = None  # "other_table.col"


@dataclass
class TableMeta:
    name: str
    schema_name: str | None = None
    description: str | None = None
    columns: list[ColumnMeta] = field(default_factory=list)
    row_count: int | None = None
    source_db: str | None = None


@dataclass
class FieldMeta:
    name: str
    field_type: str
    description: str | None = None
    required: bool = False


@dataclass
class EndpointMeta:
    path: str
    method: str | None = None
    description: str | None = None
    fields: list[FieldMeta] = field(default_factory=list)
    endpoint_type: str = "rest"  # "rest" | "graphql_query" | "graphql_mutation" | "graphql_type"
    source_api: str | None = None
