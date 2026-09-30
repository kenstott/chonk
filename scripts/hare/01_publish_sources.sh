#!/usr/bin/env bash
# Maintainer stage 1: regenerate the entity-resolution records and publish the
# corpus to R2. Needs rclone write access to the chonk bucket.
#
# The SEC, 10-K, CVE, Federal Register, and patent sources were captured once and
# are published as-is; only the gleif source has a generator.
#
# Usage: SEC_USER_AGENT="Name email@example.com" scripts/hare/01_publish_sources.sh
set -euo pipefail

: "${SEC_USER_AGENT:?set SEC_USER_AGENT to \"Name email@example.com\" (SEC fair-access policy)}"
PY="${PY:-uv run python}"

$PY scripts/hare/artifacts.py download corpus     # the published sources, verified
$PY scripts/hare/fetch_entity_records.py --sec-user-agent "$SEC_USER_AGENT"
$PY scripts/hare/artifacts.py publish corpus
echo "Commit work/fang2026/artifact_manifest.json."
