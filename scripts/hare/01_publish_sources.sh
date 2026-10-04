#!/usr/bin/env bash
# Initialization, stage 1: regenerate the entity-resolution records and publish
# the corpus to the artifact bucket.
#
# The 10-K, CVE, Federal Register, and patent sources were captured once and are
# republished as they are in the bucket; only the gleif source has a generator.
#
# Configuration (environment or .env): HARE_BUCKET_URL, AWS_ACCESS_KEY_ID,
# AWS_SECRET_ACCESS_KEY, optional AWS_ENDPOINT_OVERRIDE / AWS_REGION, and
# SEC_USER_AGENT="Name email@example.com" (SEC fair-access policy).
#
# Usage: scripts/hare/01_publish_sources.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-uv run python}"
if [ -z "${SEC_USER_AGENT:-}" ] && [ -f .env ]; then
    SEC_USER_AGENT=$(grep '^SEC_USER_AGENT=' .env | cut -d= -f2- | tr -d "\"'")
fi
: "${SEC_USER_AGENT:?set SEC_USER_AGENT to \"Name email@example.com\" (environment or .env)}"

$PY scripts/hare/artifacts.py download corpus   # current published sources
$PY scripts/hare/fetch_entity_records.py --sec-user-agent "$SEC_USER_AGENT"
$PY scripts/hare/artifacts.py publish corpus
