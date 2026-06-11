# Skill Audit — chonk RAG pipeline library Python→Go port
**Generated:** 2026-06-10
**Engagement:** chonk — RAG pipeline library Python→Go port

---

## Clusters in Scope

| Cluster | Name | Status | Manifest |
|---|---|---|---|
| A | Code Transformation | Draft | [modernization_manifest_A.md](modernization_manifest_A.md) |
| C | Test Generation | Draft | [modernization_manifest_C.md](modernization_manifest_C.md) |
| B | Documentation | Out of scope | — |
| D | Analysis & Audit | Out of scope | — |
| E | Data & SQL | Out of scope | — |

---

## Model Strategy

**Strategy:** Hybrid — one model per code cluster (both at the same tier)
**Cluster A model:** `Qwen2.5-Coder-14B-Instruct`
**Cluster C model:** `Qwen2.5-Coder-14B-Instruct`

---

## Alignment Algorithm

**Algorithm:** SFT (supervised fine-tuning) — first run baseline
**Examples per cluster:** 500

---

## Frontier Access

**Approved:** yes — enterprise API agreement in place

---

## Recommended Run Order

1. **Cluster A** first — establish the code transformation adapter. Cluster C test porting depends on Cluster A output being available for dual-run validation.
2. **Cluster C** second — port the test suite using the same adapter model; validate dual-run oracle.
