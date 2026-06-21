# ── chonk Go migration build system ──────────────────────────────────────────
# Targets mirror the Python project's CI gates (lint, typecheck, test, cover).
#
# Usage:
#   make build        Build the CLI binary
#   make test         Run all unit tests
#   make test-int     Run integration tests (requires live DuckDB / Postgres)
#   make lint         Run golangci-lint
#   make vet          Run go vet
#   make cover        Generate coverage report (min 70%)
#   make clean        Remove build artefacts

BINARY      := chonk
CMD_DIR     := ./cmd/chonk
BUILD_DIR   := ./bin
COVER_OUT   := coverage.out
COVER_HTML  := coverage.html
COVER_MIN   := 70

GO          ?= go
GOFLAGS     ?=

.PHONY: all build test test-unit test-int lint vet cover clean tidy

all: build

## ── Build ────────────────────────────────────────────────────────────────────
build:
	@mkdir -p $(BUILD_DIR)
	$(GO) build $(GOFLAGS) -o $(BUILD_DIR)/$(BINARY) $(CMD_DIR)

## ── Test ─────────────────────────────────────────────────────────────────────
test: test-unit

test-unit:
	$(GO) test $(GOFLAGS) -timeout 60s ./internal/... ./cmd/...

test-int:
	$(GO) test $(GOFLAGS) -timeout 120s -tags integration ./tests/integration/...

## ── Coverage ─────────────────────────────────────────────────────────────────
cover:
	$(GO) test -coverprofile=$(COVER_OUT) -covermode=atomic \
	    -timeout 60s ./internal/... ./cmd/...
	$(GO) tool cover -func=$(COVER_OUT) | tail -1
	@$(GO) tool cover -func=$(COVER_OUT) | awk \
	    '/^total:/ { pct=$$3+0; if (pct < $(COVER_MIN)) { \
	        printf "FAIL: coverage %.1f%% < $(COVER_MIN)%%\n", pct; exit 1 } \
	        else { printf "OK:   coverage %.1f%%\n", pct } }'
	$(GO) tool cover -html=$(COVER_OUT) -o $(COVER_HTML)

## ── Lint / Vet ───────────────────────────────────────────────────────────────
vet:
	$(GO) vet ./...

lint:
	golangci-lint run ./...

## ── Tidy ─────────────────────────────────────────────────────────────────────
tidy:
	$(GO) mod tidy

## ── Clean ────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR) $(COVER_OUT) $(COVER_HTML)
