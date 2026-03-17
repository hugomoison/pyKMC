
# ══════════════════════════════════════════════════════════════════════════════
#  TESTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Configuration ─────────────────────────────────────────────────────────────

PYTEST       = pytest
MPIRUN       = mpirun
MPI_FLAGS    = --no-monitor
N ?= 4  # nombre de ranks MPI, surchargeable via : make test-engine-mpi N=8

# ── Engine ────────────────────────────────────────────────────────────────────

.PHONY: test-engine test-engine-serial test-engine-mpi

test-engine: test-engine-serial test-engine-mpi  ## Run all engine tests (serial + MPI)

test-engine-serial:  ## Run serial engine tests
	$(PYTEST) tests/engine/

test-engine-mpi:  ## Run MPI engine tests (n=1, n=4 by default)
	$(MPIRUN) -n 1 $(PYTEST) tests/engine/ $(MPI_FLAGS)
	$(MPIRUN) -n $(N) $(PYTEST) tests/engine/ $(MPI_FLAGS)

# ── All tests ─────────────────────────────────────────────────────────────────

.PHONY: test

test: test-engine  ## Run all tests

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'