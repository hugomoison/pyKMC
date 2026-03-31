
#########
# TESTS #
#########

# Configuration 

PYTEST       = pytest
MPIRUN       = mpirun
MPI_FLAGS    = --no-monitor
N ?= 5  # nombre de ranks MPI, surchargeable via : make test-engine-mpi N=8


# All

.PHONY: test-engine test-engine-serial test-engine-mpi 

# Engine 

test-engine: test-engine-serial test-engine-mpi  ## Run all engine tests (serial + MPI)

test-engine-serial:  ## Run serial engine tests
	$(PYTEST) tests/engine/

test-engine-mpi:  ## Run MPI engine tests (n=1, n=4 by default)
	$(MPIRUN) -n 1 $(PYTEST) tests/engine/ $(MPI_FLAGS)
	$(MPIRUN) -n $(N) $(PYTEST) tests/engine/ $(MPI_FLAGS)

# Worker 

test-worker: #Run MPI worker test 
	$(MPIRUN) -n $(N) $(PYTEST) tests/manager/test_worker.py $(MPI_FLAGS)

# Session 
#
test-session:  ## Run MPI session tests
	$(MPIRUN) -n $(N) $(PYTEST)  tests/manager/test_session.py $(MPI_FLAGS) 

# Manager 
test-manager: ## Run MPI manager tests 
	$(MPIRUN) -n $(N) $(PYTEST) tests/manager/test_manager.py $(MPI_FLAGS) -s

# ── All tests ─────────────────────────────────────────────────────────────────

.PHONY: test

test: test-engine test-worker  test-session test-manager ## Run all tests

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
