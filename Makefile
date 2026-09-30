# superposition: common tasks. Every target runs inside the uv-managed environment.
# See SPEC.md §5 (engineering standards) and CLAUDE.md.

UV ?= uv
RUN := $(UV) run
EXPERIMENTS = $(shell $(RUN) superposition list --names 2>/dev/null)

.PHONY: help setup lint format test test-slow test-paper run-quick figures report app clean-results

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Create the environment from uv.lock and install pre-commit hooks
	$(UV) sync --locked
	$(RUN) pre-commit install

lint:  ## ruff check, ruff format --check, pyright (basic)
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) pyright

format:  ## Auto-fix lint issues and format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test:  ## Fast tier: unit and property tests (CI)
	$(RUN) pytest -m "not slow" -n auto

test-slow:  ## Slow tier (< 15 min): short replication tests, lab tests, short notebooks
	$(RUN) pytest -m slow --nbmake tests/replication notebooks

test-paper:  ## Paper tier (hours): every quick profile re-run and checked, long notebooks
	$(RUN) pytest -m paper --nbmake tests/replication notebooks

run-quick:  ## Run the quick profile of every registered experiment
	@for exp in $(EXPERIMENTS); do \
		echo "== $$exp"; \
		$(RUN) superposition run $$exp --profile quick || exit 1; \
	done

figures:  ## Regenerate figures for the latest run of every experiment and publish them
	@for exp in $(EXPERIMENTS); do \
		$(RUN) superposition figures $$exp --run latest --publish || exit 1; \
	done

report:  ## Regenerate the tables in docs/REPRODUCTION.md from the latest runs
	$(RUN) superposition report

app:  ## Launch the Streamlit lab
	$(RUN) superposition app

clean-results:  ## Delete all run outputs under results/ (irreversible)
	rm -rf results/
