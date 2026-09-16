PYTHON ?= python3
DATA_DIR ?=
CLEAN_FLAG := --exclude-train-year-contradictions

.PHONY: help check-data test audit popularity user-cf bpr graph experiment

help:
	@echo "make test"
	@echo "make audit DATA_DIR=/path/to/hetrec2011-movielens-2k-v2"
	@echo "make experiment DATA_DIR=/path/to/hetrec2011-movielens-2k-v2"

check-data:
	@test -n "$(DATA_DIR)" || (echo "DATA_DIR is required" >&2; exit 2)

test:
	$(PYTHON) -B -m unittest discover -v
	$(PYTHON) -m compileall -q .

audit: check-data
	$(PYTHON) -B audit_dataset.py "$(DATA_DIR)"

popularity: check-data
	$(PYTHON) -B run_popularity_baseline.py "$(DATA_DIR)" $(CLEAN_FLAG)

user-cf: check-data
	$(PYTHON) -B run_user_cf_baseline.py "$(DATA_DIR)" $(CLEAN_FLAG)

bpr: check-data
	$(PYTHON) -B run_bpr_mf_baseline.py "$(DATA_DIR)" $(CLEAN_FLAG)

graph: check-data
	$(PYTHON) -B run_meta_path_recommender.py "$(DATA_DIR)" $(CLEAN_FLAG)

experiment: popularity user-cf bpr graph
