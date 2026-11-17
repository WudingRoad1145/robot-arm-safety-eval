.PHONY: test smoke check
PYTHON ?= python3
test:
	$(PYTHON) -m unittest discover -s tests -v
smoke:
	$(PYTHON) -m robot_eval run --config configs/smoke.json --out runs
check: test smoke
