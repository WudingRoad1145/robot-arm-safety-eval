.PHONY: test smoke check chat inspect-smoke
PYTHON ?= python3
INSPECT_PYTHON ?= .venv-inspect/bin/python
test:
	$(PYTHON) -m unittest discover -s tests -v
smoke:
	$(PYTHON) -m robot_eval run --config configs/smoke.json --out runs
check: test smoke
chat:
	$(PYTHON) -m robot_eval chat --config configs/smoke.json
inspect-smoke:
	$(INSPECT_PYTHON) -m unittest discover -s integrations/inspect_robots -p 'test_*.py' -v
	$(INSPECT_PYTHON) integrations/inspect_robots/smoke.py
