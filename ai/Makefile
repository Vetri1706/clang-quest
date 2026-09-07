.PHONY: verify estimate release audit

PYTHON ?= python3

verify:
	$(PYTHON) verify.py

estimate:
	$(PYTHON) estimate.py --preset 7b --dp 2 --pp 2 --tp 2

release:
	$(PYTHON) release.py

audit:
	$(PYTHON) release.py --check
