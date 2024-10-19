.venv:
	python3.12 -m venv .venv

.PHONY: install
install: requirements.txt
	.venv/bin/pip install -r requirements.txt

.PHONY: run
run: install
	.venv/bin/python analyze.py

