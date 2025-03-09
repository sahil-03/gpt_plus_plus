.PHONY: venv install clean test lint format help

PYTHON := python3
VENV_NAME := venv
VENV_ACTIVATE := $(VENV_NAME)/bin/activate
PYTHON_VENV := $(VENV_NAME)/bin/python

help:
	@echo "make venv        - Create a virtual environment"
	@echo "make install     - Install production dependencies"
	@echo "make test        - Run tests"
	@echo "make lint        - Run linting checks"
	@echo "make format      - Format code"
	@echo "make clean       - Clean up build artifacts and virtual environment"

venv:
	$(PYTHON) -m venv $(VENV_NAME)

install: venv
	. $(VENV_ACTIVATE) && pip install -U pip setuptools wheel && pip install -e .

test: dev-install
	. $(VENV_ACTIVATE) && pytest --cov=your_project_name tests/

lint: dev-install
	. $(VENV_ACTIVATE) && flake8 your_project_name tests
	. $(VENV_ACTIVATE) && mypy your_project_name

format: dev-install
	. $(VENV_ACTIVATE) && black your_project_name tests
	. $(VENV_ACTIVATE) && isort your_project_name tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf $(VENV_NAME)
	find . -type d -name __pycache__ -exec rm -rf {} +