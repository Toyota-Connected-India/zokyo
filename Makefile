SHELL := /bin/bash -e

WORKDIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

PYTHON ?= python

GIT_HASH = $(shell git rev-parse --verify HEAD)
GIT_TAG := $(shell git describe --tags --exact-match $(GIT_HASH) 2>/dev/null)

PYTHON_VERSION := py3$(shell python -V | cut -c 10)

GITHUB_SPHINX_BASE ?= github-sphinx

.PHONY: aws-login
aws-login: ${CRED_FILE}
	eval $$(aws ecr get-login --region $(AWS_ECR_REGION) --no-include-email)

.PHONY: test-lint
test-lint: clean-py
	docker run --rm -t \
		-v `pwd`:/app \
		-w /app python:3.7-alpine \
		/bin/ash -c "pip install flake8; python -m flake8 sphinx --filename='*.py' --ignore E803,F401,F403,W293,W504,F541"

.PHONY: test-unit
test-unit: test-requirements
	$(PYTHON) -m pytest -v --durations=20 --cov-config .coveragerc --cov sphinx -p no:logging

.PHONY: test
test: test-unit test-lint
	rm -rf .coverage.* || echo "No coverage artifacts to remove"

.PHONY: report-coverage
report-coverage:
	codecov -t $(CODECOV_TOKEN)

.PHONY: test-requirements
test-requirements:
	$(PYTHON) -m pip install --user pytest pytest-cov flake8 matplotlib seaborn Jinja2 requests-mock

.PHONY: install
install:
	$(PYTHON) -m pip install .

.PHONY: VERSION
VERSION:
ifneq ($(GIT_TAG),)
	@echo $(GIT_TAG) > sphinx/VERSION
else
	@echo "Not a tagged commit, will not write to VERSION file"
endif

.PHONY: sdist
sdist: aws-login VERSION
ifneq ($(GITLAB_CI),)
	chmod -R 777 .
endif
	docker run --rm -t \
		-v `pwd`:/app \
		-w /app \
		$(PREFIX)/$(GITHUB_SPHINX_BASE):latest \
		python setup.py sdist

.PHONY: clean-py
clean-py:
	$(shell find . -name \*.pyc -o -name \*.pyo -o -name __pycache__ -delete)

.PHONY: clean
clean: clean-py
	rm -rf build
	rm -rf dist
	rm -rf sphinx.egg-info
	rm -rf .eggs
