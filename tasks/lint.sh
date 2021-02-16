#!/bin/bash

set -uo pipefail
set +e

FAILURE=false

path=$(dirname "$0")
cd "$path/.." || { echo "cd failed"; exit 1; }

echo "Running safety check"
safety check -r requirements/requirements.txt -r requirements/requirements-dev.txt || FAILURE=true

echo "Running pylint"
pylint imagenet_training || FAILURE=true

echo "Running pycodestyle"
pycodestyle imagenet_training || FAILURE=true

echo "Running pydocstyle"
pydocstyle imagenet_training || FAILURE=true

echo "Running mypy"
mypy imagenet_training || FAILURE=true

echo "Running bandit"
bandit -ll -r imagenet_training || FAILURE=true

echo "shellcheck"
shellcheck tasks/*.sh || FAILURE=true

sleep 10

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi

echo "Linting passed"
exit 0