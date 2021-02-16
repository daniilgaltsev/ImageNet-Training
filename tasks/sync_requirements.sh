#!/bin/bash

FAILURE=false

CUSTOM_COMPILE_COMMAND=$(basename "$0")
export CUSTOM_COMPILE_COMMAND

path=$(dirname "$0")
cd "$path/../requirements" || { echo "cd failed"; exit 1; }

echo "Syncing requirements"
pip-sync requirements.txt requirements-dev.txt || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Syncing failed"
  exit 1
fi

echo "Syncing done"
exit 0