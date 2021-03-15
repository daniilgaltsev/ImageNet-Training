#!/bin/bash

FAILURE=false

CUSTOM_COMPILE_COMMAND=$(basename "$0")
export CUSTOM_COMPILE_COMMAND

path=$(dirname "$0")
cd "$path/../requirements" || { echo "cd failed"; exit 1; }

echo "Compiling prod requirements"
pip-compile --find-links=https://download.pytorch.org/whl/torch_stable.html requirements.in || FAILURE=true
echo "Compiling dev requirements"
pip-compile requirements-dev.in || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Compiling failed"
  exit 1
fi

echo "Compiling done"
exit 0