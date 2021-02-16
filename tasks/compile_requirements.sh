#!/bin/bash

export CUSTOM_COMPILE_COMMAND="`basename "$0"`"
path="`dirname "$0"`"
cd "$path/../requirements"
pip-compile --find-links=https://download.pytorch.org/whl/torch_stable.html requirements.in
pip-compile --find-links=https://download.pytorch.org/whl/torch_stable.html requirements-dev.in