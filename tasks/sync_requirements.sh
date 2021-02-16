#!/bin/bash

export CUSTOM_COMPILE_COMMAND="`basename "$0"`"
path="`dirname "$0"`"
cd "$path/../requirements"
pip-sync requirements.txt requirements-dev.txt