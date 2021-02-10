export CUSTOM_COMPILE_COMMAND="`basename "$0"`"
path="`dirname "$0"`"
cd "$path/.."
pip-compile --find-links=https://download.pytorch.org/whl/torch_stable.html -q requirements.in
pip-compile --find-links=https://download.pytorch.org/whl/torch_stable.html -q requirements-dev.in