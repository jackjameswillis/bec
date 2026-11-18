#!/usr/bin/env bash
# LD_LIBRARY_PATH wrapper for TF GPU support in the ste venv.
# The pip nvidia wheels (nvidia-*-cu12) are installed under
# site-packages/nvidia/*/lib which is outside the dynamic-loader search
# path, so TF must be started with those dirs preloaded.
NV=/data/ste/venv/lib/python3.11/site-packages/nvidia
DIRS=$(ls -d "${NV}"/*/lib "${NV}"/*/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${DIRS%:}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$@"
