#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_PACKAGES_DIR="${ROOT_DIR}/.venv-k230/lib/python3.8/site-packages"
NNCASE_PLUGIN_DIR="${SITE_PACKAGES_DIR}/nncase/modules/kpu"

export DOTNET_ROOT="${ROOT_DIR}/.dotnet"
export PATH="${DOTNET_ROOT}:${ROOT_DIR}/.venv-k230/bin:${SITE_PACKAGES_DIR}:${PATH}"
export LD_LIBRARY_PATH="${SITE_PACKAGES_DIR}:${LD_LIBRARY_PATH:-}"
export NNCASE_PLUGIN_PATH="${NNCASE_PLUGIN_DIR}"
