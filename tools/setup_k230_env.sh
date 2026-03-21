#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-k230"
DOTNET_DIR="${ROOT_DIR}/.dotnet"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"
NNCASE_VERSION="${NNCASE_VERSION:-2.8.3}"

install_local_dotnet() {
    if [ -x "${DOTNET_DIR}/dotnet" ]; then
        return 0
    fi

    echo "Installing local .NET 7 runtime into ${DOTNET_DIR}..."
    mkdir -p "${DOTNET_DIR}"
    "${PYTHON_BIN}" - <<PY
from urllib.request import urlopen

url = 'https://dot.net/v1/dotnet-install.sh'
with urlopen(url, timeout=60) as response:
    content = response.read().decode('utf-8')
with open('/tmp/dotnet-install.sh', 'w', encoding='utf-8') as handle:
    handle.write(content)
PY
    bash /tmp/dotnet-install.sh --channel 7.0 --install-dir "${DOTNET_DIR}"
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

install_local_dotnet

if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
    echo "Standard venv bootstrap failed. Falling back to virtualenv..."
    "${PYTHON_BIN}" -m pip install --user virtualenv
    "${PYTHON_BIN}" -m virtualenv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/tools/requirements-k230.txt"
"${VENV_DIR}/bin/pip" install "nncase==${NNCASE_VERSION}" "nncase-kpu==${NNCASE_VERSION}"

cat <<EOF
K230 conversion environment is ready.

Load the runtime environment with:
source ${ROOT_DIR}/tools/activate_k230_env.sh

Optional: activate the Python virtual environment too:
source ${VENV_DIR}/bin/activate
EOF
