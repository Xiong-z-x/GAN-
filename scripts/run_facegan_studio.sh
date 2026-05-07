#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host="${FACEGAN_HOST:-0.0.0.0}"
port="${FACEGAN_PORT:-7860}"

cd "${project_root}"

export FACEGAN_PROJECT_ROOT="${project_root}"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"

if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo >/dev/null 2>&1 || true
fi

python -m facegan_studio.app \
  --project-root "${project_root}" \
  --host "${host}" \
  --port "${port}"
