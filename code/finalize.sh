#!/usr/bin/env bash
# Original full pipeline: finalize_legacy.sh. Revised entry preserves evidence captions.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/rebuild_narrative.sh" "$@"
