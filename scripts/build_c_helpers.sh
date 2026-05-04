#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/build"
mkdir -p "$OUT_DIR"

UNAME="$(uname -s)"
if [[ "$UNAME" == "Darwin" ]]; then
  OUT_FILE="$OUT_DIR/libvoice_mixer.dylib"
  cc -O3 -std=c11 -dynamiclib "$ROOT_DIR/c_src/voice_mixer.c" -o "$OUT_FILE"
else
  OUT_FILE="$OUT_DIR/libvoice_mixer.so"
  cc -O3 -std=c11 -fPIC -shared "$ROOT_DIR/c_src/voice_mixer.c" -o "$OUT_FILE"
fi

echo "Built $OUT_FILE"

