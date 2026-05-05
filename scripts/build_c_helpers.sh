#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/build"
mkdir -p "$OUT_DIR"

UNAME="$(uname -s)"
SRC_FILE=""
if [[ -f "$ROOT_DIR/c_src/voice_mixer.cpp" ]]; then
  SRC_FILE="$ROOT_DIR/c_src/voice_mixer.cpp"
else
  SRC_FILE="$ROOT_DIR/c_src/voice_mixer.c"
fi
if [[ "$UNAME" == "Darwin" ]]; then
  OUT_FILE="$OUT_DIR/libvoice_mixer.dylib"
  c++ -O3 -std=c++17 -dynamiclib "$SRC_FILE" -o "$OUT_FILE"
else
  OUT_FILE="$OUT_DIR/libvoice_mixer.so"
  c++ -O3 -std=c++17 -fPIC -shared "$SRC_FILE" -o "$OUT_FILE"
fi

echo "Built $OUT_FILE"

