#!/usr/bin/env bash
#
# Extract the processed new dataset into data/new_data/.
#
# The archive (data/new_data/New-dataset-split-clean.zip) is stored with Git
# LFS, so the real file has to be pulled before it can be unzipped.  The
# script pulls the archive, unzips it and verifies that the two CSVs exist,
# are not LFS pointers and reports their row count and SHA-256.
#
# Usage: bash extract_new_data.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATA_DIR="data/new_data"
ZIP_PATH="${DATA_DIR}/New-dataset-split-clean.zip"
TRAIN_CSV="${DATA_DIR}/new_train.csv"
TEST_CSV="${DATA_DIR}/new_test.csv"

log() {
    printf '%s\n' "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

# A Git LFS pointer is a small text file starting with the LFS spec version.
is_lfs_pointer() {
    local file="$1"
    head -c 200 "$file" 2>/dev/null | grep -q "git-lfs.github.com/spec"
}

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    die "python is required to count rows and hash the extracted CSVs."
fi

command -v git >/dev/null 2>&1 || die "git is required but was not found in PATH."
if ! git lfs version >/dev/null 2>&1; then
    die "git-lfs is required but is not available. Install it (https://git-lfs.com), run 'git lfs install' and try again."
fi
command -v unzip >/dev/null 2>&1 || die "unzip is required but was not found in PATH."

log "Pulling ${ZIP_PATH} with Git LFS..."
git lfs pull --include="${ZIP_PATH}"

[ -f "$ZIP_PATH" ] || die "Expected archive not found after 'git lfs pull': ${ZIP_PATH}"
if is_lfs_pointer "$ZIP_PATH"; then
    die "${ZIP_PATH} is still a Git LFS pointer. Check your network access and run 'git lfs pull' again."
fi

log "Extracting ${ZIP_PATH} into ${DATA_DIR}/..."
unzip -o "$ZIP_PATH" -d "${DATA_DIR}/"

for csv in "$TRAIN_CSV" "$TEST_CSV"; do
    [ -f "$csv" ] || die "Expected CSV not found after extraction: ${csv}"
    if is_lfs_pointer "$csv"; then
        die "${csv} is a Git LFS pointer instead of a real CSV file."
    fi
done

summarize() {
    local csv="$1"
    "${PYTHON_BIN}" - "$csv" <<'PY'
import csv
import hashlib
import sys

path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as raw:
    for block in iter(lambda: raw.read(1 << 20), b""):
        digest.update(block)

with open(path, newline="") as handle:
    reader = csv.reader(handle)
    next(reader, None)  # header
    rows = sum(1 for _ in reader)

print(f"  rows   = {rows}")
print(f"  sha256 = {digest.hexdigest()}")
PY
}

log ""
log "Extracted files:"
log "  ${TRAIN_CSV}"
summarize "$TRAIN_CSV"
log "  ${TEST_CSV}"
summarize "$TEST_CSV"
log ""
log "Next: validate headers, labels and metadata with the new_data_eval package."
