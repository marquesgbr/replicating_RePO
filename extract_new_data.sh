#!/usr/bin/env bash
#
# Extract the processed new dataset into data/new_data/.
#
# The archive is stored with Git LFS.  The script pulls it with git-lfs when
# available; when git-lfs is missing (or the checkout still holds a pointer),
# it falls back to cluster/lfs_fetch.py, which downloads the object through
# the GitHub LFS batch API and verifies its size and SHA-256.
#
# `OUTPUT_DIR` and `ZIP_PATH` make the extraction reusable by the Apuana
# setup, which keeps the CSVs in $DATA_ROOT and exposes them by symlink.
#
# Usage:
#   bash extract_new_data.sh
#   OUTPUT_DIR="$DATA_ROOT/new_data" bash extract_new_data.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/new_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}}"
ZIP_PATH="${ZIP_PATH:-${DATA_DIR}/New-dataset-split-clean.zip}"
TRAIN_CSV="${OUTPUT_DIR}/new_train.csv"
TEST_CSV="${OUTPUT_DIR}/new_test.csv"

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
command -v unzip >/dev/null 2>&1 || die "unzip is required but was not found in PATH."

mkdir -p "$OUTPUT_DIR"

# Path relative to the repository, used by `git lfs pull --include`.
ZIP_REL=""
case "$ZIP_PATH" in
    "$REPO_ROOT"/*) ZIP_REL="${ZIP_PATH#"$REPO_ROOT"/}" ;;
    /*) ZIP_REL="" ;;
    *) ZIP_REL="$ZIP_PATH" ;;
esac

ARCHIVE_PATH="$ZIP_PATH"

if [[ -f "$ZIP_PATH" ]] && ! is_lfs_pointer "$ZIP_PATH"; then
    log "Archive already materialized: ${ZIP_PATH}"
else
    if [[ -n "$ZIP_REL" ]] && git lfs version >/dev/null 2>&1; then
        log "Pulling ${ZIP_REL} with Git LFS..."
        git lfs pull --include="${ZIP_REL}" || true
    fi

    if [[ ! -f "$ZIP_PATH" ]]; then
        die "Expected archive not found: ${ZIP_PATH}"
    fi

    if is_lfs_pointer "$ZIP_PATH"; then
        log "git-lfs unavailable or pointer remains; using the API fallback..."
        # Never overwrite the tracked pointer: download beside the extracted
        # CSVs so `git status` cannot stage the 813 MB binary by accident.
        FALLBACK_ZIP="${OUTPUT_DIR}/$(basename "$ZIP_PATH")"
        if command -v realpath >/dev/null 2>&1 \
           && [[ "$(realpath -m -- "$FALLBACK_ZIP")" == "$(realpath -m -- "$ZIP_PATH")" ]]; then
            FALLBACK_ZIP="${OUTPUT_DIR}/New-dataset-split-clean.lfs.zip"
        fi
        LFS_ARGS=(--pointer "$ZIP_PATH" --output "$FALLBACK_ZIP")
        if [[ -n "${LFS_REPOSITORY:-}" ]]; then
            LFS_ARGS+=(--repository "$LFS_REPOSITORY")
        fi
        "${PYTHON_BIN}" "${REPO_ROOT}/cluster/lfs_fetch.py" "${LFS_ARGS[@]}" \
            || die "could not download the LFS object; check network access, copy the real ZIP into ${OUTPUT_DIR}, or run setup with --skip-new-data"
        ARCHIVE_PATH="$FALLBACK_ZIP"
    fi
fi

if [[ ! -f "$ARCHIVE_PATH" ]]; then
    die "Expected archive not found: ${ARCHIVE_PATH}"
fi
if is_lfs_pointer "$ARCHIVE_PATH"; then
    die "${ARCHIVE_PATH} is still a Git LFS pointer."
fi

log "Extracting ${ARCHIVE_PATH} into ${OUTPUT_DIR}/..."
unzip -o "$ARCHIVE_PATH" -d "${OUTPUT_DIR}/"

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
