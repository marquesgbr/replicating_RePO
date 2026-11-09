#!/usr/bin/env python3
"""Download a single Git LFS object without requiring ``git-lfs``.

The Apuana login nodes may not have ``git-lfs`` installed.  When the tracked
``New-dataset-split-clean.zip`` is still an LFS pointer, this helper talks to
the repository's Git LFS batch API directly, downloads the object and verifies
its size and SHA-256 before replacing the pointer atomically.

Only the standard library is used, so this runs with the system Python before
the cluster virtualenv exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

LFS_MEDIA_TYPE = "application/vnd.git-lfs+json"
LFS_VERSION_PREFIX = "version https://git-lfs.github.com/spec/"
_OID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1 << 20

_urlopen = urllib.request.urlopen


@dataclass(frozen=True)
class LfsPointer:
    """The ``oid`` and ``size`` fields of a Git LFS pointer file."""

    oid: str
    size: int


def parse_pointer(text: str) -> LfsPointer:
    """Parse and validate a Git LFS pointer document."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(LFS_VERSION_PREFIX):
        raise ValueError("not a Git LFS pointer: missing spec version line")

    fields: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition(" ")
        if not separator:
            raise ValueError(f"malformed Git LFS pointer line: {line!r}")
        fields[key] = value.strip()

    oid_field = fields.get("oid", "")
    prefix, separator, digest = oid_field.partition(":")
    if prefix != "sha256" or not separator:
        raise ValueError(f"unsupported Git LFS oid: {oid_field!r}")
    if not _OID_PATTERN.fullmatch(digest):
        raise ValueError(f"invalid Git LFS sha256 oid: {digest!r}")

    size_field = fields.get("size", "")
    if not size_field.isdigit() or int(size_field) <= 0:
        raise ValueError(f"invalid Git LFS size: {size_field!r}")

    return LfsPointer(oid=digest, size=int(size_field))


def normalize_repository(url: str) -> str:
    """Return an HTTPS repository URL from an SSH or HTTPS remote."""
    value = url.strip()
    if not value:
        raise ValueError("empty repository URL")

    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
        return f"https://{host}/{path}".rstrip("/")

    if value.startswith("ssh://"):
        parsed = urlsplit(value)
        if parsed.hostname is None or not parsed.path.strip("/"):
            raise ValueError(f"cannot normalize repository URL: {url!r}")
        return f"https://{parsed.hostname}/{parsed.path.lstrip('/')}".rstrip("/")

    if value.startswith(("https://", "http://")):
        return value.rstrip("/")

    raise ValueError(f"cannot normalize repository URL: {url!r}")


def batch_url(repository: str) -> str:
    """Return the batch endpoint of ``repository``."""
    return f"{normalize_repository(repository)}/info/lfs/objects/batch"


def batch_payload(oid: str, size: int) -> dict[str, object]:
    """Return the JSON body that requests one LFS download action."""
    return {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": int(size)}],
    }


def _headers(token: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": LFS_MEDIA_TYPE,
        "Content-Type": LFS_MEDIA_TYPE,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _remote_url() -> str:
    try:
        completed = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "cannot determine the repository URL; pass --repository or run "
            "inside the clone"
        ) from exc
    return normalize_repository(completed.stdout)


def download_object(
    repository: str,
    pointer: LfsPointer,
    output: Path,
    *,
    token: str | None = None,
) -> Path:
    """Download ``pointer`` from ``repository`` into ``output`` atomically."""
    payload = json.dumps(batch_payload(pointer.oid, pointer.size)).encode("utf-8")
    request = urllib.request.Request(
        batch_url(repository),
        data=payload,
        headers=_headers(token),
        method="POST",
    )
    with _urlopen(request) as response:
        batch = json.loads(response.read().decode("utf-8"))

    objects = batch.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError("Git LFS batch response did not contain an object")
    entry = objects[0]
    if not isinstance(entry, dict):
        raise RuntimeError("unexpected Git LFS batch response object")
    if "error" in entry:
        raise RuntimeError(f"Git LFS object error: {entry['error']}")

    actions = entry.get("actions")
    download = actions.get("download") if isinstance(actions, dict) else None
    if not isinstance(download, dict) or not download.get("href"):
        raise RuntimeError(
            "Git LFS batch response has no download action; check repository "
            "permissions and that the object exists"
        )

    download_headers = download.get("header")
    if not isinstance(download_headers, dict):
        download_headers = {}
    request = urllib.request.Request(
        str(download["href"]), headers=dict(download_headers)
    )

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".lfs", dir=str(target.parent)
    )
    os.close(descriptor)
    digest = hashlib.sha256()
    written = 0
    try:
        with _urlopen(request) as response, open(temporary, "wb") as handle:
            while True:
                block = response.read(_CHUNK)
                if not block:
                    break
                handle.write(block)
                digest.update(block)
                written += len(block)

        if written != pointer.size:
            raise RuntimeError(
                f"downloaded size does not match the pointer: "
                f"{written} != {pointer.size}"
            )
        if digest.hexdigest() != pointer.oid:
            raise RuntimeError(
                "downloaded SHA-256 does not match the pointer: "
                f"{digest.hexdigest()} != {pointer.oid}"
            )
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download one Git LFS object without git-lfs."
    )
    parser.add_argument("--pointer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository", default=None)
    args = parser.parse_args(argv)

    try:
        pointer = parse_pointer(args.pointer.read_text(encoding="utf-8"))
        repository = args.repository or _remote_url()
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GIT_LFS_TOKEN")
        download_object(repository, pointer, args.output, token=token)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"downloaded {args.output} ({pointer.size} bytes, sha256 {pointer.oid})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
