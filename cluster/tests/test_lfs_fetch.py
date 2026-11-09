from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

CLUSTER_DIR = Path(__file__).resolve().parents[1]
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import lfs_fetch  # noqa: E402


def pointer_text(payload: bytes) -> str:
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n"
        f"size {len(payload)}\n"
    )


def test_parse_pointer_reads_oid_and_size():
    payload = b"dataset"
    pointer = lfs_fetch.parse_pointer(pointer_text(payload))
    assert pointer.oid == hashlib.sha256(payload).hexdigest()
    assert pointer.size == len(payload)


@pytest.mark.parametrize(
    "text",
    [
        "not a pointer\n",
        "version https://git-lfs.github.com/spec/v1\noid sha256:xyz\nsize 1\n",
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\nsize 0\n",
        "version https://git-lfs.github.com/spec/v1\nsize 10\n",
    ],
)
def test_parse_pointer_rejects_invalid_documents(text):
    with pytest.raises(ValueError):
        lfs_fetch.parse_pointer(text)


def test_normalize_repository_accepts_ssh_and_https():
    assert (
        lfs_fetch.normalize_repository("git@github.com:o/r.git")
        == "https://github.com/o/r.git"
    )
    assert (
        lfs_fetch.normalize_repository("ssh://git@github.com/o/r.git")
        == "https://github.com/o/r.git"
    )
    assert (
        lfs_fetch.normalize_repository("https://github.com/o/r.git/")
        == "https://github.com/o/r.git"
    )


def test_normalize_repository_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        lfs_fetch.normalize_repository("/local/path")


def test_batch_payload_declares_download():
    payload = lfs_fetch.batch_payload("a" * 64, 10)
    assert payload["operation"] == "download"
    assert payload["transfers"] == ["basic"]
    assert payload["objects"] == [{"oid": "a" * 64, "size": 10}]


def test_download_object_writes_the_verified_object(monkeypatch, tmp_path):
    payload = b"the real dataset"
    pointer = lfs_fetch.parse_pointer(pointer_text(payload))
    requested: list[str] = []

    def fake_urlopen(request):
        requested.append(request.full_url)
        if request.full_url.endswith("/info/lfs/objects/batch"):
            body = {
                "objects": [
                    {
                        "oid": pointer.oid,
                        "size": pointer.size,
                        "actions": {
                            "download": {"href": "https://example.test/object"}
                        },
                    }
                ]
            }
            return io.BytesIO(json.dumps(body).encode("utf-8"))
        return io.BytesIO(payload)

    monkeypatch.setattr(lfs_fetch, "_urlopen", fake_urlopen)
    output = tmp_path / "new_data" / "New-dataset-split-clean.zip"

    lfs_fetch.download_object(
        "https://github.com/o/r.git", pointer, output
    )

    assert output.read_bytes() == payload
    assert requested[0] == "https://github.com/o/r.git/info/lfs/objects/batch"
    assert requested[1] == "https://example.test/object"


def test_download_object_rejects_a_digest_mismatch(monkeypatch, tmp_path):
    payload = b"the real dataset"
    pointer = lfs_fetch.parse_pointer(pointer_text(payload))

    def fake_urlopen(request):
        if request.full_url.endswith("/info/lfs/objects/batch"):
            body = {
                "objects": [
                    {
                        "oid": pointer.oid,
                        "size": pointer.size,
                        "actions": {
                            "download": {"href": "https://example.test/object"}
                        },
                    }
                ]
            }
            return io.BytesIO(json.dumps(body).encode("utf-8"))
        return io.BytesIO(b"corrupted")

    monkeypatch.setattr(lfs_fetch, "_urlopen", fake_urlopen)
    output = tmp_path / "object.zip"

    with pytest.raises(RuntimeError, match="size|SHA-256"):
        lfs_fetch.download_object("https://github.com/o/r.git", pointer, output)

    assert not output.exists()


def test_download_object_reports_a_missing_action(monkeypatch, tmp_path):
    pointer = lfs_fetch.LfsPointer(oid="a" * 64, size=10)

    def fake_urlopen(request):
        return io.BytesIO(json.dumps({"objects": [{"oid": pointer.oid}]}).encode())

    monkeypatch.setattr(lfs_fetch, "_urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="download action"):
        lfs_fetch.download_object(
            "https://github.com/o/r.git", pointer, tmp_path / "object.zip"
        )
