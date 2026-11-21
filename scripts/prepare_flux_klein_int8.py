"""Fetch the pinned distilled checkpoint, reusing verified local shared files.

This script downloads weights only. The reference runner performs INT8 export.
Existing Base files are never overwritten. No editing pipeline is executed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path, *, git_blob: bool = False) -> str:
    """Hash a file without loading its tensor payload into memory."""
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ranged_download(model_id: str, revision: str, relative: str, size: int, target: Path) -> None:
    """Use bounded HTTP ranges when a full-size CDN response stalls."""
    import httpx

    parts = target.parent / ".range-download"
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    ranges = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]

    def fetch(bounds: tuple[int, int]) -> Path:
        start, end = bounds
        part = parts / f"{start:012d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/{relative}?range_start={start}"
        with httpx.stream("GET", url, headers={"Range": f"bytes={start}-{end}"},
                          follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get("content-range") != f"bytes {start}-{end}/{size}":
                raise RuntimeError(f"Server did not honor requested byte range: {start}-{end}")
            with part.open("wb") as stream:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    stream.write(chunk)
        if part.stat().st_size != end - start + 1:
            raise RuntimeError(f"Incomplete byte range: {start}-{end}")
        return part

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, bounds) for bounds in ranges]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 32 == 0 or i == len(ranges):
                print(f"Downloaded byte ranges: {i}/{len(ranges)}", flush=True)
    temporary = target.with_suffix(target.suffix + ".assembling")
    with temporary.open("wb") as destination:
        for start, _ in ranges:
            part = parts / f"{start:012d}.part"
            with part.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    destination.write(chunk)
    os.replace(temporary, target)
    # Only delete individual temporary files created by this downloader.
    for start, _ in ranges:
        (parts / f"{start:012d}.part").unlink()
    parts.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/flux-klein-4b-int8.json")
    parser.add_argument("--http-range", action="store_true", help="Use resumable bounded ranges for large weights")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/flux-int8/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download

    source = ROOT / cfg["source_dir"]
    reuse = ROOT / cfg["reuse_dir"]
    source.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=False).model_info(cfg["model_id"], revision=cfg["revision"], files_metadata=True)
    if info.sha != cfg["revision"]:
        raise ValueError("Resolved revision differs from the pinned commit")
    files = [f for f in info.siblings if f.rfilename.startswith(
        ("transformer/", "text_encoder/", "tokenizer/", "vae/", "scheduler/")
    ) or f.rfilename in ("model_index.json", "README.md", "LICENSE.md", ".gitattributes")]
    manifest = []
    pending = []
    for item in files:
        rel = item.rfilename
        expected = item.lfs.sha256 if item.lfs else item.blob_id
        target = source / rel
        candidate = target if target.exists() else reuse / rel
        matched = (candidate.is_file() and candidate.stat().st_size == item.size
                   and digest(candidate, git_blob=item.lfs is None) == expected)
        if matched:
            if candidate != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(candidate, target)
            print(f"Verified/reused {rel}", flush=True)
        else:
            if target.exists():
                raise ValueError(f"Existing destination does not match pinned weights: {target}")
            pending.append(rel)
        manifest.append({"path": rel, "bytes": item.size,
                         "hash_type": "sha256" if item.lfs else "git-blob-sha1",
                         "hash": expected, "reused": matched})
    if pending:
        ranged = [item for item in manifest if item["path"] in pending and item["bytes"] > 1024**3
                  and args.http_range]
        small = [name for name in pending if name not in {item["path"] for item in ranged}]
        if small:
            snapshot_download(cfg["model_id"], revision=cfg["revision"], local_dir=source,
                              allow_patterns=small, token=False, max_workers=4)
        for item in ranged:
            ranged_download(cfg["model_id"], cfg["revision"], item["path"], item["bytes"],
                            source / item["path"])
    for item in manifest:
        path = source / item["path"]
        if path.stat().st_size != item["bytes"] or digest(
            path, git_blob=item["hash_type"] == "git-blob-sha1"
        ) != item["hash"]:
            raise ValueError(f"Downloaded file integrity check failed: {path}")
    index = json.loads((source / "model_index.json").read_text())
    if index.get("is_distilled") is not True:
        raise ValueError("The selected model is not marked as distilled")
    (source / "source_manifest.json").write_text(json.dumps({
        "model_id": cfg["model_id"], "revision": info.sha, "files": manifest,
        "shared_files": "Verified identical files are hardlinked; never modify weights in place.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Verified distilled source ready: {source}", flush=True)


if __name__ == "__main__":
    main()
