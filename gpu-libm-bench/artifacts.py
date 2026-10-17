# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Product integrity and cooperative build/measurement exclusion (Linux)."""
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import tempfile

SCHEMA = 2
REVIEW_CHECKS = ["math-in-loop", "argument-dependencies", "observable-results",
                 "scalar-count", "register-states"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = pathlib.Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        pathlib.Path(name).unlink(missing_ok=True)


@contextlib.contextmanager
def product_lock(root, exclusive=False):
    root = pathlib.Path(root)
    if exclusive:
        root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+b") as f:
        try:
            fcntl.flock(f, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("device products are in use by a build or collection") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def build_session(root):
    """Never leave an old manifest describing partially overwritten products."""
    root = pathlib.Path(root)
    with product_lock(root, exclusive=True):
        path = root / "manifest.json"
        previous = json.loads(path.read_text()) if path.exists() else None
        (root / "manifest.json").unlink(missing_ok=True)
        yield previous


def prepare_variant(path, configuration):
    """A removed option or a replaced compiler must not survive in the cache."""
    path = pathlib.Path(path)
    marker = path / ".configuration.json"
    old = json.loads(marker.read_text()) if marker.exists() else None
    if path.exists() and old != configuration:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, configuration)


def verify_manifest(path, candidates=None, streams=None, require_review=True):
    path = pathlib.Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported/incomplete device manifest; rebuild")
    products = {p["id"]: p for p in manifest["products"]}
    ids = candidates if candidates is not None else [c["id"] for c in manifest["candidates"]]
    ks = set(streams if streams is not None else manifest["streams"])
    if not ks <= set(manifest["streams"]):
        raise ValueError("stream count missing from device build")
    for ident in ids:
        if ident not in products:
            raise ValueError(f"missing product: {ident}")
        p = products[ident]
        cubin = path.parent / ident / "kernel.cubin"
        if not cubin.is_file() or digest(cubin) != p["cubin_sha256"]:
            raise ValueError(f"{ident}: cubin missing or hash differs from manifest; rebuild")
        if require_review:
            review = p.get("inspection", {})
            if (review.get("cubin_sha256") != p["cubin_sha256"] or
                    review.get("architecture") != manifest["architecture"] or
                    not ks <= set(review.get("streams", [])) or
                    set(review.get("checks", [])) != set(REVIEW_CHECKS) or
                    not review.get("reviewer")):
                raise ValueError(f"{ident}: final cubin inspection required; use review_code.py")
    return manifest
