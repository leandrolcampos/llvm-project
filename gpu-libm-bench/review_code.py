#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Inspect final SASS, then record a human/agent code review bound to its hash."""
import argparse
import datetime
import pathlib
import re
import subprocess

from artifacts import REVIEW_CHECKS, atomic_json, product_lock, verify_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=pathlib.Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--streams", help="comma-separated subset; default all compiled K")
    p.add_argument("--nvdisasm", type=pathlib.Path, default=pathlib.Path("/usr/local/cuda/bin/nvdisasm"))
    p.add_argument("--record-review", action="store_true")
    p.add_argument("--reviewer", default="")
    a = p.parse_args()
    with product_lock(a.manifest.parent, exclusive=a.record_review):
        m = verify_manifest(a.manifest, [a.candidate], require_review=False)
        ks = list(map(int, a.streams.split(","))) if a.streams else m["streams"]
        if not set(ks) <= set(m["streams"]):
            p.error("K not compiled")
        product = next(x for x in m["products"] if x["id"] == a.candidate)
        if not a.record_review:
            sass = subprocess.check_output([str(a.nvdisasm), "--print-code",
                                           str(a.manifest.parent / a.candidate / "kernel.cubin")], text=True)
            for k in ks:
                match = re.search(rf"\.section\s+\.text\.expf_k{k},.*?(?=\n\s*\.section|\Z)", sass, re.S)
                if not match:
                    raise ValueError(f"missing expf_k{k} in final cubin")
                print(match[0])
            print(product["ptxas"])
            print("Review required: " + ", ".join(REVIEW_CHECKS))
            print("The IR checks and checksum smoke are not proof of executed scalar counts.")
        else:
            if not a.reviewer.strip():
                p.error("--record-review requires --reviewer")
            old = product.get("inspection", {})
            previous = old.get("streams", []) if old.get("cubin_sha256") == product["cubin_sha256"] else []
            product["inspection"] = dict(cubin_sha256=product["cubin_sha256"],
                                         architecture=m["architecture"], streams=sorted(set(ks + previous)),
                                         reviewer=a.reviewer, checks=REVIEW_CHECKS,
                                         utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            atomic_json(a.manifest, m)
            print(f"Recorded review by {a.reviewer}: {a.candidate}, K={ks}, {product['cubin_sha256']}")


if __name__ == "__main__":
    main()
