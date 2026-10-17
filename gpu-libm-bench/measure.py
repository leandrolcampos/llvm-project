#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""NVBench external rounds and paired, whole-round bootstrap (standard library)."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import tempfile

from build_device import registry
from artifacts import digest, product_lock, verify_manifest


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo = int(x)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (x - lo)


def point(row):
    return (row["id"], *(int(row[x]) for x in ("B", "T", "K", "R")))


def summarize(rows, seed=3, resamples=10000, expected_rounds=None, expected_points=None):
    """One shared index resample of complete external rounds for all points."""
    if resamples < 1000:
        raise ValueError("use at least 1000 bootstrap resamples")
    rounds = (list(range(expected_rounds)) if expected_rounds is not None
              else sorted({int(row["round"]) for row in rows}))
    data = {tuple(p): {} for p in (expected_points or [])}
    seen = set()
    for row in rows:
        p, r = point(row), int(row["round"])
        values = data.setdefault(p, {})
        if (p, r) in seen or r not in rounds:
            raise ValueError(f"duplicate point/round: {p}, {r}")
        seen.add((p, r))
        if row["status"] == "complete":
            q = float(row["gcalls_s"])
            if not math.isfinite(q) or q <= 0:
                raise ValueError("nonpositive/nonfinite complete throughput")
            values[r] = q
    rng = random.Random(seed)
    indices = [[rng.randrange(len(rounds)) for _ in rounds] for _ in range(resamples)]
    result = {}
    for p, values in data.items():
        if len(rounds) < 10 or set(values) != set(rounds):
            result[p] = {"status": "incomplete external rounds"}
            continue
        v = [values[r] for r in rounds]
        boot = [statistics.median(v[i] for i in sample) for sample in indices]
        result[p] = dict(status="complete", median=statistics.median(v),
                         q25=quantile(v, .25), q75=quantile(v, .75),
                         low=quantile(boot, .025), high=quantile(boot, .975), boot=boot)
    return result


def ratio(a, b):
    if a.get("status") != "complete" or b.get("status") != "complete":
        return None
    values = [x / y for x, y in zip(a["boot"], b["boot"])]
    return a["median"] / b["median"], quantile(values, .025), quantile(values, .975)


def saturation(results, p, epsilon):
    ident, b, t, k, r = p
    if b % 4 or r % 4:
        return False
    pairs = [((ident, b, t, k, r // 2), (ident, b, t, k, r // 4)),
             (p, (ident, b, t, k, r // 2)),
             ((ident, b // 2, t, k, r), (ident, b // 4, t, k, r)),
             (p, (ident, b // 2, t, k, r))]
    for high, low in pairs:
        q = ratio(results.get(high, {}), results.get(low, {}))
        if q is None or q[1] < 1 - epsilon or q[2] > 1 + epsilon:
            return False
    return True


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty measurement table")
    for key in ("seed", "corpus_size", "corpus_fnv1a64", "gpu", "architecture",
                "protocol", "phase", "build_sha256"):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f"mixed experiment metadata: {key}")
    return rows


def analyze(a):
    rows = load(a.table)
    protocol = json.loads(rows[0]["protocol"])
    seed, n, epsilon = (protocol[x] for x in ("bootstrap_seed", "resamples", "epsilon"))
    points = list(itertools.product(protocol["candidates"], protocol["B"], protocol["T"],
                                    protocol["K"], protocol["R"]))
    results = summarize(rows, seed, n, protocol["rounds"], points)
    print("id B T K R | median Gcalls/s | IQR | 95% CI | saturation")
    for p, s in sorted(results.items()):
        if s["status"] != "complete":
            print(*p, '|', s["status"], '| saturation not demonstrated')
            continue
        status = "demonstrated" if saturation(results, p, epsilon) else "not demonstrated"
        print(*p, f"| {s['median']:.7g} | [{s['q25']:.7g}, {s['q75']:.7g}]"
              f" | [{s['low']:.7g}, {s['high']:.7g}] | {status}")
        baseline = results.get((a.baseline, *p[1:]), {})
        q = ratio(s, baseline)
        if q is not None and p[0] != a.baseline:
            print(f"  /{a.baseline}: {q[0]:.6g}, paired 95% CI [{q[1]:.6g}, {q[2]:.6g}]")
    print(f"Whole-round percentile bootstrap: seed={seed}, resamples={n}; epsilon={epsilon}.")
    print("A plateau is operational evidence, not proof of an expf-only bottleneck.")


def positive_grid(text):
    values = [int(v) for v in text.split(",")]
    if not values or len(set(values)) != len(values) or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected distinct positive comma-separated integers")
    return values


def require_driver_release(gpu):
    release = gpu.get("nvidia_driver_version")
    if not isinstance(release, str) or not release.strip():
        raise ValueError("NVIDIA driver release missing; cannot verify platform identity")


def validate_confirmation(pilot, protocol, points, seed, corpus_size, build_hash):
    if pilot[0]["phase"] != "pilot":
        raise ValueError("--pilot must identify a pilot table")
    old = json.loads(pilot[0]["protocol"])
    require_driver_release(old.get("gpu", {}))
    require_driver_release(protocol["gpu"])
    for key in ("epsilon", "batch_seconds", "timeout", "min_samples", "cold_samples", "clock_policy",
                "runner_sha256", "analysis_sha256", "python", "gpu"):
        if old.get(key) != protocol[key]:
            raise ValueError(f"final protocol differs from pilot: {key}")
    if not set(points) <= {point(r) for r in pilot}:
        raise ValueError("final configurations must have been measured in the pilot")
    if (pilot[0]["seed"] != str(seed) or pilot[0]["corpus_size"] != str(corpus_size) or
            pilot[0]["build_sha256"] != build_hash):
        raise ValueError("pilot corpus or device build differs")
    if old["shuffle_seed"] == protocol["shuffle_seed"]:
        raise ValueError("use a fresh shuffle seed for final confirmation")


def collect(a):
    if (a.rounds < 10 or not 0 < a.epsilon < 1 or a.resamples < 1000 or
            not 0 <= a.seed < 2**64 or a.device < 0 or a.min_samples < 1 or a.cold_samples < 1):
        raise ValueError("invalid rounds, epsilon, resamples, seed, device or minimum samples")
    if any(not math.isfinite(x) or x <= 0 for x in (a.batch_seconds, a.timeout)):
        raise ValueError("collection budgets must be finite and positive")
    if not 0 < a.corpus_size <= 2**32 or a.corpus_size & (a.corpus_size - 1):
        raise ValueError("corpus size must be a power of two <= 2^32")
    # This shared lock spans verification, preflight, all measurements and export.
    with product_lock(a.manifest.resolve().parent):
        return collect_locked(a)


def collect_locked(a):
    candidates = registry(a.registry)
    ids = [c["id"] for c in candidates]
    if a.candidates:
        requested = a.candidates.split(",")
        if not set(requested) <= set(ids) or len(set(requested)) != len(requested):
            raise ValueError("unknown or duplicate candidates")
        ids = requested
    points = list(itertools.product(ids, a.B, a.T, a.K, a.R))
    manifest = verify_manifest(a.manifest, ids, a.K)
    if candidates != manifest["registry"]:
        raise ValueError("registry differs from the compiled device build; rebuild")
    if not set(ids) <= {c["id"] for c in manifest["candidates"]}:
        raise ValueError("candidate missing from device build")
    # GPU identity is queried at execution, never at configure/build time.
    gpu = json.loads(subprocess.check_output([str(a.runner.resolve()),
                     f"--device-info={a.device}"], text=True))
    require_driver_release(gpu)
    if not gpu.get("gpu_uuid") or gpu["architecture"] != manifest["architecture"].rstrip("af"):
        raise ValueError("physical GPU/architecture does not match the device build")
    build_hash = digest(a.manifest)
    protocol = dict(rounds=a.rounds, shuffle_seed=a.shuffle_seed, bootstrap_seed=a.bootstrap_seed,
                    resamples=a.resamples, epsilon=a.epsilon, batch_seconds=a.batch_seconds,
                    timeout=a.timeout, min_samples=a.min_samples, cold_samples=a.cold_samples, device=a.device,
                    B=a.B, T=a.T, K=a.K, R=a.R, candidates=ids, clock_policy=a.clock_policy,
                    python=sys.version, gpu=gpu, runner_sha256=digest(a.runner),
                    analysis_sha256=digest(__file__))
    if a.phase == "final":
        if a.pilot is None:
            raise ValueError("final collection requires --pilot")
        validate_confirmation(load(a.pilot), protocol, points, a.seed, a.corpus_size, build_hash)
        protocol["pilot_sha256"] = digest(a.pilot)
    rng = random.Random(a.shuffle_seed)
    schedule = []
    for round_number in range(a.rounds):
        order = points.copy()
        rng.shuffle(order)
        for index, (ident, b, t, k, r) in enumerate(order):
            schedule.append(dict(id=ident, B=b, T=t, K=k, R=r, round=round_number, order=index))
    with a.output.open("x", newline="") as output, tempfile.TemporaryDirectory() as tmp:
        plan = pathlib.Path(tmp) / "schedule.json"
        plan.write_text(json.dumps(schedule))
        command = [str(a.runner.resolve()), "--json-lines", f"--schedule={plan}",
                   f"--registry={a.registry.resolve()}", f"--device-dir={a.manifest.resolve().parent}",
                   f"--expect-uuid={gpu['gpu_uuid']}", f"--expect-arch={gpu['architecture']}",
                   f"--expect-driver={gpu['nvidia_driver_version']}",
                   f"--seed={a.seed}", f"--corpus-size={a.corpus_size}",
                   f"--clock-policy={a.clock_policy}", "--devices", str(a.device),
                   "--batch-target-time", str(a.batch_seconds), "--timeout", str(a.timeout),
                   "--min-samples", str(a.min_samples), "--stopping-criterion", "sample-count",
                   "--target-samples", str(a.cold_samples)]
        writer = None
        received = 0
        result = 0
        fingerprint = None
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            for line in process.stdout:
                if not line.startswith("GPU_LIBM_ROW "):
                    print(line, end="", flush=True)
                    continue
                row = json.loads(line[len("GPU_LIBM_ROW "):])
                if received >= len(schedule):
                    raise ValueError("runner emitted extra states")
                expected = schedule[received]
                if ((point(row), int(row["round"]), int(row["order"])) !=
                        (point(expected), expected["round"], expected["order"])):
                    raise ValueError("runner deviated from the requested schedule; earlier rows preserved")
                if any(row.get(k) != v for k, v in gpu.items()):
                    raise ValueError("GPU/toolchain changed after preflight; earlier rows preserved")
                if int(row["seed"]) != a.seed or int(row["corpus_size"]) != a.corpus_size:
                    raise ValueError("runner changed corpus parameters")
                if fingerprint is not None and row["corpus_fnv1a64"] != fingerprint:
                    raise ValueError("runner changed input bits between states")
                fingerprint = row["corpus_fnv1a64"]
                if row["status"] == "complete":
                    count = math.prod(int(row[x]) for x in ("B", "T", "K", "R"))
                    time = float(row["batch_gpu_seconds"])
                    if (count != int(row["C"]) or not math.isfinite(time) or time <= 0 or
                            not math.isclose(float(row["gcalls_s"]), count/(1e9*time), rel_tol=1e-12) or
                            int(row["launches"]) < int(row["effective_min_samples"]) or
                            time * int(row["launches"]) < a.batch_seconds):
                        raise ValueError("complete state violates work/timing/collection definition")
                elif row["status"] == "incomplete":
                    if result != 1:
                        result = 2
                else:
                    result = 1
                row.update(phase=a.phase, protocol=json.dumps(protocol, sort_keys=True),
                           build_sha256=build_hash, build=json.dumps(manifest, sort_keys=True))
                row = {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict,list)) else v
                       for k,v in row.items()}
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                output.flush()
                os.fsync(output.fileno())
                received += 1
        except KeyboardInterrupt:
            print("Collection interrupted; completed rows preserved.", file=sys.stderr)
            process.terminate()
            return 2
        except BaseException:
            if process.poll() is None:
                process.terminate()
            raise
        finally:
            process.wait()
            process.stdout.close()
        if process.returncode not in (0,2):
            print(f"Runner failed ({process.returncode}); completed rows preserved.", file=sys.stderr)
            return 1
        if received != len(schedule) or process.returncode == 2 or result == 2:
            print("Incomplete collection; absent states remain incomplete in analysis.")
            return 1 if result == 1 else 2
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--runner", type=pathlib.Path, required=True)
    p.add_argument("--manifest", type=pathlib.Path, required=True)
    p.add_argument("--registry", type=pathlib.Path, default=pathlib.Path(__file__).with_name("candidates.json"))
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--phase", choices=("pilot", "final"), required=True)
    p.add_argument("--pilot", type=pathlib.Path)
    p.add_argument("--candidates", default="")
    p.add_argument("--B", type=positive_grid, default=[128,256,512])
    p.add_argument("--T", type=positive_grid, default=[128,256])
    p.add_argument("--K", type=positive_grid, default=[1,4])
    p.add_argument("--R", type=positive_grid, default=[256,512,1024])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--corpus-size", type=int, default=1 << 20)
    p.add_argument("--shuffle-seed", type=int, default=2)
    p.add_argument("--bootstrap-seed", type=int, default=3)
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--epsilon", type=float, default=.03)
    p.add_argument("--batch-seconds", type=float, default=.5)
    p.add_argument("--timeout", type=float, default=15)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--cold-samples", type=int, default=10,
                   help="NVBench diagnostic Cold target, using sample-count criterion")
    p.add_argument("--clock-policy", default="unlocked; externally managed")
    p.set_defaults(func=collect)
    p = sub.add_parser("analyze")
    p.add_argument("table", type=pathlib.Path)
    p.add_argument("--baseline", default="llvm-double")
    p.set_defaults(func=analyze)
    a = parser.parse_args()
    try:
        return a.func(a) or 0
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
