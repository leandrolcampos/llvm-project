#!/usr/bin/env python3
"""Build isolated libc configurations and identical final device pipelines."""
import argparse
import json
import pathlib
import re
import subprocess
import shlex

from artifacts import SCHEMA, atomic_json, build_session, digest, prepare_variant


def registry(path):
    records = json.loads(pathlib.Path(path).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("registry must be a nonempty JSON array")
    seen = set()
    for c in records:
        ident = c.get("id", "")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", ident) or ident in seen:
            raise ValueError(f"invalid or duplicate candidate ID: {ident!r}")
        seen.add(ident)
        if c.get("provider") == "llvm-libc":
            if set(c) != {"id", "provider", "cmake"} or not isinstance(c["cmake"], dict):
                raise ValueError(f"{ident}: expected id, provider, cmake")
            if "LIBC_CONF_MATH_OPTIMIZATIONS" not in c["cmake"]:
                raise ValueError(f"{ident}: explicit math configuration required")
            for key, value in c["cmake"].items():
                if not key.startswith("LIBC_") or not isinstance(value, str):
                    raise ValueError(f"{ident}: invalid libc CMake option {key}")
        elif c.get("provider") == "libdevice":
            if set(c) != {"id", "provider", "symbol"} or not re.fullmatch(
                    r"[a-zA-Z_][a-zA-Z0-9_]*", c.get("symbol", "")):
                raise ValueError(f"{ident}: invalid libdevice symbol")
        else:
            raise ValueError(f"{ident}: unknown provider")
    return records


def run(args, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def effective_implementation(tools, variant):
    commands = json.loads((variant / "compile_commands.json").read_text())
    command = next(x for x in commands if x["file"].endswith("/math/generic/expf.cpp"))
    flags = shlex.split(command["command"])
    # Preprocess the exact public expf translation unit with its actual flags.
    # Drop object/dependency output options; stdout is inspected, not saved.
    clean = []
    i = 0
    while i < len(flags):
        if flags[i] in ("-o", "-MF", "-MT", "-MQ"):
            i += 2
        elif flags[i] in ("-c", "-MD", "-MMD"):
            i += 1
        else:
            clean.append(flags[i])
            i += 1
    text = run(clean + ["-E"], cwd=command["directory"], capture_output=True, text=True).stdout
    match = re.search(r"float expf\(float x\)\s*\{\s*return ([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\(x\);", text)
    if not match:
        raise ValueError("could not verify preprocessed expf selector")
    macros = run(clean + ["-E", "-dM"], cwd=command["directory"],
                 capture_output=True, text=True).stdout
    settings = [line for line in macros.splitlines() if line.startswith("#define LIBC_MATH")]
    return {"implementation": match[1], "math_defines": settings}


def audit_ir(tool, bc, ks, symbol=None):
    ir = run([tool, "-S", bc, "-o", "-"], capture_output=True, text=True).stdout
    if "@__nvvm_reflect(" in ir:
        raise ValueError("unresolved NVVMReflect after optimization")
    for k in ks:
        match = re.search(rf"^define [^\n]*@expf_k{k}\(.*?^\}}", ir, re.M | re.S)
        if not match:
            raise ValueError(f"missing optimized expf_k{k}")
        body = match[0]
        # Structural regression checks, not proof about the final machine code.
        if body.count('asm sideeffect "xor.b32') != k or "phi " not in body:
            raise ValueError(f"expf_k{k}: stream updates/runtime loop not preserved")
        if symbol is not None and re.search(r"\balloca\b", body):
            raise ValueError(f"expf_k{k}: unexpected stack allocation; inspect IR")
        if symbol is not None:
            calls = re.findall(rf"(%[\w.]+) = [^\n]*call float @{re.escape(symbol)}\(float[^\n]* (%[\w.]+)\)", body)
            if len(calls) != k or body.count("store i32 ") != k:
                raise ValueError(f"expf_k{k}: missing scalar calls or observable stores")
            blocks = re.split(r"^([\w.]+):[^\n]*\n", body, flags=re.M)
            loop = next((blocks[i + 1] for i in range(1, len(blocks), 2)
                         if "!llvm.loop" in blocks[i + 1]), "")
            if (len(re.findall(rf"call float @{re.escape(symbol)}\(", loop)) != k or
                    "%repeats" not in loop or "phi i64" not in loop):
                raise ValueError(f"expf_k{k}: scalar calls must remain in the runtime loop")
            for result, argument in calls:
                cast = re.search(rf"{re.escape(argument)} = bitcast i32 (%[\w.]+) to float", loop)
                if not cast or not re.search(rf'{re.escape(cast[1])} = [^\n]*asm sideeffect "xor.b32', loop):
                    raise ValueError(f"expf_k{k}: call does not depend on opaque argument update")
                if not re.search(rf"bitcast float {re.escape(result)} to i32", loop):
                    raise ValueError(f"expf_k{k}: result does not contribute checksum bits")
        elif not re.search(r"\b(fmul|fadd|fsub|fdiv)\b|call[^\n]*(?:@llvm\.(?:fma|nvvm)|float @)", body):
            raise ValueError(f"expf_k{k}: no mathematical evaluation survives")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=pathlib.Path, required=True)
    p.add_argument("--tools", type=pathlib.Path, required=True)
    p.add_argument("--cuda", type=pathlib.Path, required=True)
    p.add_argument("--arch", default="sm_120")
    p.add_argument("--registry", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--streams", default="1,2,4,8")
    p.add_argument("--c-launcher", default="")
    p.add_argument("--cxx-launcher", default="")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--candidates", default="", help="comma-separated subset")
    a = p.parse_args()
    if not re.fullmatch(r"sm_[0-9]+[af]?", a.arch):
        p.error("use an explicit NVIDIA architecture, e.g. sm_120")
    ks = [int(x) for x in a.streams.split(",")]
    if not ks or len(set(ks)) != len(ks) or any(k < 1 or k > 32 for k in ks):
        p.error("streams must contain distinct counts in [1,32]")
    records = registry(a.registry)
    selected = set(a.candidates.split(",")) if a.candidates else {c["id"] for c in records}
    if selected - {c["id"] for c in records}:
        p.error("unknown candidate in --candidates")
    a.output = a.output.resolve()
    with build_session(a.output) as previous:
        build(a, records, selected, ks, previous)


def build(a, records, selected, ks, previous=None):
    tools = {t: (a.tools / t).absolute() for t in
             ("clang", "clang++", "llvm-link", "opt", "llc", "llvm-nm", "llvm-ar", "llvm-ranlib")}
    for path in tools.values():
        if not path.is_file():
            raise ValueError(f"missing tool: {path}")
    hashes = {}
    toolchain = {}
    for name, path in tools.items():
        resolved = path.resolve()
        if resolved not in hashes:
            hashes[resolved] = digest(resolved)
        toolchain[name] = dict(path=str(path), sha256=hashes[resolved])
    libdevice = a.cuda / "nvvm/libdevice/libdevice.10.bc"
    source = a.repo / "gpu-libm-bench/kernels/expf.cu"
    symbols = ",".join([f"expf_k{k}" for k in ks] + ["expf_probe"])
    results = []
    for c in records:
        if c["id"] not in selected:
            continue
        out = a.output / c["id"]
        out.mkdir(exist_ok=True)
        if c["provider"] == "llvm-libc":
            variant = out / "libc"
            prepare_variant(variant, dict(cmake=c["cmake"], toolchain=toolchain,
                                         builder=digest(__file__), cuda=str(a.cuda.resolve()),
                                         c_launcher=a.c_launcher, cxx_launcher=a.cxx_launcher))
            command = ["cmake", "-G", "Ninja", "-S", a.repo / "runtimes", "-B", variant,
                       "-DLLVM_ENABLE_RUNTIMES=libc", "-DLLVM_LIBC_FULL_BUILD=ON",
                       "-DLLVM_INCLUDE_TESTS=OFF", "-DLLVM_INCLUDE_BENCHMARKS=OFF",
                       "-DLIBC_INCLUDE_BENCHMARKS=OFF", "-DCMAKE_BUILD_TYPE=Release",
                       "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                       "-DLLVM_DEFAULT_TARGET_TRIPLE=nvptx64-nvidia-cuda",
                       "-DLLVM_RUNTIMES_TARGET=nvptx64-nvidia-cuda",
                       "-DCMAKE_SYSTEM_NAME=Linux",
                       "-DCMAKE_C_COMPILER_TARGET=nvptx64-nvidia-cuda",
                       "-DCMAKE_CXX_COMPILER_TARGET=nvptx64-nvidia-cuda",
                       "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
                       "-DCMAKE_C_FLAGS=-flto -nogpulib",
                       "-DCMAKE_CXX_FLAGS=-flto -nogpulib",
                       f"-DCMAKE_C_COMPILER={tools['clang']}",
                       f"-DCMAKE_CXX_COMPILER={tools['clang++']}",
                       f"-DCMAKE_AR={tools['llvm-ar']}", f"-DCMAKE_RANLIB={tools['llvm-ranlib']}",
                       f"-DLIBC_LLVM_LINK={tools['llvm-link']}",
                       f"-DCUDAToolkit_ROOT={a.cuda}",
                       f"-DCMAKE_C_COMPILER_LAUNCHER={a.c_launcher}",
                       f"-DCMAKE_CXX_COMPILER_LAUNCHER={a.cxx_launcher}"]
            command += [f"-D{k}:STRING={v}" for k, v in c["cmake"].items()]
            run(command)
            cache = (variant / "CMakeCache.txt").read_text()
            for key, value in c["cmake"].items():
                if not re.search(rf"^{re.escape(key)}:[^=]+={re.escape(value)}$", cache, re.M):
                    raise ValueError(f"{c['id']}: effective CMake cache mismatch: {key}")
            run(["cmake", "--build", variant, "--target", "libmbitcode", "--parallel", a.jobs])
            found = list(variant.rglob("libm.bc"))
            if len(found) != 1:
                raise ValueError(f"expected one isolated libm.bc, found {found}")
            bc, symbol = found[0], "expf"
            effective = effective_implementation(tools, variant)
        else:
            bc, symbol = libdevice, c["symbol"]
            effective = {"implementation": symbol}
        nm = run([tools["llvm-nm"], "--defined-only", "--format=posix", bc],
                 capture_output=True, text=True).stdout
        if not re.search(rf"^{re.escape(symbol)}\s", nm, re.M):
            raise ValueError(f"{c['id']}: undefined {symbol} in {bc}")
        run([tools["clang"], "-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib",
             f"--cuda-gpu-arch={a.arch}", "-O3", "-ffp-contract=off", "-fno-builtin",
             "-emit-llvm", "-c", source, f"-DBENCH_SYMBOL={symbol}",
             "-DBENCH_KERNELS=" + " ".join(f"KERNEL({k})" for k in ks),
             "-o", out / "kernel.bc"])
        audit_ir(tools["opt"], out / "kernel.bc", ks, symbol)
        run([tools["llvm-link"], out / "kernel.bc", "--only-needed", bc,
             "-o", out / "linked.bc"])
        run([tools["opt"], f"-mcpu={a.arch}", "-mtriple=nvptx64-nvidia-cuda",
             f"-internalize-public-api-list={symbols}",
             "-passes=internalize,globaldce,default<O3>", out / "linked.bc",
             "-o", out / "optimized.bc"])
        audit_ir(tools["opt"], out / "optimized.bc", ks)
        run([tools["llc"], f"-mcpu={a.arch}", "-O3", out / "optimized.bc",
             "-o", out / "kernel.ptx"])
        assembled = run([a.cuda / "bin/ptxas", f"-arch={a.arch}", "-O3", "-v", out / "kernel.ptx",
                         "-o", out / "kernel.cubin"], capture_output=True, text=True)
        print(assembled.stderr, end="")
        results.append(dict(id=c["id"], symbol=symbol, bitcode_sha256=digest(bc),
                            cubin_sha256=digest(out / "kernel.cubin"),
                            effective=effective, ptxas=re.sub(
                                r"^ptxas info\s*: Compile time = .*\n", "", assembled.stderr, flags=re.M)))
        if previous and previous.get("architecture") == a.arch:
            old = next((p for p in previous["products"] if p["id"] == c["id"]), {})
            if old.get("cubin_sha256") == results[-1]["cubin_sha256"] and "inspection" in old:
                results[-1]["inspection"] = old["inspection"]
    versions = {name: run([path, "--version"], capture_output=True, text=True).stdout.strip()
                for name, path in (("clang", tools["clang"]), ("ptxas", a.cuda / "bin/ptxas"),
                                   ("cmake", "cmake"))}
    commit = run(["git", "-C", a.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = dict(schema=SCHEMA, architecture=a.arch, streams=ks, registry=records,
                    candidates=[c for c in records if c["id"] in selected],
                    products=results, versions=versions, llvm_commit=commit,
                    workload_sha256=digest(source), nvbench_revision=
                    "a7fe2f3d47980a48a21dac5d407dd8eff007dfc4",
                    builder_sha256=digest(__file__), toolchain=toolchain,
                    nvbench_patch_sha256=digest(a.repo / "third-party/nvbench-local.patch"))
    atomic_json(a.output / "manifest.json", manifest)
    print("Device build complete; no GPU execution performed.")


if __name__ == "__main__":
    main()
