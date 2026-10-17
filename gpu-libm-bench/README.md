# NVIDIA scalar expf throughput MVP

This private external LLVM project measures **the sustained throughput of a
specified GPU workload**, in scalar evaluations per second. It does not estimate
the exclusive execution time, serial latency, or intrinsic maximum throughput of
`expf`. No baseline subtraction is performed.

## Build

Requirements: Linux, CMake >= 3.30.4, Ninja, Python >= 3.9, a native C++17 compiler
compatible with the selected CUDA Toolkit, and an LLVM build with Clang and the
NVPTX backend. The toolkit must support the selected architecture (`sm_120`
requires CUDA >= 12.8). A GPU/working driver is required only for execution.
NVBench, fmt, nlohmann/json, RAPIDS CMake and CPM are local, pinned sources.
There are no configure/build downloads and no system installation of NVBench.

Add these options to the usual LLVM configuration (including existing ccache,
assertion, install-prefix and runtime options):

```sh
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS='clang;clang-tools-extra;lld' \
  -DLLVM_ENABLE_RUNTIMES='openmp;offload;libc' \
  -DLLVM_RUNTIME_TARGETS='default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda' \
  -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=libc \
  -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=libc \
  -DLLVM_EXTERNAL_PROJECTS=gpu-libm-bench \
  -DLLVM_EXTERNAL_GPU_LIBM_BENCH_SOURCE_DIR="$PWD/gpu-libm-bench" \
  -DGPU_LIBM_BENCH_ARCH=sm_120 \
  -DGPU_LIBM_BENCH_HOST_CXX=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gpu-libm-bench --parallel 4
```

Products are under `build/tools/gpu-libm-bench/{device,host}`. The native runner
and NVBench use nvcc plus `GPU_LIBM_BENCH_HOST_CXX`. Only the measured kernels use
the selected build's Clang/LLVM tools. Compiler launchers propagate to both
native and libc subbuilds. Tool dependencies are registered after the enclosing
LLVM tools/runtimes targets have been created; isolated libc builds do not
consume the enclosing runtime libraries. The project is excluded from `all`.
Removing it from `LLVM_EXTERNAL_PROJECTS` removes its CUDA/NVBench requirements.

To reuse an already built compiler without reconfiguring the main LLVM build:

```sh
cmake -S gpu-libm-bench -B build/gpu-libm-bench -G Ninja \
  -DGPU_LIBM_BENCH_TOOLS="$PWD/build/bin" \
  -DGPU_LIBM_BENCH_HOST_CXX=/usr/bin/g++ \
  -DGPU_LIBM_BENCH_ARCH=sm_120 \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build/gpu-libm-bench --target gpu-libm-bench --parallel 4
```

Use a separate build directory and `GPU_LIBM_BENCH_ARCH=sm_90` for H100 or
`sm_89` for L4. CUDA's architecture is explicit even without a GPU. Configure
`GPU_LIBM_BENCH_CUDA_ROOT` once (default `/usr/local/cuda`); libdevice is resolved
there as `nvvm/libdevice/libdevice.10.bc`. `GPU_LIBM_BENCH_K=1,2,4,8` determines
the compiled stream counts. There is no install step or automatic execution.

Each libc candidate has an independent `device/<id>/libc` runtimes build, with
`libmbitcode` producing its own `libm.bc`. Explicit `-D` options override GPU
JSON defaults, which remain unchanged. The bitcode is compiled for libc's
generic NVPTX/PTX 6.3 target, as intended by its bitcode packaging; the final
kernel pipeline explicitly targets the requested GPU:

```
Clang CUDA device IR + candidate bitcode
  -> llvm-link --only-needed
  -> opt (internalize, globaldce, default<O3>, NVPTX target callbacks)
  -> llc -mcpu=sm_N -> PTX -> ptxas -arch=sm_N -> cubin
```

Inlining is enabled; there is no device function-pointer dispatch. NVIDIA
symbols are selected explicitly. This Clang's
`clang/lib/Headers/__clang_cuda_device_functions.h` maps `__expf` to
`__nv_fast_expf`. NVVMReflect is resolved during optimization, never deleted as
an undefined dependency. All candidates share the final pipeline and
`-ffp-contract=off`; explicit algorithmic FMAs remain FMAs.

`device/manifest.json` (schema 2) records configurations, actual preprocessed
selector/math defines, LLVM commit, tool versions and executable hashes,
workload/bitcode/cubin hashes, and ptxas resource reports. An exclusive
`device/.lock` protects builds. The old manifest is invalidated before products
are modified, and a new manifest is published atomically only on success.
Collection verifies every selected cubin's hash under a shared lock held for the
whole campaign; runner processes also hold a shared lock. Cooperating build and
review commands refuse to modify products while they are in use. Do not edit
build products manually during a campaign. Old manifest schemas require a
rebuild. libc caches are cleared when the complete configuration or toolchain
identity changes, including removed options and compilers replaced in place.
Launcher lists, including their arguments, propagate without shell evaluation.

Before linking, structural IR checks require K calls in the runtime loop, their
dependence on opaque input updates, checksum-result bits and K observable stores.
After optimization, additional checks catch missing math and unresolved
reflection. These checks and the checksum smoke are **not proof** of the number
of evaluations after inlining and ptxas. A final-code inspection is mandatory
for each candidate/architecture/K before collection:

```sh
python3 gpu-libm-bench/review_code.py \
  --manifest build/gpu-libm-bench/device/manifest.json --candidate llvm-double
# Inspect the printed SASS and ptxas report; then record the completed review:
python3 gpu-libm-bench/review_code.py \
  --manifest build/gpu-libm-bench/device/manifest.json --candidate llvm-double \
  --record-review --reviewer '<name>'
```

Repeat for every candidate. `--streams` can select a compiled K subset and
`--nvdisasm` selects a disassembler from a different toolkit. Check mathematical
operations inside the loop, input/result dependencies, the R-controlled backedge
and K scalar evaluations per iteration, initial loads/final stores, and register
residency of argument/checksum states. Algorithm-owned table/temporary accesses
must be distinguished from workload states. Review records are build metadata
bound to the exact cubin hash, architecture and K. A rebuild preserves a review
only when its cubin is byte-identical. Collection refuses unreviewed products.
SASS dumps are optional temporary inspection aids, not required deliverables.

## Fixed workload and inputs

`runner/Inputs.h` copies the integer SplitMix64 initialization (including its
zero-state fallback) and xorshift64* algorithm from
`libc/benchmarks/gpu/Random.h`. The seed is an unsigned 64-bit integer, default
1. The runner creates one immutable host corpus per invocation, before running
any NVBench state; the study driver uses one invocation for all external rounds.
For increasing indices, `k=next32()>>8`, `u=double(k)*0x1p-24`, and
`x=float(-10.0+20.0*u)`. The host must use RN ties-to-even. Binary64 intermediate
operations here are exact; the final cast rounds to binary32. The power-of-two
corpus size defaults to 2^20. The CSV includes a reproducible FNV-1a fingerprint
over little-endian binary32 bytes. `runner --inputs --seed=1 --corpus-size=16`
prints a fingerprint and prefix without initializing CUDA.

For `q=blockIdx.x*T+threadIdx.x`, stream `s in [0,K)` loads
`base=x[(q*K+s) mod corpus_size]` once. For iteration `i in [0,R)`, its argument
has bits `bits(base) XOR (((i+1) mod 2)*0x80000000)`. Thus each stream alternates
`-base,+base,...` independently of the candidate and previous function results.
This is deliberate corpus reuse, not R new random inputs per stream; both
signs remain finite and in [-10,10]. Special input distributions and nonfinite
results are outside this MVP.

A volatile PTX `xor.b32` with a read/write register operand implements the sign
update. Its value is opaque to LLVM, preventing pure-call loop hoisting while
performing the stated transformation on the device. Each stream maintains a
separate uint32 checksum, initially zero, updated after every result as
`h=rotl32(h,1) + bits(y)` with modulo-2^32 unsigned addition. The K
checksums are stored at `output[q*K+s]` after the loop. The loop bound R is a
runtime kernel parameter; K is compile-time and its loops are fully unrolled.
Each launch resets states. No PRNG, global corpus loads, or checksum stores
occur inside the evaluation loop. Implementation table accesses remain there.
K argument/checksum streams and threads are independent; each individual stream
has argument-update and integer-checksum dependencies. There is no floating
feedback dependency on the preceding `expf` result. This avoids the `0*Inf/NaN`
issue of IEEE zero multiplication barriers.

All B*T threads execute K*R evaluations without a tail predicate. The public
`expf_probe` is a separate, unmeasured numerical smoke kernel. Before timing a
module, up to 4096 signed corpus values plus endpoints/zeros are compared with
the native host's independent double-precision `std::exp`, using relative
tolerance 4e-6 for every candidate. This detects gross ABI/configuration errors;
it does not certify correct rounding or bound errors outside this corpus.
For every candidate/K, an additional unmeasured checksum smoke compares 32
threads at R=1,2,7 with separately evaluated signed inputs through the probe.
This checks iteration count, sign order, corpus reuse and observable outputs.
Validation successes and failures are cached for the lifetime of the runner.
A numerical smoke failure rejects that candidate on the selected device; a
workload smoke failure rejects its candidate/K on that device. Later states
reuse the original failure and cannot enter timing. Only successfully checked
modules enter the module cache. Affected states retain the cause and status
`failed`, with exit 1; previously exported rows remain available.

## Measurement protocol

For theta=(B,T,K,R), `C=B*T*K*R` is checked with integer overflow detection and
fits NVBench's signed 64-bit element count. It counts scalar evaluations by
threads; a warp instruction is not one scalar evaluation.

NVBench owns benchmark registration, Candidate/B/T/K/R axes, selected devices,
warmup, adaptive launch repetition, streams, CUDA events and summaries. Round
and Order are additional metadata axes. Each scheduled point is an NVBench
benchmark with singleton workload axes; scheduling only orders points and does
not perform timing. Modules, parameters, memory and the smoke test are prepared
outside `state.exec`. The launcher passed to
`state.exec(nvbench::exec_tag::gpu, launcher)` does exactly one `cuLaunchKernel`
on `launch.get_stream()` in the selected device's primary CUDA context. It does
not synchronize or perform host/device copies or allocation.

The sole throughput denominator is the pinned NVBench revision's
`nv/batch/time/gpu/mean` (Batch GPU): sum of measured batch GPU event durations
divided by sum of measured kernel launches. The CSV computes
`Q_kernel=C/(1e9*t)` Gcalls/s, with t in seconds. Cold GPU/CPU timings and
NVBench's automatic Elem/s column are diagnostic and never used for this
calculation. Batch warmup is excluded from its summaries. In this revision,
the effective Batch minimum launch count is also raised to the Cold sample
count when available; therefore short requested batches can collect longer.
Missing/invalid Batch summaries and unmet collection budgets are incomplete.
`runner/Measurement.h` classifies the budget once, independently of CSV output:
exit 0 means complete, 2 incomplete, and 1 an execution/validation failure.
The study driver propagates those outcomes after preserving the received rows.

The interval includes all kernel work: initial input loads, math evaluation,
sign updates, checksum arithmetic, loop control and final stores. Stream gaps
between launches may also enter the interval. CPU input generation, transfers,
module loading, allocations and the numerical smoke are excluded. Outputs are
not inputs to subsequent launches. The primary cache regime is warmed reuse of
the same modules and immutable corpus, without explicit cache flushing. It
does not assume that all tables fit in cache.

The study driver runs at least ten external rounds; each visits every candidate
and configuration once in a seeded permutation. Keep the GPU otherwise idle.
Clock locking is an external experimental choice: pass its actual policy as
`--clock-policy`; the runner never silently changes clocks. CSV telemetry samples
SM/memory clocks, temperature, power and throttle flags before and after each
state via NVML (unavailable fields are absent). These endpoint samples do not
describe instantaneous variation within a batch. No nominal-frequency
normalization is performed. The study driver explicitly uses NVBench's
`sample-count` criterion for Cold diagnostics, with `--cold-samples=10` by
default; the criterion does not stop Batch timing. The Cold target, Batch
duration, minimum samples and timeout are fixed and recorded in the protocol.
`cuda_driver_api_version` records the CUDA version supported by the driver;
`nvidia_driver_version` records the installed NVIDIA driver release queried
through NVML. The runtime version and a stable GPU UUID are also recorded. A
runtime-only `runner --device-info=0` query identifies the platform before a
study; final confirmation checks the UUID, architecture, installed driver,
toolchain and budgets
against the pilot, even when the visible ordinal has changed.
The release must be a nonempty string in preflight and in the pilot protocol.
An unavailable release is a validation error, not a match between missing
values. The runner checks `--expect-driver` before measured work, and every
exported state includes the release. NVML driver identification is required for
execution; configuration and compilation still do not query the GPU.

Use `measure.py collect` for campaigns: it performs product/hash/review
preflight, schedules all rounds in one runner process, and receives completed
states through an explicit JSON-lines export outside the timed callback. It
validates order, corpus, device identity and C/t/Q before appending each row to
the final CSV, flushing and fsyncing each state. No temporary measurement table
is discarded on failure. Interrupted or failed campaigns retain the valid
prefix; analysis uses the declared grid and rounds to mark absent observations
and completely missing configurations incomplete. SIGINT returns 2 after
terminating the runner; an execution/validation error returns 1. Output files
must not already exist. The direct runner interface remains useful for smoke
and diagnostic runs; it flushes `--csv` after each state, but the study driver's
preflight and protocol metadata are required for a recorded campaign.

Example using the independent build directory above:

```sh
python3 gpu-libm-bench/test.py
python3 gpu-libm-bench/measure.py collect \
  --runner build/gpu-libm-bench/host/gpu-libm-bench \
  --manifest build/gpu-libm-bench/device/manifest.json \
  --phase pilot --output pilot.csv \
  --B 128,256,512 --T 128,256 --K 1,4 --R 256,512,1024
python3 gpu-libm-bench/measure.py analyze pilot.csv
```

Use the pilot to choose the fixed workload, collection duration, candidate and
configuration grid, and epsilon **before** the final comparison. If these need
adjustment, rerun a pilot with the new design. To confirm independently, repeat
the command with `--phase final --pilot pilot.csv --output final.csv
--shuffle-seed 12345`. The final design may be a subset of pilot points, but
must use the same build, corpus and sampling budgets/epsilon. Keep both tables;
never pool pilot and final rounds for inference. The independent permutation
seed distinguishes confirmation from the pilot; measurements are newly
collected. For candidate-specific tuning, retain its pilot selection and label
the resulting configurations explicitly; default comparisons use common theta.

Analysis reports median Q, IQR and a 95% percentile-bootstrap CI of the median.
It samples whole external rounds with replacement using the same indices for
every point, preserving pairing. Ratios use ratios of medians for each such
resample. Defaults are seed=3 and 10000 resamples. The protocol and Python
version are recorded. Internal NVBench samples and lanes are not independent
rounds. Ten rounds do not guarantee exact CI coverage. A fixed corpus's temporal
CI does not estimate uncertainty across input corpora; independent corpus seeds
are needed for distributional generalization.

For each fixed T,K and endpoint B,R, saturation requires four equivalence
checks: R/4 -> R/2 -> R at B, and B/4 -> B/2 -> B at R. Each ratio's paired 95%
CI must be contained in [1-epsilon,1+epsilon], epsilon=0.03 by default.
Missing neighbors, incomplete rounds, or insufficient precision produce
**saturation not demonstrated**. Preserve all measured points, including failed
equivalence checks. These checks establish operational amortization and a
concurrency plateau only. Attribute a bottleneck to `expf` compute only after
separate profiling and code inspection exclude memory, spills and helper
instructions as the limiting resource. Profiling is not part of timing.

## Candidates and updates

`candidates.json` is the only enumeration. LLVM-libc records contain id, provider
and the complete CMake math/options map; every LLVM candidate exports `expf`.
NVIDIA records contain id, provider and the explicit libdevice symbol. Current
LLVM choices select float, double with/without accurate pass and cr using the
existing selector in `libc/src/__support/math/expf.h`. Neither SMALL_TABLES nor
INTERMEDIATE_COMP_IN_FLOAT is enabled for the double candidates.

To add an already CMake-selectable variant, append a record with a unique ID
and its complete configuration, then rebuild. No runner/kernel dispatch cases
are needed. Future `cr_expf_tbN_poly` records use this same mechanism after the
algorithm and selector exist; this MVP creates neither algorithms nor speculative
selectors. Do not edit `libc/config/gpu/config.json` to select candidates.
In the current double implementation, the accurate-path correction handles
input bits `0xc236bd8c`, outside [-10,10]. This workload compares the presence of
that guard/code with its omission; it does not exercise the correction itself.

```sh
third-party/update_nvbench.sh --yes
```

The updater pins revisions and SHA256 archives, stages/prunes sources, preserves
licenses, reapplies `third-party/nvbench-local.patch`, replaces only these vendor
directories, and cleans temporary downloads. The local patch suppresses
NVBench's developer-only source-tree compile_commands symlink when embedded.
See `third-party/nvbench-deps/README.llvm` for versions and
`third-party/nvbench/docs/benchmarks.md` for the runner API. Updating the pin
requires rechecking Batch summary semantics/identifiers and the numerical and
code-generation checks. Normal artifacts are build products, raw CSV tables and
this README; analysis prints to stdout. Extra dumps are optional inspection aids.

## Validation in this workspace

The initial build used Clang/LLVM 24.0.0git, CUDA Toolkit 13.4.92, GCC 15.2,
CMake 4.2.3 and NVBench revision
`a7fe2f3d47980a48a21dac5d407dd8eff007dfc4`. Clang warns that CUDA 13.4 is only
partially supported; this combination was exercised here, not certified for
other CUDA features. The RTX 5070 reports sm_120 and driver 595.99.02.
All six candidates were rebuilt after the first review fixes and all 24 candidate/K
combinations passed the numerical/checksum smokes, with a 16-value corpus to
exercise wraparound. Prelink IR dataflow checks passed; final SASS inspection
found the math inside R-controlled loops, initial corpus loads before them,
observable result stores after them and no local state accesses. ptxas reports
zero stack/spills for K=1,2,4,8. The inspections are bound to these cubin hashes.
A real concurrent build attempt was rejected while collection held its lock,
preserving the manifest. Deliberately impossible budgets returned 2 both with
and without CSV. Fourteen host tests cover cache removal/compiler replacement,
launchers with arguments (including literal pipes), product integrity and build
interruption, SIGINT recovery, incomplete exits, wholly absent configurations,
GPU changes, inspection gating, variant names, inputs and paired inference.

The new pilot collected 2160 complete states: all six candidates, B=96,192,384,
T=128,256, K=1,4, R=256,512,1024, ten external rounds, corpus seed=1/size=2^20,
shuffle seed=2, Batch target=0.02 s, Cold target=10, minimum samples=10,
timeout=5 s and unlocked clocks. Epsilon=0.03; bootstrap seed=3/10000 resamples.
The common T=256,K=4 configuration was selected from this pilot for independent
confirmation, retaining all B/R neighbors and using shuffle seed=12345. These
short collections validate the protocol; missing equivalence evidence remains
`saturation not demonstrated`, and no expf-only bottleneck is asserted.
Confirmation completed 540 states in ten new rounds with the then-current runner.
At B=384,T=256,K=4,R=1024, the four equivalence checks passed for llvm-cr,
llvm-double and llvm-double-skip; saturation was not demonstrated for llvm-float
or either libdevice candidate. Every other configuration retains its own status.
Raw tables are saved under `build/gpu-libm-bench-validation-20261016/` as
`pilot.csv` and `final.csv`; they embed the exact manifest, runner/analysis
hashes and protocol. Those copies remain available outside the temporary build.
These campaigns predate the second review's persistent smoke-failure cache and
explicit NVIDIA driver release field. Their original `driver_version` is the
CUDA API compatibility version, not the release. The tables retain their
original runner/analysis hashes and are not rewritten or relabeled after these
corrections. They remain analyzable, but cannot serve as a pilot for the updated
runner. For the second-review fixes, six focused host regressions passed:
injected numerical/workload failures remain rejected across later states,
earlier rows survive validation errors, driver-release changes with the same
CUDA API version reject confirmation, missing releases reject preflight and
confirmation, and existing partial-exit/GPU-identity checks still pass. The
native runner rebuilt successfully, and `--device-info=0` returned
`cuda_driver_api_version=13020` and `nvidia_driver_version=595.99.02` on the RTX
5070. No performance campaign or unchanged device kernel was rerun.
Libdevice kernels also cross-compiled for sm_89/sm_90; H100 and L4 were not
available for execution.
An additional copy of the LLVM cr configuration was built and executed at K=1,4
with a new registry ID without changing runner or kernel code. The vendor updater was exercised
against the pinned archives. Disabling the external project was checked with
an invalid CUDA root and removed its targets/requirements.
The encompassing LLVM configuration with all requested projects/runtimes was
configured separately; the standalone bridge was built against the existing
LLVM tools. The earlier 60-row one-point CSV is historical evidence with its
original manifest/runner identity; it was not relabeled as a new build's data.
Retain raw pilot and final tables separately. These tests do not certify rounding,
generalize across corpora or establish saturation for every implementation.
