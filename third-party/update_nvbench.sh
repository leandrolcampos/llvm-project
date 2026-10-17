#!/usr/bin/env bash
# Update only the vendored NVBench sources and their offline build dependencies.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NV_BENCH_REVISION=a7fe2f3d47980a48a21dac5d407dd8eff007dfc4
RAPIDS_VERSION=25.12.00
FMT_VERSION=12.2.0
JSON_VERSION=3.12.0
CPM_VERSION=0.42.0
if [[ ${1:-} != --yes ]]; then
  read -r -p "Replace vendored NVBench and nvbench-deps? [y/N] " answer
  [[ $answer == y || $answer == Y ]] || exit 0
fi
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
fetch() {
  curl --fail --location --retry 3 "$1" --output "$stage/archive"
  echo "$2  $stage/archive" | sha256sum --check --status
  mkdir -p "$stage/$3"
  tar -xzf "$stage/archive" -C "$stage/$3" --strip-components=1
}
fetch "https://codeload.github.com/NVIDIA/nvbench/tar.gz/$NV_BENCH_REVISION" \
  9ea3fcffd91786cff7e0c5f1f54b929ba411a53bced5064597977cc8c54002aa nvbench
fetch "https://codeload.github.com/rapidsai/rapids-cmake/tar.gz/refs/tags/v$RAPIDS_VERSION" \
  5c2cd7854f1f4609be29e60785fe60f1e6e776b18870b99b235a11b8d07c2b96 nvbench-deps/rapids-cmake
fetch "https://codeload.github.com/fmtlib/fmt/tar.gz/refs/tags/$FMT_VERSION" \
  8b852bb5aa6e7d8564f9e81394055395dd1d1936d38dfd3a17792a02bebd7af0 nvbench-deps/fmt
fetch "https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v$JSON_VERSION" \
  4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187 nvbench-deps/json
curl --fail --location --retry 3 \
  "https://github.com/cpm-cmake/CPM.cmake/releases/download/v$CPM_VERSION/CPM.cmake" \
  --output "$stage/nvbench-deps/CPM.cmake"
echo "2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a  $stage/nvbench-deps/CPM.cmake" | sha256sum --check --status
python3 - "$stage" <<'PY'
import pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
keep = {
 'nvbench': {'CMakeLists.txt','LICENSE','README.md','cmake','nvbench','docs'},
 'nvbench-deps/rapids-cmake': {'CMakeLists.txt','LICENSE','README.md','VERSION','RAPIDS_BRANCH','rapids-cmake','init.cmake'},
 'nvbench-deps/fmt': {'CMakeLists.txt','LICENSE','README.md','ChangeLog.md','include','src','support'},
 'nvbench-deps/json': {'CMakeLists.txt','LICENSE.MIT','README.md','include','single_include','cmake'},
}
for name, allowed in keep.items():
    for child in (root / name).iterdir():
        if child.name not in allowed:
            if child.is_dir(): shutil.rmtree(child)
            else: child.unlink()
PY
patch --no-backup-if-mismatch --directory "$stage/nvbench" --strip=1 < "$SCRIPT_DIR/nvbench-local.patch"
cp -- "$SCRIPT_DIR/nvbench-deps/README.llvm" "$stage/nvbench-deps/README.llvm"
rm -rf -- "$SCRIPT_DIR/nvbench" "$SCRIPT_DIR/nvbench-deps"
mv -- "$stage/nvbench" "$stage/nvbench-deps" "$SCRIPT_DIR/"
echo "NVBench $NV_BENCH_REVISION and offline dependencies updated."
