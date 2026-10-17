# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""End-to-end host fixtures: no real CUDA execution, downloads or installed libs."""
import csv
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from artifacts import (SCHEMA, REVIEW_CHECKS, atomic_json, build_session, digest,
                       prepare_variant, product_lock, verify_manifest)
from build_device import audit_ir, effective_implementation
from measure import summarize, validate_confirmation

HERE = pathlib.Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="gpu-libm-protocol-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.device = self.root / "device"
        self.device.mkdir()
        self.cubin = self.device / "nvidia-expf/kernel.cubin"
        self.cubin.parent.mkdir()
        self.cubin.write_bytes(b"fixture cubin; never executed on a GPU")
        self.records = [dict(id="nvidia-expf", provider="libdevice", symbol="__nv_expf")]
        self.registry = self.root / "candidates.json"
        atomic_json(self.registry, self.records)
        sha = digest(self.cubin)
        self.manifest = self.device / "manifest.json"
        atomic_json(self.manifest, dict(schema=SCHEMA, architecture="sm_120", streams=[4],
            registry=self.records, candidates=self.records, products=[dict(id="nvidia-expf",
            cubin_sha256=sha, inspection=dict(cubin_sha256=sha, architecture="sm_120", streams=[4],
            reviewer="fixture", checks=REVIEW_CHECKS))]))
        self.gpu = dict(gpu="fixture GPU", gpu_uuid="GPU-0001", architecture="sm_120",
                        cuda_driver_api_version=13020, nvidia_driver_version="595.99.02",
                        runtime_version=13040,
                        host_compiler="fixture", nvcc_version="13.4")
        atomic_json(self.root / "gpu.json", self.gpu)
        self.mode = self.root / "mode"
        self.mode.write_text("complete")
        self.runner = self.root / "runner"
        self.runner.write_text("#!" + sys.executable + "\n" + r'''
import sys,json,pathlib,time
root=pathlib.Path(__file__).parent
gpu=json.load(open(root/'gpu.json'))
if any(a.startswith('--device-info=') for a in sys.argv):
 print(json.dumps(gpu));sys.exit(0)
args=dict(a[2:].split('=',1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
assert args['expect-driver']==gpu['nvidia_driver_version']
plan=json.load(open(args['schedule']))
mode=(root/'mode').read_text()
for i,p in enumerate(plan):
 if mode=='partial' and i==1: sys.exit(1)
 if mode=='wait' and i==1: time.sleep(60)
 row=dict(gpu,**p,seed=int(args['seed']),corpus_size=int(args['corpus-size']),
   corpus_fnv1a64='stable',status=('incomplete' if mode=='incomplete' else
                                'failed' if mode=='smoke-failed' and i>0 else 'complete'),
   reason='',
   C=p['B']*p['T']*p['K']*p['R'],batch_gpu_seconds=1e-6,
   launches=1 if mode=='incomplete' else 10001,effective_min_samples=1)
 row['gcalls_s']=row['C']/(1e9*row['batch_gpu_seconds'])
 if mode=='smoke-failed' and i>0: row['reason']='numerical smoke failed'
 if mode=='bad-order' and i==1: row['order']=99
 print('GPU_LIBM_ROW '+json.dumps(row),flush=True)
sys.exit(2 if mode=='incomplete' else 1 if mode=='smoke-failed' else 0)
''')
        self.runner.chmod(0o755)

    def command(self, output="table.csv", phase="pilot"):
        return [sys.executable, str(HERE / "measure.py"), "collect",
                "--runner", str(self.runner), "--registry", str(self.registry),
                "--manifest", str(self.manifest), "--output", str(self.root / output),
                "--phase", phase, "--B", "1", "--T", "32", "--K", "4", "--R", "8",
                "--corpus-size", "16", "--batch-seconds", ".01", "--min-samples", "1"]

    def run_collection(self, **kwargs):
        return subprocess.run(self.command(**kwargs), capture_output=True, text=True)

    def rows(self, name="table.csv"):
        with (self.root / name).open(newline="") as f:
            return list(csv.DictReader(f))

    def test_manifest_mismatch_and_missing_cubin_rejected(self):
        self.cubin.write_bytes(b"different candidate")
        r = self.run_collection()
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertFalse((self.root / "table.csv").exists())
        self.cubin.unlink()
        with self.assertRaises(ValueError):
            verify_manifest(self.manifest)

    def test_interrupted_build_invalidates_manifest_and_locks_exclude_mutation(self):
        with product_lock(self.device):
            with self.assertRaises(RuntimeError):
                with build_session(self.device):
                    self.fail("build entered during collection")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with build_session(self.device):
                self.cubin.write_bytes(b"partially rebuilt")
                raise RuntimeError("injected build failure")
        self.assertFalse(self.manifest.exists())
        self.assertEqual(self.run_collection().returncode, 1)

    def test_removed_option_and_compiler_replacement_clear_cache(self):
        variant = self.root / "libc"
        compiler = self.root / "clang"
        compiler.write_bytes(b"compiler version one")
        config = dict(cmake={"LIBC_COMPILE_OPTIONS_DEFAULT": "-ffp-contract=fast"},
                      compiler_path=str(compiler), compiler_hash=digest(compiler))
        prepare_variant(variant, config)
        cache = variant / "CMakeCache.txt"
        cache.write_text("LIBC_COMPILE_OPTIONS_DEFAULT:STRING=-ffp-contract=fast\n")
        prepare_variant(variant, config)
        self.assertTrue(cache.exists())
        config["cmake"] = {}
        prepare_variant(variant, config)
        self.assertFalse(cache.exists())
        cache.write_text("stale objects")
        compiler.write_bytes(b"compiler version two in the same path")
        config["compiler_hash"] = digest(compiler)
        prepare_variant(variant, config)
        self.assertFalse(cache.exists())

    def test_status_codes_and_progressive_rows(self):
        self.assertEqual(self.run_collection().returncode, 0)
        self.assertEqual(len(self.rows()), 10)
        self.mode.write_text("incomplete")
        self.assertEqual(self.run_collection(output="incomplete.csv").returncode, 2)
        self.assertEqual({r["status"] for r in self.rows("incomplete.csv")}, {"incomplete"})
        self.mode.write_text("partial")
        self.assertEqual(self.run_collection(output="partial.csv").returncode, 1)
        self.assertEqual(len(self.rows("partial.csv")), 1)
        self.mode.write_text("bad-order")
        self.assertEqual(self.run_collection(output="bad.csv").returncode, 1)
        self.assertEqual(len(self.rows("bad.csv")), 1)

    def test_failed_validation_preserves_prior_rows(self):
        self.mode.write_text("smoke-failed")
        r = self.run_collection()
        self.assertEqual(r.returncode, 1, r.stderr)
        rows = self.rows()
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["status"], "complete")
        self.assertEqual({r["status"] for r in rows[1:]}, {"failed"})
        self.assertEqual({r["reason"] for r in rows[1:]}, {"numerical smoke failed"})
        result = summarize(rows, resamples=1000, expected_rounds=10)
        self.assertEqual(next(iter(result.values()))["status"], "incomplete external rounds")

    def test_interruption_preserves_completed_rows(self):
        self.mode.write_text("wait")
        with (self.root / "log").open("w") as log:
            p = subprocess.Popen(self.command(), stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    if (self.root / "table.csv").exists() and len(self.rows()) == 1:
                        break
                    time.sleep(.02)
                else:
                    self.fail("first state was not persisted before runner completed")
                p.send_signal(signal.SIGINT)
                self.assertEqual(p.wait(timeout=5), 2)
                self.assertEqual(len(self.rows()), 1)
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()

    def test_missing_configuration_is_reported(self):
        row = dict(id="a", B=1,T=32,K=4,R=8,status="complete",gcalls_s=1)
        rows = [dict(row, round=i) for i in range(10)]
        results = summarize(rows, resamples=1000, expected_rounds=10,
                            expected_points=[("a",1,32,4,8), ("b",1,32,4,8)])
        self.assertEqual(results[("b",1,32,4,8)]["status"], "incomplete external rounds")

    def test_final_rejects_changed_physical_gpu(self):
        self.assertEqual(self.run_collection().returncode, 0)
        self.gpu["gpu_uuid"] = "GPU-another-physical-device"
        atomic_json(self.root / "gpu.json", self.gpu)
        command = self.command("final.csv", "final") + ["--pilot", str(self.root / "table.csv"),
                                                       "--shuffle-seed", "987"]
        r = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("differs from pilot: gpu", r.stderr)
        self.assertFalse((self.root / "final.csv").exists())

    def test_final_checks_installed_driver_release(self):
        self.assertEqual(self.run_collection().returncode, 0)
        command = self.command("same.csv", "final") + ["--pilot", str(self.root / "table.csv"),
                                                      "--shuffle-seed", "987"]
        r = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual({r["nvidia_driver_version"] for r in self.rows("same.csv")}, {"595.99.02"})
        self.assertEqual({r["cuda_driver_api_version"] for r in self.rows("same.csv")}, {"13020"})
        self.gpu["nvidia_driver_version"] = "595.99.03"
        atomic_json(self.root / "gpu.json", self.gpu)
        command = self.command("changed.csv", "final") + ["--pilot", str(self.root / "table.csv"),
                                                         "--shuffle-seed", "987"]
        r = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("differs from pilot: gpu", r.stderr)
        self.assertFalse((self.root / "changed.csv").exists())

    def test_driver_release_is_required(self):
        self.assertEqual(self.run_collection().returncode, 0)
        pilot = self.rows()
        protocol = json.loads(pilot[0]["protocol"])
        for index, release in enumerate((None, "", " ", 13020)):
            with self.subTest(release=release):
                self.gpu["nvidia_driver_version"] = release
                if release is None:
                    del self.gpu["nvidia_driver_version"]
                atomic_json(self.root / "gpu.json", self.gpu)
                output = f"missing-{index}.csv"
                r = self.run_collection(output=output)
                self.assertEqual(r.returncode, 1)
                self.assertIn("driver release missing", r.stderr)
                self.assertFalse((self.root / output).exists())
                # Even identical missing/invalid values cannot confirm a pilot.
                changed = dict(protocol, gpu=self.gpu)
                old = [dict(pilot[0], protocol=json.dumps(changed))]
                with self.assertRaisesRegex(ValueError, "driver release missing"):
                    validate_confirmation(old, changed, [], 1, 16, pilot[0]["build_sha256"])
                with self.assertRaisesRegex(ValueError, "driver release missing"):
                    validate_confirmation(old, protocol, [], 1, 16, pilot[0]["build_sha256"])

    def test_missing_cubin_inspection_rejected(self):
        m = json.loads(self.manifest.read_text())
        del m["products"][0]["inspection"]
        atomic_json(self.manifest, m)
        self.assertEqual(self.run_collection().returncode, 1)

    def test_audit_rejects_kernel_without_evaluations(self):
        ir = '''define ptx_kernel void @expf_k1() {
entry:
  %r = phi i64 [0, %entry], [1, %entry]
  %x = call i32 asm sideeffect "xor.b32 $0, $0, 0x80000000;", "=r,0"(i32 0)
  ret void
}'''
        with mock.patch("build_device.run", return_value=mock.Mock(stdout=ir)):
            with self.assertRaises(ValueError):
                audit_ir("opt", "kernel.bc", [1], "expf")
            with self.assertRaises(ValueError):
                audit_ir("opt", "optimized.bc", [1])

    def test_selector_metadata_allows_numeric_variant_names(self):
        atomic_json(self.root / "compile_commands.json", [dict(
            file="/libc/src/math/generic/expf.cpp", directory=str(self.root),
            command="clang++ -c expf.cpp -o expf.o")])
        result = "float expf(float x) { return float_eval::cr_expf_tb128_horner(x); }"
        with mock.patch("build_device.run", side_effect=[mock.Mock(stdout=result), mock.Mock(stdout="")]):
            self.assertEqual(effective_implementation({}, self.root)["implementation"],
                             "float_eval::cr_expf_tb128_horner")

    def test_external_project_preserves_launcher_lists(self):
        source = self.root / "source"
        (source / "runner").mkdir(parents=True)
        (source / "runner/CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.30.4)
project(Fixture LANGUAGES NONE)
file(WRITE "${CMAKE_BINARY_DIR}/launchers" "${CMAKE_CXX_COMPILER_LAUNCHER}\\n${CMAKE_CUDA_COMPILER_LAUNCHER}\\n")
''')
        (source / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.30.4)
project(Fixture LANGUAGES NONE)
add_subdirectory("{HERE}" bridge)
set(GPU_LIBM_BENCH_SOURCE "{source}" CACHE INTERNAL "" FORCE)
''')
        (source / "build_device.py").write_text('''import json,pathlib,sys
pathlib.Path(__file__).with_name('kernel-arguments.json').write_text(json.dumps(sys.argv[1:]))
''')
        binary = self.root / "build"
        cxx = "/usr/bin/env;CCACHE_DISABLE=1;PIPE=left|right;/usr/bin/ccache"
        cuda = "/usr/bin/env;CUDA_FIXTURE=1;/usr/bin/ccache"
        log = self.root / "cmake.log"
        with log.open("w") as out:
            subprocess.run(["cmake", "-G", "Ninja", "-S", str(source), "-B", str(binary),
                f"-DCMAKE_CXX_COMPILER_LAUNCHER={cxx}", f"-DCMAKE_CUDA_COMPILER_LAUNCHER={cuda}"],
                stdout=out, stderr=out, check=True)
            subprocess.run(["cmake", "--build", str(binary), "--target", "gpu-libm-bench-runner"],
                stdout=out, stderr=out, check=True)
            subprocess.run(["cmake", "--build", str(binary), "--target", "gpu-libm-bench-kernels"],
                stdout=out, stderr=out, check=True)
        self.assertEqual((binary / "bridge/host/launchers").read_text().splitlines(), [cxx,cuda])
        arguments = json.loads((source / "kernel-arguments.json").read_text())
        self.assertEqual(arguments[arguments.index("--cxx-launcher") + 1], cxx)
