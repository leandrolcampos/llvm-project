#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Host-only checks for inputs, registry extensibility, counts and inference."""
import json
import os
import pathlib
import struct
import subprocess
import tempfile
import unittest

from build_device import registry
from measure import ratio, saturation, summarize
from test_protocol import ProtocolTests

HERE = pathlib.Path(__file__).resolve().parent


class HostTests(unittest.TestCase):
    def test_registry_extension(self):
        records = registry(HERE / "candidates.json")
        self.assertEqual(len(records), 6)
        records.append(dict(records[0], id="future-tb128-horner"))
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "candidates.json"
            p.write_text(json.dumps(records))
            self.assertEqual(registry(p)[-1]["id"], "future-tb128-horner")
            records.append(records[0])
            p.write_text(json.dumps(records))
            with self.assertRaises(ValueError):
                registry(p)

    def test_input_bits_and_work_count(self):
        # Golden seed=1 prefix, independently generated from the integer PRNG
        # and IEEE binary32 conversion; also verify three boundary seeds below.
        golden = [0xc083cf64, 0x40dbb4da, 0xc0249764, 0xc0ac10de,
                  0x40e5c9a0, 0xc11f582a, 0xc0ef63c5, 0x3fe04b88]
        code = r'''
#include "Inputs.h"
#include "Measurement.h"
#include <iostream>
int main() {
  for (auto seed : {UINT64_C(1), UINT64_C(0), UINT64_MAX, UINT64_C(1)<<63}) {
    auto v = bench::corpus(seed, 8);
    for (float x : v) std::cout << bench::bits(x) << ' ';
    std::cout << '\n';
  }
  if (bench::work(128, 128, 4, 1024) != 67108864) return 1;
  try { bench::work(UINT64_MAX, 2, 1, 1); return 2; }
  catch (const std::invalid_argument &) {}
  try { bench::work(1, 1, 0, 1); return 3; }
  catch (const std::invalid_argument &) {}
  try { bench::corpus(1, 3); return 4; }
  catch (const std::invalid_argument &) {}
  std::fesetround(FE_DOWNWARD);
  try { bench::corpus(1, 8); return 5; }
  catch (const std::runtime_error &) {}
  bench::Measurement m;
  m.target = .1; m.timeout = 1; m.requested_minimum = 10;
  m.effective_minimum = 20; m.has_batch = true;
  m.seconds = .01; m.launches = 20;
  if (m.code() != 0) return 6;
  m.launches = 19;
  if (m.code() != 2) return 7;
  m.launches = 20; m.seconds = .001;
  if (m.code() != 2) return 8;
  m.failed = true;
  if (m.code() != 1) return 9;
  m.failed = false; m.seconds = .01; m.skipped = true;
  if (m.code() != 2) return 10;
  m.skipped = false; m.seconds = std::numeric_limits<double>::infinity();
  if (m.code() != 2) return 11;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp)
            (p / "test.cpp").write_text(code)
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-ffp-contract=off", "-I", str(HERE / "runner"),
                            str(p / "test.cpp"), "-o", str(p / "test")], check=True)
            lines = subprocess.check_output([str(p / "test")], text=True).splitlines()
        values = [list(map(int, line.split())) for line in lines]
        self.assertEqual(values[0], golden)
        mask = 2**64 - 1
        for actual, seed in zip(values, (1, 0, mask, 2**63)):
            x = (seed + 0x9E3779B97F4A7C15) & mask
            x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
            x = (x ^ (x >> 31)) or 0x9E3779B97F4A7C15
            expected = []
            for _ in actual:
                x ^= x >> 12
                x ^= (x << 25) & mask
                x ^= x >> 27
                k = (((x * 0x2545F4914F6CDD1D) & mask) >> 32) >> 8
                expected.append(struct.unpack("<I", struct.pack("<f", -10 + 20 * k * 2**-24))[0])
            self.assertEqual(actual, expected)

    def test_paired_bootstrap_and_missing_rounds(self):
        rows = []
        for r in range(10):
            # Strong common round drift; a paired ratio must cancel it exactly.
            for ident, scale in (("a", 1), ("b", 2)):
                for b, repeats in ((4,1), (4,2), (4,4), (1,4), (2,4)):
                    rows.append(dict(id=ident, B=b, T=128, K=4, R=repeats,
                                     round=r, status="complete", gcalls_s=scale * (10 + r)))
        s = summarize(rows, resamples=1000)
        self.assertEqual(ratio(s[("b",4,128,4,4)], s[("a",4,128,4,4)]), (2,2,2))
        self.assertTrue(saturation(s, ("a",4,128,4,4), .03))
        self.assertFalse(saturation(s, ("a",2,128,4,4), .03))
        changed = [dict(row, gcalls_s=float(row["gcalls_s"]) *
                        (1.10 if int(row["R"]) == 4 else 1)) for row in rows]
        accelerated = summarize(changed, resamples=1000)
        self.assertFalse(saturation(accelerated, ("a",4,128,4,4), .03))
        rows[-1]["status"] = "incomplete"
        s = summarize(rows, resamples=1000)
        self.assertEqual(s[("b",2,128,4,4)]["status"], "incomplete external rounds")
        self.assertFalse(saturation(s, ("b",4,128,4,4), .03))
        s = summarize(rows, resamples=1000, expected_rounds=11)
        self.assertEqual(s[("a",4,128,4,4)]["status"], "incomplete external rounds")
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]], resamples=1000)

    def test_smoke_failures_remain_rejected(self):
        code = r'''
#include "Validation.h"
#include "Measurement.h"
#include <cassert>
#include <stdexcept>
#include <string>
#include <tuple>
int main() {
  using ModuleKey = std::pair<int, std::string>;
  using WorkloadKey = std::tuple<int, std::string, unsigned>;
  for (bool fail_numerical : {true, false}) {
    bench::ValidationCache<ModuleKey> numerical;
    bench::ValidationCache<WorkloadKey> workloads;
    std::map<ModuleKey, int> modules;
    int numerical_attempts = 0, workload_attempts = 0, timings = 0;
    const std::string reason = fail_numerical ? "numerical smoke" : "workload smoke";
    auto state = [&](const std::string &candidate, unsigned k) {
      ModuleKey key{0, candidate};
      try {
        numerical.ensure(key, [&] {
          // The injected failure would disappear on retry. It must stay rejected.
          if (candidate == "bad" && ++numerical_attempts == 1 && fail_numerical)
            throw std::runtime_error(reason);
          modules.emplace(key, 1);
        });
        assert(modules.count(key));
        workloads.ensure({0, candidate, k}, [&] {
          if (candidate == "bad" && k == 4 && ++workload_attempts == 1 && !fail_numerical)
            throw std::runtime_error(reason);
        });
        ++timings;
        return 0;
      } catch (const std::exception &e) {
        assert(e.what() == reason);
        bench::Measurement m;
        m.failed = true;
        return m.code();
      }
    };
    assert(state("good", 4) == 0);
    for (int round = 0; round < 3; ++round)
      assert(state("bad", 4) == 1);
    assert(state("bad", 1) == (fail_numerical ? 1 : 0));
    assert(state("good", 4) == 0);
    assert(numerical_attempts == 1);
    assert(workload_attempts == (fail_numerical ? 0 : 1));
    assert(timings == (fail_numerical ? 2 : 3));
    assert(modules.count({0, "bad"}) == (fail_numerical ? 0 : 1));
    // A failure on device 0 must not reject the same candidate on device 1.
    numerical.ensure({1, "bad"}, [] {});
    workloads.ensure({1, "bad", 4}, [] {});
  }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp)
            (p / "test.cpp").write_text(code)
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-I", str(HERE / "runner"), str(p / "test.cpp"),
                            "-o", str(p / "test")], check=True)
            subprocess.run([str(p / "test")], check=True)


if __name__ == "__main__":
    unittest.main()
