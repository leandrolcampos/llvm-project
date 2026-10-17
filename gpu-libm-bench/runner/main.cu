// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "Inputs.h"
#include "Measurement.h"
#include "Validation.h"
#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <nvbench/main.cuh>
#include <nvbench/nvbench.cuh>
#include <nvml.h>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <tuple>
#include <unistd.h>

using json = nlohmann::json;
static void driver(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *s = nullptr;
    cuGetErrorString(result, &s);
    throw std::runtime_error(s ? s : "CUDA Driver API failure");
  }
}
static void runtime(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
struct Options {
  std::uint64_t seed = 1, count = 1 << 20;
  unsigned round = 0, order = 0;
  std::string csv, schedule, clock_policy = "unlocked; externally managed";
  std::string registry = BENCH_REGISTRY, device_dir = BENCH_DEVICE_DIR;
  bool inputs = false, json_lines = false;
  int info_device = -1;
  std::string expected_uuid, expected_arch, expected_driver;
};
struct Module {
  CUmodule handle = nullptr;
  explicit Module(const std::string &path) {
    driver(cuModuleLoad(&handle, path.c_str()));
  }
  ~Module() {
    if (handle)
      cuModuleUnload(handle);
  }
  CUfunction function(const std::string &name) {
    CUfunction f;
    driver(cuModuleGetFunction(&f, handle, name.c_str()));
    return f;
  }
};
struct Buffer {
  CUdeviceptr ptr = 0;
  explicit Buffer(std::size_t n) { driver(cuMemAlloc(&ptr, n)); }
  ~Buffer() {
    if (ptr)
      cuMemFree(ptr);
  }
};
static json telemetry(int id) {
  json j;
  char bus[64];
  if (cudaDeviceGetPCIBusId(bus, sizeof(bus), id) != cudaSuccess)
    return j;
  nvmlDevice_t d;
  if (nvmlInit_v2() != NVML_SUCCESS ||
      nvmlDeviceGetHandleByPciBusId_v2(bus, &d) != NVML_SUCCESS)
    return j;
  unsigned value;
  if (nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &value) == NVML_SUCCESS)
    j["sm_mhz"] = value;
  if (nvmlDeviceGetClockInfo(d, NVML_CLOCK_MEM, &value) == NVML_SUCCESS)
    j["memory_mhz"] = value;
  if (nvmlDeviceGetTemperature(d, NVML_TEMPERATURE_GPU, &value) == NVML_SUCCESS)
    j["temperature_c"] = value;
  if (nvmlDeviceGetPowerUsage(d, &value) == NVML_SUCCESS)
    j["power_mw"] = value;
  unsigned long long reasons;
  if (nvmlDeviceGetCurrentClocksThrottleReasons(d, &reasons) == NVML_SUCCESS)
    j["throttle_reasons"] = reasons;
  nvmlShutdown();
  return j;
}

// Independent host double libm reference, not a certification of rounding.
static void smoke(Module &module, const std::vector<float> &input) {
  std::vector<float> x;
  for (std::size_t i = 0; i < std::min<std::size_t>(input.size(), 2048); ++i) {
    x.push_back(input[i]);
    x.push_back(-input[i]);
  }
  x.insert(x.end(), {-10.f, -1.f, -0.f, 0.f, 1.f, 10.f});
  Buffer in(x.size() * sizeof(float)), out(x.size() * sizeof(float));
  driver(cuMemcpyHtoD(in.ptr, x.data(), x.size() * sizeof(float)));
  unsigned n = unsigned(x.size());
  void *args[] = {&in.ptr, &out.ptr, &n};
  driver(cuLaunchKernel(module.function("expf_probe"), (n + 127) / 128, 1, 1,
                        128, 1, 1, 0, nullptr, args, nullptr));
  std::vector<float> y(n);
  driver(cuMemcpyDtoH(y.data(), out.ptr, n * sizeof(float)));
  for (unsigned i = 0; i < n; ++i) {
    double ref = std::exp(double(x[i]));
    if (!std::isfinite(y[i]) || std::abs(double(y[i]) - ref) > 4e-6 * ref)
      throw std::runtime_error("numerical smoke failed at x=" +
                               std::to_string(x[i]));
  }
}

// Check the measured loop against the same candidate's independently launched
// probe, including sign order, corpus wraparound, observable stores and R.
static void smoke_workload(Module &module, const std::vector<float> &input,
                           unsigned k) {
  const unsigned n = 32 * k;
  std::vector<float> signed_inputs(2 * n), values(2 * n);
  for (unsigned j = 0; j < n; ++j) {
    float x = input[j & (input.size() - 1)];
    signed_inputs[2 * j] = -x;
    signed_inputs[2 * j + 1] = x;
  }
  Buffer probe_in(2 * n * sizeof(float)), probe_out(2 * n * sizeof(float));
  driver(
      cuMemcpyHtoD(probe_in.ptr, signed_inputs.data(), 2 * n * sizeof(float)));
  unsigned probe_n = 2 * n;
  void *probe_args[] = {&probe_in.ptr, &probe_out.ptr, &probe_n};
  driver(cuLaunchKernel(module.function("expf_probe"), (probe_n + 127) / 128, 1,
                        1, 128, 1, 1, 0, nullptr, probe_args, nullptr));
  driver(cuMemcpyDtoH(values.data(), probe_out.ptr, 2 * n * sizeof(float)));
  Buffer in(input.size() * sizeof(float)), out(n * sizeof(std::uint32_t));
  driver(cuMemcpyHtoD(in.ptr, input.data(), input.size() * sizeof(float)));
  unsigned mask = unsigned(input.size() - 1);
  CUfunction f = module.function("expf_k" + std::to_string(k));
  for (std::uint64_t r : {1, 2, 7}) {
    void *args[] = {&in.ptr, &out.ptr, &r, &mask};
    driver(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
    std::vector<std::uint32_t> actual(n);
    driver(cuMemcpyDtoH(actual.data(), out.ptr, n * sizeof(std::uint32_t)));
    for (unsigned j = 0; j < n; ++j) {
      std::uint32_t h = 0;
      for (std::uint64_t i = 0; i < r; ++i)
        h = ((h << 1) | (h >> 31)) + bench::bits(values[2 * j + i % 2]);
      if (actual[j] != h)
        throw std::runtime_error("workload checksum/count smoke failed for K=" +
                                 std::to_string(k));
    }
  }
}

static std::string cell(const std::string &s) {
  std::string result = "\"";
  for (char c : s) {
    if (c == '"')
      result += '"';
    result += c;
  }
  return result + '"';
}

struct ProductLock {
  int fd;
  explicit ProductLock(const std::string &directory) {
    fd = open((directory + "/.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC,
              0666);
    if (fd < 0 || flock(fd, LOCK_SH | LOCK_NB) != 0) {
      if (fd >= 0)
        close(fd);
      throw std::runtime_error(
          "device products are unavailable or being rebuilt");
    }
  }
  ~ProductLock() { close(fd); }
};

static std::string nvidia_driver_version() {
  auto result = nvmlInit_v2();
  if (result != NVML_SUCCESS)
    throw std::runtime_error(std::string("NVIDIA driver identification: ") +
                             nvmlErrorString(result));
  char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
  result = nvmlSystemGetDriverVersion(version, sizeof(version));
  nvmlShutdown();
  if (result != NVML_SUCCESS || !version[0])
    throw std::runtime_error(
        std::string("NVIDIA driver identification: ") +
        (result == NVML_SUCCESS ? "empty release" : nvmlErrorString(result)));
  return version;
}

static json device_metadata(int id) {
  cudaDeviceProp prop;
  runtime(cudaGetDeviceProperties(&prop, id));
  std::ostringstream uuid;
  uuid << "GPU-" << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10)
      uuid << '-';
    uuid << std::setw(2)
         << unsigned(static_cast<unsigned char>(prop.uuid.bytes[i]));
  }
  int dv, rv;
  runtime(cudaDriverGetVersion(&dv));
  runtime(cudaRuntimeGetVersion(&rv));
  return {{"gpu", prop.name},
          {"gpu_uuid", uuid.str()},
          {"architecture",
           "sm_" + std::to_string(prop.major) + std::to_string(prop.minor)},
          {"cuda_driver_api_version", dv},
          {"nvidia_driver_version", nvidia_driver_version()},
          {"runtime_version", rv},
          {"host_compiler", __VERSION__},
          {"nvcc_version", std::to_string(__CUDACC_VER_MAJOR__) + '.' +
                               std::to_string(__CUDACC_VER_MINOR__) + '.' +
                               std::to_string(__CUDACC_VER_BUILD__)}};
}

static bench::Measurement measurement(const nvbench::state &s, bool failed) {
  bench::Measurement m;
  m.failed = failed;
  m.skipped = s.is_skipped();
  m.target = s.get_batch_target_time();
  m.timeout = s.get_timeout();
  m.effective_minimum = m.requested_minimum = s.get_min_samples();
  try {
    m.effective_minimum =
        std::max(m.effective_minimum,
                 s.get_summary("nv/cold/sample_size").get_int64("value"));
  } catch (const std::exception &) {
  }
  try {
    m.seconds = s.get_summary("nv/batch/time/gpu/mean").get_float64("value");
    m.launches = s.get_summary("nv/batch/sample_size").get_int64("value");
    m.wall_seconds = s.get_summary("nv/batch/walltime").get_float64("value");
    m.has_batch = true;
  } catch (const std::exception &) {
  }
  return m;
}

int main(int argc, char **argv) try {
  Options options;
  std::vector<std::string> args{argv[0]};
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&](const char *key) -> std::string {
      std::string prefix = std::string(key) + '=';
      return arg.compare(0, prefix.size(), prefix) == 0
                 ? arg.substr(prefix.size())
                 : "";
    };
    if (arg == "--inputs")
      options.inputs = true;
    else if (arg == "--json-lines")
      options.json_lines = true;
    else if (!value("--device-info").empty())
      options.info_device = std::stoi(value("--device-info"));
    else if (!value("--expect-uuid").empty())
      options.expected_uuid = value("--expect-uuid");
    else if (!value("--expect-arch").empty())
      options.expected_arch = value("--expect-arch");
    else if (!value("--expect-driver").empty())
      options.expected_driver = value("--expect-driver");
    else if (!value("--seed").empty())
      options.seed = std::stoull(value("--seed"));
    else if (!value("--corpus-size").empty())
      options.count = std::stoull(value("--corpus-size"));
    else if (!value("--csv").empty())
      options.csv = value("--csv");
    else if (!value("--schedule").empty())
      options.schedule = value("--schedule");
    else if (!value("--registry").empty())
      options.registry = value("--registry");
    else if (!value("--device-dir").empty())
      options.device_dir = value("--device-dir");
    else if (!value("--round").empty())
      options.round = std::stoul(value("--round"));
    else if (!value("--order").empty())
      options.order = std::stoul(value("--order"));
    else if (!value("--clock-policy").empty())
      options.clock_policy = value("--clock-policy");
    else
      args.push_back(arg);
  }
  // Device identification is a runtime-only preflight, without a corpus or
  // timing.
  if (options.info_device >= 0) {
    driver(cuInit(0));
    runtime(cudaSetDevice(options.info_device));
    std::cout << device_metadata(options.info_device).dump() << '\n';
    return 0;
  }
  const auto inputs = bench::corpus(options.seed, options.count);
  const auto fingerprint = bench::fingerprint(inputs);
  if (options.inputs) {
    std::cout << "seed=" << options.seed << " count=" << options.count
              << " fnv1a64=" << std::hex << fingerprint << " prefix=";
    for (std::size_t i = 0; i < std::min<std::size_t>(inputs.size(), 16); ++i)
      std::cout << std::setw(8) << std::setfill('0') << bench::bits(inputs[i])
                << ' ';
    std::cout << '\n';
    return 0;
  }
  ProductLock product_lock(options.device_dir);
  json candidates;
  std::ifstream(options.registry) >> candidates;
  std::vector<std::string> ids;
  for (const auto &c : candidates)
    ids.push_back(c.at("id").get<std::string>());
  std::ofstream csv;
  if (!options.csv.empty()) {
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv.open(options.csv);
  }
  bool wrote_header = false;
  std::size_t emitted_states = 0;
  int exit_status = 0;
  std::map<int, json> devices;
  auto emit = [&](nvbench::state &state, std::uint64_t count,
                  const json &before, const json &after,
                  const std::string &failure) {
    const auto m = measurement(state, !failure.empty());
    int code = m.code();
    if (code == 1 || exit_status == 1)
      exit_status = 1;
    else
      exit_status = std::max(exit_status, code);
    json row = devices.at(state.get_device()->get_id());
    row.update({{"id", state.get_string("Candidate")},
                {"seed", options.seed},
                {"corpus_size", options.count},
                {"corpus_fnv1a64", fingerprint},
                {"C", count},
                {"timing", "Batch GPU"},
                {"batch_gpu_seconds", m.seconds},
                {"gcalls_s", m.seconds > 0 && std::isfinite(m.seconds)
                                 ? double(count) / (1e9 * m.seconds)
                                 : 0},
                {"launches", m.launches},
                {"round", state.get_int64("Round")},
                {"order", state.get_int64("Order")},
                {"status", code == 0   ? "complete"
                           : code == 2 ? "incomplete"
                                       : "failed"},
                {"reason", !failure.empty()     ? failure
                           : state.is_skipped() ? state.get_skip_reason()
                                                : ""},
                {"batch_target_seconds", m.target},
                {"min_samples", m.requested_minimum},
                {"effective_min_samples", m.effective_minimum},
                {"batch_wall_seconds", m.wall_seconds},
                {"timeout_seconds", m.timeout},
                {"timeout_limit_reached", m.wall_seconds >= m.timeout},
                {"clock_policy", options.clock_policy},
                {"before", before},
                {"after", after}});
    for (const char *axis : {"B", "T", "K", "R"})
      row[axis] = state.get_int64(axis);
    if (csv.is_open()) {
      // JSON objects have stable sorted keys. Flush every completed state.
      if (!wrote_header) {
        bool first = true;
        for (auto i = row.begin(); i != row.end(); ++i) {
          if (!first)
            csv << ',';
          csv << cell(i.key());
          first = false;
        }
        csv << '\n';
        wrote_header = true;
      }
      bool first = true;
      for (const auto &v : row) {
        if (!first)
          csv << ',';
        csv << cell(v.is_string() ? v.get<std::string>() : v.dump());
        first = false;
      }
      csv << '\n';
      csv.flush();
    }
    if (options.json_lines)
      std::cout << "\nGPU_LIBM_ROW " << row.dump() << std::endl;
    ++emitted_states;
  };
  std::map<std::pair<int, std::string>, std::shared_ptr<Module>> modules;
  bench::ValidationCache<std::pair<int, std::string>> checked_modules;
  bench::ValidationCache<std::tuple<int, std::string, unsigned>>
      checked_workloads;
  std::function<void(nvbench::state &, nvbench::type_list<>)> measure =
      [&](nvbench::state &state, nvbench::type_list<>) {
        const auto &device = state.get_device().value();
        const int id = device.get_id();
        if (!devices.count(id))
          devices[id] = device_metadata(id);
        std::uint64_t count = 0;
        json before, after;
        std::string failure;
        try {
          if ((!options.expected_uuid.empty() &&
               devices[id].at("gpu_uuid") != options.expected_uuid) ||
              (!options.expected_arch.empty() &&
               devices[id].at("architecture") != options.expected_arch) ||
              (!options.expected_driver.empty() &&
               devices[id].at("nvidia_driver_version") !=
                   options.expected_driver))
            throw std::runtime_error("GPU/driver differs from preflight/pilot");
          const std::string candidate = state.get_string("Candidate");
          std::uint64_t b = state.get_int64("B"), t = state.get_int64("T"),
                        k = state.get_int64("K"), r = state.get_int64("R");
          count = bench::work(b, t, k, r);
          if (t > unsigned(device.get_max_threads_per_block()) ||
              b > INT32_MAX || b * t > UINT32_MAX || k > 32)
            throw std::invalid_argument(
                "kernel launch dimensions exceed workload limits");
          runtime(cudaFree(nullptr));
          const auto module_key = std::make_pair(id, candidate);
          checked_modules.ensure(module_key, [&] {
            auto module = std::make_shared<Module>(options.device_dir + '/' +
                                                   candidate + "/kernel.cubin");
            smoke(*module, inputs);
            modules.emplace(module_key, std::move(module));
          });
          auto &module = modules.at(module_key);
          CUfunction f = module->function("expf_k" + std::to_string(k));
          checked_workloads.ensure({id, candidate, unsigned(k)}, [&] {
            smoke_workload(*module, inputs, unsigned(k));
          });
          Buffer in(inputs.size() * sizeof(float)),
              out(b * t * k * sizeof(std::uint32_t));
          driver(cuMemcpyHtoD(in.ptr, inputs.data(),
                              inputs.size() * sizeof(float)));
          unsigned mask = unsigned(inputs.size() - 1);
          void *params[] = {&in.ptr, &out.ptr, &r, &mask};
          before = telemetry(id);
          state.add_element_count(static_cast<nvbench::int64_t>(count));
          state.exec(nvbench::exec_tag::gpu, [&](nvbench::launch &launch) {
            driver(cuLaunchKernel(f, unsigned(b), 1, 1, unsigned(t), 1, 1, 0,
                                  launch.get_stream(), params, nullptr));
          });
          after = telemetry(id);
        } catch (const std::exception &e) {
          // NVBench marks policy skips (e.g. skip_time) before throwing.
          // Those are incomplete measurements, not execution failures.
          if (!state.is_skipped()) {
            failure = e.what();
            state.skip(failure);
          }
        }
        emit(state, count, before, after, failure);
      };
  if (options.schedule.empty()) {
    auto benchmark =
        std::make_unique<nvbench::benchmark<decltype(measure)>>(measure);
    benchmark->set_name("expf")
        .add_string_axis("Candidate", ids)
        .add_int64_axis("B", {256})
        .add_int64_axis("T", {128})
        .add_int64_axis("K", {4})
        .add_int64_axis("R", {1024})
        .add_int64_axis("Round", {options.round})
        .add_int64_axis("Order", {options.order});
    nvbench::benchmark_manager::get().add(std::move(benchmark));
  } else {
    json schedule;
    std::ifstream(options.schedule) >> schedule;
    for (const auto &point : schedule) {
      std::string id = point.at("id");
      if (std::find(ids.begin(), ids.end(), id) == ids.end())
        throw std::invalid_argument("unknown candidate");
      auto benchmark =
          std::make_unique<nvbench::benchmark<decltype(measure)>>(measure);
      benchmark
          ->set_name("expf_" +
                     std::to_string(point.at("round").get<unsigned>()) + '_' +
                     std::to_string(point.at("order").get<unsigned>()))
          .add_string_axis("Candidate", {id});
      for (const char *axis : {"B", "T", "K", "R"})
        benchmark->add_int64_axis(axis,
                                  {point.at(axis).get<nvbench::int64_t>()});
      benchmark
          ->add_int64_axis("Round", {point.at("round").get<nvbench::int64_t>()})
          .add_int64_axis("Order", {point.at("order").get<nvbench::int64_t>()});
      nvbench::benchmark_manager::get().add(std::move(benchmark));
    }
  }
  driver(cuInit(0));
  nvbench::detail::main_initialize(argc, argv);
  nvbench::option_parser parser;
  parser.parse(args);
  nvbench::detail::main_print_preamble(parser);
  nvbench::detail::main_run_benchmarks(parser);
  std::size_t expected_states = 0;
  for (const auto &benchmark : parser.get_benchmarks())
    expected_states += benchmark->get_states().size();
  if (expected_states == 0 || expected_states != emitted_states)
    exit_status = 1;
  nvbench::detail::main_print_epilogue(parser);
  nvbench::detail::main_print_results(parser);
  for (auto &entry : modules) {
    runtime(cudaSetDevice(entry.first.first));
    entry.second.reset();
  }
  nvbench::detail::main_finalize();
  return exit_status;
} catch (const std::exception &e) {
  std::cerr << "gpu-libm-bench: " << e.what() << '\n';
  return 1;
}
