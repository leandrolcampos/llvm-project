# Optional private LLVM external project. No changes to LLVM/libc defaults.
cmake_minimum_required(VERSION 3.30.4)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(GpuLibmBench LANGUAGES NONE)
endif()
get_filename_component(GPU_LIBM_BENCH_REPO "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
if(LLVM_RUNTIME_OUTPUT_INTDIR)
  set(default_tools "${LLVM_RUNTIME_OUTPUT_INTDIR}")
else()
  set(default_tools "${GPU_LIBM_BENCH_REPO}/build/bin")
endif()
set(GPU_LIBM_BENCH_TOOLS "${default_tools}" CACHE PATH "Clang/LLVM tools for measured kernels")
set(GPU_LIBM_BENCH_ARCH "sm_120" CACHE STRING "Explicit NVIDIA architecture (sm_120, sm_90, sm_89)")
set(GPU_LIBM_BENCH_REGISTRY "${CMAKE_CURRENT_SOURCE_DIR}/candidates.json" CACHE FILEPATH "Candidate registry")
set(GPU_LIBM_BENCH_CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA Toolkit")
set(GPU_LIBM_BENCH_K "1,2,4,8" CACHE STRING "Compiled independent stream counts")
if(CMAKE_CXX_COMPILER)
  set(default_host_cxx "${CMAKE_CXX_COMPILER}")
else()
  find_program(default_host_cxx NAMES c++ g++ REQUIRED)
endif()
set(GPU_LIBM_BENCH_HOST_CXX "${default_host_cxx}" CACHE FILEPATH "Native compiler compatible with nvcc")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(GPU_LIBM_BENCH_PYTHON "${Python3_EXECUTABLE}" CACHE INTERNAL "")
include(ExternalProject)

# Defer until tools/runtimes have been registered by the enclosing LLVM build.
function(gpu_libm_bench_setup)
  set(bridge "${GPU_LIBM_BENCH_BRIDGE_BINARY}")
  get_filename_component(repo "${GPU_LIBM_BENCH_SOURCE}/.." ABSOLUTE)
  set(common --repo "${repo}" --tools "${GPU_LIBM_BENCH_TOOLS}"
    --cuda "${GPU_LIBM_BENCH_CUDA_ROOT}" --arch "${GPU_LIBM_BENCH_ARCH}"
    --registry "${GPU_LIBM_BENCH_REGISTRY}" --output "${bridge}/device"
    --streams "${GPU_LIBM_BENCH_K}")
  set(tool_deps)
  get_filename_component(selected_tools "${GPU_LIBM_BENCH_TOOLS}" ABSOLUTE)
  if(LLVM_RUNTIME_OUTPUT_INTDIR)
    get_filename_component(build_tools "${LLVM_RUNTIME_OUTPUT_INTDIR}" ABSOLUTE)
    if(selected_tools STREQUAL build_tools)
      foreach(tool clang llvm-link opt llc llvm-nm llvm-ar llvm-ranlib)
        if(TARGET ${tool})
          list(APPEND tool_deps ${tool})
        endif()
      endforeach()
    endif()
  endif()
  add_custom_target(gpu-libm-bench-kernels
    COMMAND "${GPU_LIBM_BENCH_PYTHON}" "${GPU_LIBM_BENCH_SOURCE}/build_device.py" ${common}
      --c-launcher "${CMAKE_C_COMPILER_LAUNCHER}"
      --cxx-launcher "${CMAKE_CXX_COMPILER_LAUNCHER}"
    DEPENDS ${tool_deps} USES_TERMINAL VERBATIM)
  string(REPLACE "sm_" "" cuda_arch "${GPU_LIBM_BENCH_ARCH}")
  # ExternalProject flattens lists in CMAKE_ARGS. Encode them until its command
  # generation restores the actual semicolon-separated launcher arguments.
  if(CMAKE_CUDA_COMPILER_LAUNCHER)
    set(cuda_launcher "${CMAKE_CUDA_COMPILER_LAUNCHER}")
  else()
    set(cuda_launcher "${CMAKE_CXX_COMPILER_LAUNCHER}")
  endif()
  set(launcher_separator "__GPU_LIBM_BENCH_LIST__")
  set(native_arguments "${CMAKE_CXX_COMPILER_LAUNCHER}${cuda_launcher}${GPU_LIBM_BENCH_HOST_CXX}${GPU_LIBM_BENCH_CUDA_ROOT}${GPU_LIBM_BENCH_REGISTRY}${bridge}")
  string(FIND "${native_arguments}" "${launcher_separator}" separator_position)
  while(NOT separator_position EQUAL -1)
    string(APPEND launcher_separator "_")
    string(FIND "${native_arguments}" "${launcher_separator}" separator_position)
  endwhile()
  string(REPLACE ";" "${launcher_separator}" native_cxx_launcher "${CMAKE_CXX_COMPILER_LAUNCHER}")
  string(REPLACE ";" "${launcher_separator}" native_cuda_launcher "${cuda_launcher}")
  ExternalProject_Add(gpu-libm-bench-runner
    SOURCE_DIR "${GPU_LIBM_BENCH_SOURCE}/runner"
    BINARY_DIR "${bridge}/host"
    LIST_SEPARATOR "${launcher_separator}"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
      "-DCMAKE_CXX_COMPILER=${GPU_LIBM_BENCH_HOST_CXX}"
      "-DCMAKE_CXX_COMPILER_LAUNCHER=${native_cxx_launcher}"
      "-DCMAKE_CUDA_COMPILER=${GPU_LIBM_BENCH_CUDA_ROOT}/bin/nvcc"
      "-DCMAKE_CUDA_HOST_COMPILER=${GPU_LIBM_BENCH_HOST_CXX}"
      "-DCMAKE_CUDA_COMPILER_LAUNCHER=${native_cuda_launcher}"
      "-DCMAKE_CUDA_ARCHITECTURES=${cuda_arch}"
      "-DCUDAToolkit_ROOT=${GPU_LIBM_BENCH_CUDA_ROOT}"
      "-DBENCH_DEVICE_DIR=${bridge}/device"
      "-DBENCH_REGISTRY=${GPU_LIBM_BENCH_REGISTRY}"
    BUILD_ALWAYS TRUE INSTALL_COMMAND "" EXCLUDE_FROM_ALL TRUE
    BUILD_BYPRODUCTS "${bridge}/host/gpu-libm-bench")
  add_custom_target(gpu-libm-bench DEPENDS gpu-libm-bench-runner gpu-libm-bench-kernels)
endfunction()
set(GPU_LIBM_BENCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")
set(GPU_LIBM_BENCH_BRIDGE_BINARY "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL gpu_libm_bench_setup)
