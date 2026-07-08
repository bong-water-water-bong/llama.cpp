# ggml-rocm: custom ROCm kernel library for 1-bit/ternary inference
#
# Strategy: set ALL sources as LANGUAGE HIP so CMake uses the HIP compiler
# (amdclang++) for everything. amdclang++ handles .cpp files as host C++
# and .hip files with device offload. This avoids C++ compiler flag leaks.

# ROCm root detection
if (NOT EXISTS $ENV{ROCM_PATH})
    if (EXISTS /opt/rocm)
        set(ROCM_PATH /opt/rocm)
    else()
        set(ROCM_PATH /usr)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

enable_language(HIP)
find_package(hip REQUIRED)

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "ggml-rocm: HIP ${hip_VERSION} at ${ROCM_PATH}")

if (NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx1151")
endif()

set(_R1B "${CMAKE_CURRENT_SOURCE_DIR}/../../../1bit")
if (NOT EXISTS "${_R1B}")
    set(_R1B "${CMAKE_CURRENT_SOURCE_DIR}/../../../../1bit")
endif()
message(STATUS "ggml-rocm: 1bit source = ${_R1B}")

set(ROCM_ALL_SRCS
    "ggml-rocm.cpp"
    "${_R1B}/src/kv_rotorquant.cpp"
    "${_R1B}/src/h1b_loader.cpp"
    "${_R1B}/src/tokenizer.cpp"
    "${_R1B}/src/prefill_dispatcher.cpp"
    "${_R1B}/src/ternary_gemv_launchers.hip"
    "${_R1B}/src/prefill_standalone.hip"
    "${_R1B}/src/wmma_peak_probe.hip"
    "${_R1B}/src/prim_kernels.hip"
    "${_R1B}/src/kv_cache_attn.hip"
    "${_R1B}/src/kv_cache_attn_i8.hip"
    "${_R1B}/src/kv_cache_attn_fd.hip"
    "${_R1B}/src/kv_cache_attn_fd_rotor.hip"
    "${_R1B}/src/ternary_gemm_smallm.hip"
    "${_R1B}/src/ternary_gemm_smallm_scalar_ref.hip"
    "${_R1B}/src/sherry_gemv.hip"
    "${_R1B}/src/sherry_gemv_scalar_ref.hip"
    "${_R1B}/src/bonsai_q1_gemv.hip"
    "${_R1B}/src/bonsai_tq2_gemv.hip"
    "${_R1B}/src/bonsai_gemv_scalar_ref.hip"
    "${_R1B}/src/medusa_tree_attn.hip"
    "${_R1B}/src/medusa_small_m_gemv.hip"
    "${_R1B}/kernels/ternary_gemv_phase5_dot4.hip"
    "${_R1B}/kernels/ternary_gemv_phase5_halo.hip"
    "${_R1B}/kernels/ternary_gemv_sherry.hip"
    "${_R1B}/kernels/ternary_gemv_tq1_halo.hip"
    "${_R1B}/kernels/rotorquant_pack.hip"
    "${_R1B}/kernels/hadamard_rotate_butterfly.hip"
    "${_R1B}/kernels/zaya_moe_ternary_gemv.hip"
    "${_R1B}/src/zaya_moe_launcher.hip"
)

# All sources as LANGUAGE HIP so amdclang++ handles everything
set_source_files_properties(${ROCM_ALL_SRCS} PROPERTIES LANGUAGE HIP)

add_library(ggml-rocm SHARED ${ROCM_ALL_SRCS})

target_include_directories(ggml-rocm PRIVATE
    "${_R1B}/include"
    "${ROCM_PATH}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/..")

target_compile_options(ggml-rocm PRIVATE
    -O3 -ffast-math -munsafe-fp-atomics
    -Wno-unused-result -Wno-unused-parameter -Wno-unused-variable)

target_link_libraries(ggml-rocm PRIVATE hip::host hip::device)

set_target_properties(ggml-rocm PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON)

message(STATUS "ggml-rocm: ${ROCM_ALL_SRCS} sources")
