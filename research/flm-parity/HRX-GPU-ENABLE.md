# Enabling the ggml-hrx GPU device (AMD Radeon 8060S / gfx1151) — run recipe

## Why this was needed (root cause, fixed 2026-09-05)

1. The refreshed fork builds its own hrx runtime in-tree (ExternalProject
   ggml-hrx-deps from HRX_SOURCE_DIR). That build ran with an EMPTY
   IREE_ROCM_PATH, so the IREE AMDGPU HAL driver never compiled and libhrx
   shipped with HRX_HAS_IREE_AMDGPU_DRIVER undefined (hrx_gpu_initialize ->
   code 3 INVALID_ARGUMENT, "built without AMDGPU support").
2. Even with the driver compiled, hrx dlopens libhsa-runtime64.so.1 from the
   system (/usr/lib -> 1.18.0) which fails the HSA_AMD_AGENT_INFO_PM4_EMULATION
   query. TheRock libhsa (1.21) works; put its dir first on LD_LIBRARY_PATH.
3. llama-cli/llama-server constructed the ggml backend registry during
   common_params_parse (the -ngl/-sm/-ts/-mg/-ngld option setters called
   llama_supports_gpu_offload()), registering HRX with 0 devices before
   llama_backend_init() — cached for the whole process, so those tools
   silently fell back to CPU. Fixed in commit c80f41f (no parse-time registry
   init; accurate warning at model load).
4. llama-bench never had the parse-time call, so it registered HRX correctly.
   Its "backend HRX" column lists REGISTERED backends, not the backend the
   test actually used — with a broken HRX it silently reported CPU numbers.
   Always verify real offload (offload log lines at -lv, or /dev/dri maps).

## Recipe

    export ROCMLIB=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib
    export LD_LIBRARY_PATH=$ROCMLIB:$PWD/build/bin
    # one-time: rebuild deps WITH amdgpu
    cmake -S . -B build -DIREE_ROCM_PATH=$ROCMLIB
    cmake --build build --target ggml-hrx-deps -j 16

Verify: ./build/bin/regprobe (research/flm-parity/regprobe.cpp) ->
  "ggml_hrx: init OK", "dev 0: HRX0 ... AMD Radeon 8060S Graphics (gfx1151)",
  "supports_gpu_offload: 1"

Run: llama-server / llama-bench / llama-cli with LD_LIBRARY_PATH above and -ngl 99.
