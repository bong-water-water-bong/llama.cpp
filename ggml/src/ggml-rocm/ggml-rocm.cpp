// ggml-rocm: ROCm custom kernel library for 1-bit/ternary inference
//
// AMD Strix Halo (gfx1151) hand-tuned HIP kernels folded into llama.cpp.
// Provides C API functions callable from model-loading code.
//
// Kernels:
//   Ternary GEMV, Bonsai Q1/TQ2 GEMV, Sherry GEMV, KV cache attention,
//   Prefill GEMM, Medusa, model loader, tokenizer, FP16 helpers, etc.

#include <hip/hip_runtime.h>
#include <cstdio>

// =========================================================================
// Version info
// =========================================================================

extern "C" {

const char * ggml_rocm_backend_name(void) { return "ROCM"; }
const char * ggml_rocm_backend_version(void) { return "0.1.0"; }

// Check if a ROCm/HIP device is available and compatible
int ggml_rocm_available(void) {
    int nd = 0;
    hipGetDeviceCount(&nd);
    if (nd == 0) return 0;
    hipSetDevice(0);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "ggml-rocm: device 0: %s (compute %d.%d)\n",
            prop.name, prop.major, prop.minor);
    return 1;
}

// Print device info for all HIP devices
void ggml_rocm_print_devices(void) {
    int nd = 0;
    hipGetDeviceCount(&nd);
    fprintf(stderr, "ggml-rocm: %d HIP device(s) found:\n", nd);
    for (int i = 0; i < nd; i++) {
        hipSetDevice(i);
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        size_t free = 0, total = 0;
        hipMemGetInfo(&free, &total);
        fprintf(stderr, "  [%d] %s | VRAM: %.1f GB free / %.1f GB total\n",
                i, prop.name,
                (double)free / (1<<30), (double)total / (1<<30));
    }
}

} // extern "C"

// =========================================================================
// The actual kernel launchers are defined in the .hip source files.
// This TU exists to provide the C API entry points and version info.
// Kernel symbols are exported from the .hip compilation units and are
// available via dlsym() on libggml-rocm.so.
// =========================================================================
