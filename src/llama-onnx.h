// llama-onnx.h - ONNX Runtime inference engine integration
//
// Provides C++ API for loading and running ONNX models alongside GGUF inference.
// Uses ONNX Runtime C++ API for cross-platform model execution.
//
// Supported execution providers (in priority order):
//   1. ROCm (AMD GPU) - if available at build time
//   2. CUDA - if available at build time  
//   3. DNNL (oneDNN/Intel)
//   4. CPU (fallback)

#pragma once

#include "llama.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

// Forward declare ONNX Runtime types
struct OrtSession;
struct OrtSessionOptions;
struct OrtMemoryInfo;
struct OrtValue;
struct OrtRunOptions;
struct OrtEnv;
struct OrtIoBinding;

namespace llama {

// ONNX tensor data types
enum class onnx_tensor_type {
    ONNX_TYPE_F32,
    ONNX_TYPE_F16,
    ONNX_TYPE_I32,
    ONNX_TYPE_I64,
    ONNX_TYPE_U8,
    ONNX_TYPE_UNKNOWN,
};

// ONNX execution provider
enum class onnx_provider {
    CPU,
    DNNL,
    ROCM,
    CUDA,
    AUTO, // auto-select best available
};

// Tensor info for ONNX model
struct onnx_tensor_info {
    std::string               name;
    onnx_tensor_type          type;
    std::vector<int64_t>      shape;
    size_t                    num_elements;
};

// ONNX model runner
class onnx_model {
public:
    onnx_model();
    ~onnx_model();

    // Load an ONNX model from file
    // Returns true on success
    bool load(const std::string & model_path, onnx_provider provider = onnx_provider::AUTO);

    // Check if model is loaded
    bool is_loaded() const;

    // Get model info
    const std::string & model_path() const;
    std::vector<onnx_tensor_info> input_info() const;
    std::vector<onnx_tensor_info> output_info() const;

    // Run inference
    // inputs: map of input_name -> pointer to float data
    // outputs: map of output_name -> pointer to pre-allocated buffer
    // All tensors are assumed to be float32 with shape matching model expectations
    bool run(const std::unordered_map<std::string, float *> & inputs,
             const std::unordered_map<std::string, float *> & outputs);

    // Run inference with shape info
    bool run_with_shapes(
        const std::unordered_map<std::string, float *> & inputs,
        const std::unordered_map<std::string, std::vector<int64_t>> & input_shapes,
        const std::unordered_map<std::string, float *> & outputs,
        const std::unordered_map<std::string, std::vector<int64_t>> & output_shapes);

    // Get the loaded model's input/output count
    size_t num_inputs() const;
    size_t num_outputs() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Check if ONNX Runtime is available (library loaded successfully)
bool onnx_available();

// Get ONNX Runtime version string
std::string onnx_version();

// List available execution providers
std::vector<onnx_provider> onnx_available_providers();

// Convert onnx_provider to string
const char * onnx_provider_name(onnx_provider provider);

} // namespace llama
