// llama-onnx.cpp - ONNX Runtime C++ inference implementation
//
// Integrates with onnxruntime (ORT) shared library.
// Supports CPU, DNNL, and optionally ROCm/CUDA providers.

#include "llama-onnx.h"

#ifdef LLAMA_ONNX

#include <algorithm>
#include <cstring>
#include <mutex>
#include <sstream>

#include <onnxruntime_cxx_api.h>
#include <dnnl_provider_options.h>

// Internal implementation
struct llama::onnx_model::impl {
    std::string                         _model_path;
    std::shared_ptr<Ort::Env>           _env;
    std::shared_ptr<Ort::Session>       _session;
    std::vector<onnx_tensor_info>       _inputs;
    std::vector<onnx_tensor_info>       _outputs;
    std::vector<const char *>           _input_names;
    std::vector<const char *>           _output_names;
    Ort::MemoryInfo                     _memory_info{nullptr};
    onnx_provider                       _provider = onnx_provider::CPU;
    bool                                _loaded = false;
    std::mutex                          _mutex;

    ~impl() = default;

    static onnx_tensor_type ort_type_to_llama(ONNXTensorElementDataType ort_type) {
        switch (ort_type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return onnx_tensor_type::ONNX_TYPE_F32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return onnx_tensor_type::ONNX_TYPE_F16;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return onnx_tensor_type::ONNX_TYPE_I32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return onnx_tensor_type::ONNX_TYPE_I64;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:  return onnx_tensor_type::ONNX_TYPE_U8;
            default:                                   return onnx_tensor_type::ONNX_TYPE_UNKNOWN;
        }
    }

    bool load_model(const std::string & model_path, onnx_provider provider) {
        std::lock_guard<std::mutex> lock(_mutex);

        _model_path = model_path;
        _provider = provider;

        try {
            _env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "llama-onnx");

            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(4);
            session_options.SetInterOpNumThreads(2);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            bool provider_set = false;

            if (provider == onnx_provider::AUTO || provider == onnx_provider::ROCM) {
                try {
                    OrtROCMProviderOptions rocm_opts{};
                    rocm_opts.device_id = 0;
                    session_options.AppendExecutionProvider_ROCM(rocm_opts);
                    _provider = onnx_provider::ROCM;
                    provider_set = true;
                } catch (const Ort::Exception &) {}
            }

            if (!provider_set && (provider == onnx_provider::AUTO || provider == onnx_provider::CUDA)) {
                try {
                    OrtCUDAProviderOptions cuda_opts{};
                    cuda_opts.device_id = 0;
                    session_options.AppendExecutionProvider_CUDA(cuda_opts);
                    _provider = onnx_provider::CUDA;
                    provider_set = true;
                } catch (const Ort::Exception &) {}
            }

            if (!provider_set && (provider == onnx_provider::AUTO || provider == onnx_provider::DNNL)) {
                try {
                    OrtDnnlProviderOptions dnnl_opts{};
                    session_options.AppendExecutionProvider_Dnnl(dnnl_opts);
                    _provider = onnx_provider::DNNL;
                    provider_set = true;
                } catch (const Ort::Exception &) {}
            }

            if (!provider_set) {
                _provider = onnx_provider::CPU;
            }

            _session = std::make_shared<Ort::Session>(*_env, model_path.c_str(), session_options);
            _memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::AllocatorWithDefaultOptions allocator;

            size_t num_inputs = _session->GetInputCount();
            size_t num_outputs = _session->GetOutputCount();

            _inputs.clear();
            _outputs.clear();
            _input_names.clear();
            _output_names.clear();

            for (size_t i = 0; i < num_inputs; ++i) {
                auto name = _session->GetInputNameAllocated(i, allocator);
                auto type_info = _session->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                onnx_tensor_info info;
                info.name = name.get();
                info.type = ort_type_to_llama(tensor_info.GetElementType());
                info.shape = tensor_info.GetShape();
                info.num_elements = 1;
                for (auto d : info.shape) {
                    if (d > 0) info.num_elements *= d;
                }

                _inputs.push_back(info);
                _input_names.push_back(_inputs.back().name.c_str());
            }

            for (size_t i = 0; i < num_outputs; ++i) {
                auto name = _session->GetOutputNameAllocated(i, allocator);
                auto type_info = _session->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                onnx_tensor_info info;
                info.name = name.get();
                info.type = ort_type_to_llama(tensor_info.GetElementType());
                info.shape = tensor_info.GetShape();
                info.num_elements = 1;
                for (auto d : info.shape) {
                    if (d > 0) info.num_elements *= d;
                }

                _outputs.push_back(info);
                _output_names.push_back(_outputs.back().name.c_str());
            }

            _loaded = true;
            return true;

        } catch (const Ort::Exception & e) {
            fprintf(stderr, "llama-onnx: Failed to load model '%s': %s\n",
                    model_path.c_str(), e.what());
            _loaded = false;
            return false;
        }
    }

    bool run_model(
        const std::unordered_map<std::string, float *> & inputs,
        const std::unordered_map<std::string, float *> & outputs) {

        if (!_loaded || !_session) { return false; }

        std::lock_guard<std::mutex> lock(_mutex);

        try {
            std::vector<Ort::Value> ort_inputs;
            std::vector<const char *> run_input_names;

            for (const auto & input : _inputs) {
                auto it = inputs.find(input.name);
                if (it == inputs.end()) { return false; }

                Ort::Value tensor = Ort::Value::CreateTensor<float>(
                    _memory_info, const_cast<float *>(it->second), input.num_elements,
                    input.shape.data(), input.shape.size());

                ort_inputs.push_back(std::move(tensor));
                run_input_names.push_back(input.name.c_str());
            }

            std::vector<const char *> run_output_names;
            for (const auto & output : _outputs) {
                auto it = outputs.find(output.name);
                if (it == outputs.end()) { return false; }
                run_output_names.push_back(output.name.c_str());
            }

            auto ort_outputs = _session->Run(Ort::RunOptions{nullptr},
                run_input_names.data(), ort_inputs.data(), ort_inputs.size(),
                run_output_names.data(), run_output_names.size());

            for (size_t i = 0; i < ort_outputs.size(); ++i) {
                auto it = outputs.find(_outputs[i].name);
                if (it != outputs.end() && ort_outputs[i].IsTensor()) {
                    float * ort_data = ort_outputs[i].GetTensorMutableData<float>();
                    size_t count = ort_outputs[i].GetTensorTypeAndShapeInfo().GetElementCount();
                    std::memcpy(it->second, ort_data, count * sizeof(float));
                }
            }

            return true;

        } catch (const Ort::Exception & e) {
            fprintf(stderr, "llama-onnx: Run failed: %s\n", e.what());
            return false;
        }
    }
};

// ---- Public API ----

llama::onnx_model::onnx_model()
    : pimpl(std::make_unique<impl>()) {
}

llama::onnx_model::~onnx_model() = default;

bool llama::onnx_model::load(const std::string & model_path, onnx_provider provider) {
    return pimpl->load_model(model_path, provider);
}

bool llama::onnx_model::is_loaded() const {
    return pimpl->_loaded;
}

const std::string & llama::onnx_model::model_path() const {
    return pimpl->_model_path;
}

std::vector<llama::onnx_tensor_info> llama::onnx_model::input_info() const {
    return pimpl->_inputs;
}

std::vector<llama::onnx_tensor_info> llama::onnx_model::output_info() const {
    return pimpl->_outputs;
}

bool llama::onnx_model::run(
    const std::unordered_map<std::string, float *> & inputs,
    const std::unordered_map<std::string, float *> & outputs) {
    return pimpl->run_model(inputs, outputs);
}

size_t llama::onnx_model::num_inputs() const {
    return pimpl->_inputs.size();
}

size_t llama::onnx_model::num_outputs() const {
    return pimpl->_outputs.size();
}

bool llama::onnx_available() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "llama-onnx-check");
        return true;
    } catch (...) {
        return false;
    }
}

std::string llama::onnx_version() {
    return std::to_string(ORT_API_VERSION);
}

std::vector<llama::onnx_provider> llama::onnx_available_providers() {
    std::vector<onnx_provider> providers;
    providers.push_back(onnx_provider::CPU);

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "llama-onnx-providers");
        Ort::SessionOptions options;

        try {
            OrtDnnlProviderOptions dnnl_opts{};
            options.AppendExecutionProvider_Dnnl(dnnl_opts);
            providers.push_back(onnx_provider::DNNL);
        } catch (const Ort::Exception &) {}

        try {
            OrtROCMProviderOptions rocm_opts{};
            rocm_opts.device_id = 0;
            options.AppendExecutionProvider_ROCM(rocm_opts);
            providers.push_back(onnx_provider::ROCM);
        } catch (const Ort::Exception &) {}

        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            options.AppendExecutionProvider_CUDA(cuda_opts);
            providers.push_back(onnx_provider::CUDA);
        } catch (const Ort::Exception &) {}

    } catch (const Ort::Exception &) {}

    return providers;
}

const char * llama::onnx_provider_name(onnx_provider provider) {
    switch (provider) {
        case onnx_provider::CPU:  return "CPU";
        case onnx_provider::DNNL: return "DNNL";
        case onnx_provider::ROCM: return "ROCM";
        case onnx_provider::CUDA: return "CUDA";
        case onnx_provider::AUTO: return "AUTO";
        default:                  return "UNKNOWN";
    }
}
