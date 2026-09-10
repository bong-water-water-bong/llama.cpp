#pragma once
#include "../dispatch-registry.h"
namespace ggml::hrx {
void register_concat_dispatches(DispatchRegistryBuilder & builder);
}  // namespace ggml::hrx
