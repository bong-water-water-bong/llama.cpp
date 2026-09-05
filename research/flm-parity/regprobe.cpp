#include "llama.h"
#include "ggml.h"
#include <cstdio>
int main(){
  printf("reg_count before init: %zu\n", ggml_backend_reg_count());
  llama_backend_init();
  printf("reg_count after init: %zu\n", ggml_backend_reg_count());
  for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
    ggml_backend_reg_t r = ggml_backend_reg_get(i);
    printf("reg %zu: %s (%zu devices)\n", i, ggml_backend_reg_name(r), ggml_backend_reg_dev_count(r));
    for (size_t d = 0; d < ggml_backend_reg_dev_count(r); ++d) {
      ggml_backend_dev_t dev = ggml_backend_reg_dev_get(r, d);
      printf("   dev %zu: %s type=%d desc=%s\n", d, ggml_backend_dev_name(dev), (int)ggml_backend_dev_type(dev), ggml_backend_dev_description(dev));
    }
  }
  printf("supports_gpu_offload: %d\n", (int)llama_supports_gpu_offload());
  return 0;
}
