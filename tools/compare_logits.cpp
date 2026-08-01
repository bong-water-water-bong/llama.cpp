// compare_logits.cpp — save logits from GGUF for comparison with HF
#include "common.h"
#include "llama.h"
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    
    llama_model* model = llama_load_model_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    llama_context* ctx = llama_new_context_with_model(model, cparams);
    
    // Same input as HF: [2, 23391, 1902] (<bos> hello)
    std::vector<llama_token> tokens = {2, 23391, 1902};
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
    
    float* logits = llama_get_logits(ctx);
    int n_vocab = llama_n_vocab(model);
    
    FILE* f = fopen("/tmp/zaya_gguf/logits.bin", "wb");
    fwrite(logits, sizeof(float), n_vocab, f);
    fclose(f);
    
    // Find top 10
    std::vector<std::pair<float, int>> scored;
    for (int i = 0; i < n_vocab; i++) scored.push_back({logits[i], i});
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    
    fprintf(stderr, "GGUF Top-10 logits:\n");
    for (int i = 0; i < 10; i++)
        fprintf(stderr, "  %d: %.2f\n", scored[i].second, scored[i].first);
    
    fprintf(stderr, "\nHF Top-10 logits:\n");
    fprintf(stderr, "  107: 24.62  (\\n)\n  108: 22.25  (\\n\\n)\n  827: 21.62  (\",)\n  236761: 21.50  (.)\n  563: 21.25  ( is)\n");
    
    // Compare top-1
    fprintf(stderr, "\nGGUF top-1=%d, HF top-1=107\n", scored[0].second);
    fprintf(stderr, "GGUF value=%.2f, HF value=24.62\n", scored[0].first);
    
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return scored[0].second == 107 ? 0 : 1;
}
