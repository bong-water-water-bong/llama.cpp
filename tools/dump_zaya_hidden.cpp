// dump_zaya_hidden.cpp — dump hidden states from Zaya GGUF for comparison with HF
#include "common.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    
    llama_model* model = llama_load_model_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    
    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    
    // Tokenize "hello" (same as HF: [2, 23391, 1902])
    std::vector<llama_token> tokens = {2, 23391, 1902};
    
    // Run batch
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); return 1; }
    
    // Get logits for last token
    float* logits = llama_get_logits(ctx);
    int n_vocab = llama_n_vocab(model);
    
    // Save logits
    FILE* f = fopen("/tmp/zaya_gguf/logits.bin", "wb");
    fwrite(logits, sizeof(float), n_vocab, f);
    fclose(f);
    
    // Get hidden states via logits (last token only)
    // The hidden states are NOT directly exposed in llama.cpp API.
    // We can only get the final logits.
    // To get per-layer hidden states, we need to modify the model code.
    
    fprintf(stderr, "Logits saved to /tmp/zaya_gguf/logits.bin\n");
    
    // Top 5 tokens
    std::vector<std::pair<float, int>> scored;
    for (int i = 0; i < n_vocab; i++) scored.push_back({logits[i], i});
    std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
        [](auto& a, auto& b) { return a.first > b.first; });
    
    fprintf(stderr, "Top 5: ");
    for (int i = 0; i < 5; i++)
        fprintf(stderr, "%d (%.2f)%s", scored[i].second, scored[i].first, i < 4 ? ", " : "\n");
    
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
