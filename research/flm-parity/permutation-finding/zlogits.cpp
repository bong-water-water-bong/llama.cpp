// zlogits: print full top-12 prefill logits + the 9079/563 scores
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/home/bcloud/zaya-q4nx-c43.gguf";
    const int    ngl        = argc > 2 ? atoi(argv[2]) : 0;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load FAILED\n"); return 2; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "The capital of France is";
    std::vector<llama_token> toks(512);
    int n = llama_tokenize(vocab, prompt, strlen(prompt), toks.data(), 512, true, true);
    toks.resize(n);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n)) != 0) { fprintf(stderr, "decode fail\n"); return 3; }
    const float * logits = llama_get_logits(ctx);
    int nv = llama_vocab_n_tokens(vocab);
    int top[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
    for (int i = 1; i < nv; i++)
        for (int k = 0; k < 12; k++) if (logits[i] > logits[top[k]]) { for (int j = 11; j > k; j--) top[j] = top[j-1]; top[k] = i; break; }
    for (int k = 0; k < 12; k++) fprintf(stderr, "top%d=%d(%.4f)\n", k, top[k], logits[top[k]]);
    fprintf(stderr, "score9079=%.4f score563=%.4f\n", logits[9079], logits[563]);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
