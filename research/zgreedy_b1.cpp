// zgreedy variant: decode the prompt in 1-token batches (decode-shaped prefill)
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/home/bcloud/zaya-q4nx-c43.gguf";
    const int    ngl        = argc > 2 ? atoi(argv[2]) : 0;
    const char * prompt     = argc > 3 ? argv[3] : "The capital of France is";
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load FAILED\n"); return 2; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 1; cp.n_ubatch = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(512);
    int n = llama_tokenize(vocab, prompt, strlen(prompt), toks.data(), 512, true, true);
    toks.resize(n);
    fprintf(stderr, "prompt tokens(%zu):", toks.size());
    for (auto t : toks) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");
    // decode prompt one token at a time (single-token prefill batches)
    for (size_t i = 0; i < toks.size(); i++) {
        llama_token t = toks[i];
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) { fprintf(stderr, "prefill decode fail @%zu\n", i); return 3; }
    }
    const int n_steps = 8;
    for (int step = 0; step < n_steps; step++) {
        const float * logits = llama_get_logits(ctx);
        int nv = llama_vocab_n_tokens(vocab);
        int best = 0; for (int i = 1; i < nv; i++) if (logits[i] > logits[best]) best = i;
        if (step == 0) {
            int top[5] = {0,0,0,0,0};
            for (int i = 1; i < nv; i++) {
                for (int k = 0; k < 5; k++) if (logits[i] > logits[top[k]]) { for (int j = 4; j > k; j--) top[j] = top[j-1]; top[k] = i; break; }
            }
            fprintf(stderr, "top5 step0:");
            for (int k = 0; k < 5; k++) fprintf(stderr, " %d(%.3f)", top[k], logits[top[k]]);
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "tok%d=%d\n", step, best);
        llama_token t = best;
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) { fprintf(stderr, "decode fail step %d\n", step); break; }
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
