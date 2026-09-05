#include "llama.h"
#include <cstdio>
#include <vector>
#include <cstring>
#include <string>
int main() {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file("/home/bcloud/zaya-f32.gguf", mp);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "The capital of France is";
    std::vector<llama_token> toks(512);
    int n = llama_tokenize(vocab, prompt, strlen(prompt), toks.data(), 512, true, true);
    toks.resize(n);
    llama_decode(ctx, llama_batch_get_one(toks.data(), n));
    // greedy decode 8 tokens
    fprintf(stderr, "greedy tokens:");
    std::vector<llama_token> out = toks;
    for (int step = 0; step < 8; step++) {
        const float * logits = llama_get_logits(ctx);
        int nv = llama_vocab_n_tokens(vocab);
        int best = 0; for (int i=1;i<nv;i++) if (logits[i]>logits[best]) best=i;
        fprintf(stderr, " %d", best);
        llama_token t = best;
        out.push_back(t);
        llama_decode(ctx, llama_batch_get_one(&t, 1));
    }
    fprintf(stderr, "\n");
    // decode text
    char buf[512]; int blen = llama_token_to_piece(vocab, out[0], buf, 512, 0, true);
    // simpler: print pieces
    std::string s;
    for (auto t : out) { char b[64]; int l = llama_token_to_piece(vocab, t, b, 64, 0, true); if (l>0) s += std::string(b, l); }
    fprintf(stderr, "text: %s\n", s.c_str());
    llama_free(ctx); llama_free_model(model); llama_backend_free();
    return 0;
}
