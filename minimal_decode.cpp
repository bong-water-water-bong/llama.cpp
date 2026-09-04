// minimal_decode.cpp — drive llama_decode() directly, bypassing the CLI's
// common-library diagnostic (common_context_can_seq_rm) and warmup, to isolate
// the recurrent-state (cca_state) carry-over across decodes.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file("/home/bcloud/zaya-f32.gguf", mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // replicate the common_context_can_seq_rm diagnostic (decode 2 dummy + clear)
    if (getenv("MIN_REPLICATE_DIAG")) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_token d[2] = {0, 0};
        llama_decode(ctx, llama_batch_get_one(d, 2));
        llama_memory_seq_rm(llama_get_memory(ctx), 0, 1, -1);
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_synchronize(ctx);
        fprintf(stderr, "[min] diagnostic replicated\n");
    }

    const char * prompt = "The capital of France is";
    std::vector<llama_token> toks(512);
    int n;

    if (getenv("MIN_CHAT")) {
        const char * tmpl = llama_model_chat_template(model, nullptr);
        fprintf(stderr, "[min] chat template: %s\n", tmpl ? (tmpl[0] ? tmpl : "(empty)") : "(null)");
        llama_chat_message msg;
        msg.role = "user";
        msg.content = prompt;
        char buf[4096];
        int blen = llama_chat_apply_template(tmpl, &msg, 1, true, buf, sizeof(buf));
        if (blen < 0) { fprintf(stderr, "template failed %d\n", blen); return 1; }
        buf[blen] = 0;
        fprintf(stderr, "[min] chat prompt: %s\n", buf);
        n = llama_tokenize(vocab, buf, blen, toks.data(), 512, true, true);
    } else {
        n = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), 512, true, true);
    }
    fprintf(stderr, "[min] tokenize n=%d\n", n);
    if (n <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    fprintf(stderr, "[min] prompt has %d tokens:", n);
    for (int i = 0; i < n; i++) fprintf(stderr, " %d", toks[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[min] === decode prompt (%d tokens) ===\n", n);
    int r = llama_decode(ctx, llama_batch_get_one(toks.data(), n));
    fprintf(stderr, "[min] decode prompt ret=%d\n", r);

    // read logits RIGHT AFTER the prompt decode (predict next token = oracle 9079)
    {
        const float * logits = llama_get_logits(ctx);
        int n_vocab = llama_vocab_n_tokens(vocab);
        int argmax = 0;
        for (int i = 1; i < n_vocab; i++) { if (logits[i] > logits[argmax]) argmax = i; }
        fprintf(stderr, "[min] AFTER-PROMPT argmax=%d val=%g logits[9079]=%g\n", argmax, logits[argmax], logits[9079]);
    }

    llama_token gen = 0;
    fprintf(stderr, "[min] === decode 1 generated token ===\n");
    r = llama_decode(ctx, llama_batch_get_one(&gen, 1));
    fprintf(stderr, "[min] decode gen ret=%d\n", r);

    const float * logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int argmax = 0;
    for (int i = 1; i < n_vocab; i++) { if (logits[i] > logits[argmax]) argmax = i; }
    fprintf(stderr, "[min] n_vocab=%d argmax=%d val=%g logits[9079]=%g\n", n_vocab, argmax, logits[argmax], logits[9079]);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
