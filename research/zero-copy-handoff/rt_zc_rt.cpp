// rt_zc_rt.cpp — zero-copy round-trip proof (single process, q35 CPU build):
//   1. prefill a random token prompt once
//   2. ROUND TRIP F (today): llama_state_save_file -> fresh ctx -> load_file -> continue
//   3. ROUND TRIP M (zero-copy): llama_state_get_data -> memfd (shared mem, no file)
//      -> fresh ctx -> llama_state_set_data from the shared pages -> continue
// Streams F* and M* must be token-identical. M performs zero file I/O.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/syscall.h>

static int argmax(llama_context* ctx, const llama_vocab* vocab) {
    const float* lg = llama_get_logits(ctx);
    int nv = llama_vocab_n_tokens(vocab);
    int best = 0;
    for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
    return best;
}

static llama_context* make_ctx(llama_model* model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = getenv("RT_NC") ? atoi(getenv("RT_NC")) : 4096;
    cp.n_ubatch = 512; cp.n_batch = 2048; cp.n_threads = 8;
    llama_context* ctx = llama_init_from_model(model, cp);
    llama_set_n_threads(ctx, cp.n_threads, cp.n_threads);
    return ctx;
}

static std::vector<llama_token> g_last_input; // last prefill input token (filler)

static void cont_and_print(llama_context* ctx, const llama_vocab* vocab, int n_cont, const char* tag) {
    // mirror rt_session run-mode semantics: the ctx after load/set is positioned at
    // token index P (KV holds 0..P-1). Decoding the last prefill input token again at
    // auto-pos P yields logits for the true next token; print-then-decode loop.
    int t = g_last_input.back();
    for (int i = 0; i < n_cont; i++) {
        printf("%s %d\n", tag, t);
        fflush(stdout);
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) { fprintf(stderr, "%s decode fail %d\n", tag, i); return; }
        t = argmax(ctx, vocab);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: rt_zc_rt <model> <n_prompt> <n_cont>\n"); return 1; }
    const char* model_path = argv[1];
    const int np = atoi(argv[2]);
    const int n_cont = atoi(argv[3]);
    const char* tmpfile = "/tmp/zc_rt_session.bin";

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("RT_NGL") ? atoi(getenv("RT_NGL")) : 0; // CPU for hermetic proof
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load FAILED\n"); return 2; }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // ---- prefill once ----
    llama_context* ctx0 = make_ctx(model);
    std::vector<llama_token> ptoks(np);
    unsigned s = 12345;
    for (int i = 0; i < np; i++) { s = s*1103515245+12345; ptoks[i] = (llama_token)((s>>16) % 2000 + 10); }
    for (int off = 0; off < np; off += 512) {
        int nch = (np-off) < 512 ? (np-off) : 512;
        if (llama_decode(ctx0, llama_batch_get_one(ptoks.data()+off, nch)) != 0) { fprintf(stderr, "prefill fail\n"); return 3; }
    }
    g_last_input = ptoks;
    fprintf(stderr, "[i] prefilled %d tokens\n", np);
    size_t state_sz = llama_state_get_size(ctx0);
    fprintf(stderr, "[i] state size %zu\n", state_sz);

    // ---- ROUND TRIP F: file ----
    {
        if (!llama_state_save_file(ctx0, tmpfile, ptoks.data(), ptoks.size())) { fprintf(stderr, "F save fail\n"); return 4; }
        llama_context* ctx = make_ctx(model);
        std::vector<llama_token> toks(np + 16);
        size_t n = 0;
        if (!llama_state_load_file(ctx, tmpfile, toks.data(), toks.size(), &n)) { fprintf(stderr, "F load fail\n"); return 5; }
        fprintf(stderr, "[F] file round-trip: %zu tokens\n", n);
        cont_and_print(ctx, vocab, n_cont, "F");
        llama_free(ctx);
        remove(tmpfile);
    }

    // ---- ROUND TRIP M: shared memory (memfd), zero file I/O ----
    {
        int mfd = (int)syscall(319, "zc-state", 0); // memfd_create (x86-64)
        if (mfd < 0) { perror("memfd_create"); return 6; }
        ftruncate(mfd, (off_t)state_sz);
        void* shm = mmap(nullptr, state_sz, PROT_READ|PROT_WRITE, MAP_SHARED, mfd, 0);
        if (shm == MAP_FAILED) { perror("mmap"); return 7; }
        size_t got = llama_state_get_data(ctx0, (uint8_t*)shm, state_sz);
        fprintf(stderr, "[M] state -> shared memfd: %zu B (no file)\n", got);
        llama_context* ctx = make_ctx(model);
        size_t used = llama_state_set_data(ctx, (const uint8_t*)shm, got);
        fprintf(stderr, "[M] state <- shared memfd: %zu B\n", used);
        cont_and_print(ctx, vocab, n_cont, "M");
        llama_free(ctx);
        munmap(shm, state_sz);
        close(mfd);
    }

    llama_free(ctx0);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
