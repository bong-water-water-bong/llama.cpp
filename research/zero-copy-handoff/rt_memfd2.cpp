// rt_memfd2.cpp - zero-copy proof, minimal: import a llama_state blob via
// llama_state_set_data (in-memory, from a shared-memory mmap) and compare
// continuation tokens vs the file-based llama_state_load_file baseline.
// Usage: rt_memfd2 <model> <session.bin> <n_cont>
//   prints "F <tok>" lines (file baseline) then "M <tok>" lines (memfd/set_data).
// Identical F/M streams => the file API is removable (shared memory suffices).
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

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

static void continue_from(llama_context* ctx, const llama_vocab* vocab, int first_tok, int n_cont, const char* tag) {
    int t = first_tok;
    for (int i = 0; i < n_cont; i++) {
        printf("%s %d\n", tag, t);
        fflush(stdout);
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) { fprintf(stderr, "decode fail %d\n", i); return; }
        t = argmax(ctx, vocab);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: rt_memfd2 <model> <session.bin> <n_cont>\n"); return 1; }
    const char* model_path = argv[1];
    const char* sess_path = argv[2];
    const int n_cont = atoi(argv[3]);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("RT_NGL") ? atoi(getenv("RT_NGL")) : 0;
    if (const char* dev = getenv("RT_DEV")) {
        ggml_backend_dev_t d = ggml_backend_dev_by_name(dev);
        if (d) { static ggml_backend_dev_t dl[2]; dl[0]=d; dl[1]=nullptr; mp.devices = dl; }
    }
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load FAILED\n"); return 2; }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // read blob into shared memory (mmap MAP_SHARED of a memfd = zero-copy carrier)
    int fd = open(sess_path, O_RDONLY);
    if (fd < 0) { perror("open"); return 3; }
    off_t fsz = lseek(fd, 0, SEEK_END); lseek(fd, 0, SEEK_SET);
    int mfd = memfd_create("state", 0);
    if (mfd < 0) { perror("memfd"); return 4; }
    ftruncate(mfd, fsz);
    void* shm = mmap(nullptr, (size_t)fsz, PROT_READ|PROT_WRITE, MAP_SHARED, mfd, 0);
    if (shm == MAP_FAILED) { perror("mmap"); return 5; }
    ssize_t rd = read(fd, shm, (size_t)fsz); (void)rd;
    close(fd);
    fprintf(stderr, "[i] blob %lld bytes mapped shared (memfd)\n", (long long)fsz);

    // ---- baseline: FILE import ----
    {
        llama_context* ctx = make_ctx(model);
        std::vector<llama_token> toks(32768);
        size_t n = 0;
        if (!llama_state_load_file(ctx, sess_path, toks.data(), toks.size(), &n)) { fprintf(stderr, "file load FAILED\n"); return 6; }
        fprintf(stderr, "[F] file import: %zu stored tokens; resume=%d\n", n, toks[n-1]);
        continue_from(ctx, vocab, toks[n-1], n_cont, "F");
        llama_free(ctx);
    }

    // ---- zero-copy: in-memory import from the SHARED pages ----
    {
        // session file = header(12B) + token array (4*n) + RAW STATE. set_data wants
        // only the raw state region; the file API strips header+tokens internally.
        const uint8_t* h = (const uint8_t*)shm;
        uint32_t ntok = 0; memcpy(&ntok, h + 8, 4);
        int32_t lasttok = 0; memcpy(&lasttok, h + 12 + 4*(size_t)(ntok-1), 4);
        const uint8_t* state = h + 12 + 4*(size_t)ntok;
        size_t state_sz = (size_t)fsz - 12 - 4*(size_t)ntok;
        fprintf(stderr, "[M] header: %u tokens; state region %zu bytes @ +%zu\n", ntok, state_sz, 12+4*(size_t)ntok);
        llama_context* ctx = make_ctx(model);
        size_t consumed = llama_state_set_data(ctx, state, state_sz);
        fprintf(stderr, "[M] set_data (shared mem) consumed %zu bytes (expect %zu)\n", consumed, state_sz);
        continue_from(ctx, vocab, lasttok, n_cont, "M");
        llama_free(ctx);
    }

    llama_free_model(model);
    llama_backend_free();
    return 0;
}
