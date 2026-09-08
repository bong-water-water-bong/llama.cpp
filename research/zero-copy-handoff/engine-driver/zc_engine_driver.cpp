// zc_engine_driver.cpp — drives the REAL engine Inprocess code (not a harness
// reimplementation): constructs hrx::Inprocess, loads the model, imports a
// shared-memory state via Inprocess::load_session_mem(fd) (committed 0a54070c),
// and decodes. Links the engine source files directly so the tested code IS
// the shipped engine code.
//
// Usage: zc_engine_driver <bundle_libllama.so> <model.gguf> <state_fd> <n_cont>
//   state_fd: a memfd/session-v9 state blob fd produced by any prefill lane
//   (vendored HIP llama_state_get_data, or the fork rt harness).
#include "hrx_inprocess.h"
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fcntl.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: zc_engine_driver <model.gguf> <state_fd> <n_cont>\n");
        return 2;
    }
    const char* model = argv[1];
    // state source: a session-file path or a pre-opened fd number (42 for a
    // memfd held by the caller). We load the bytes into a fresh memfd so the
    // REAL engine seam (load_session_mem on a shared-memory fd) is exercised.
    const char* state_path = argv[2];
    int n_cont = atoi(argv[3]);
    int fd = -1;
    if (state_path[0] >= '0' && state_path[0] <= '9') {
        fd = atoi(state_path);
    } else {
        FILE* f = fopen(state_path, "rb");
        if (!f) { perror("open state"); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        fd = (int)syscall(319, "eng-state", 0);
        ftruncate(fd, sz);
        void* buf = malloc((size_t)sz);
        fread(buf, 1, (size_t)sz, f);
        fclose(f);
        write(fd, buf, (size_t)sz);
        lseek(fd, 0, SEEK_SET);
        free(buf);
        fprintf(stderr, "[driver] loaded state file (%ld B) into memfd %d\n", sz, fd);
    }

    hrx::Inprocess hrx;
    if (!hrx.init()) { fprintf(stderr, "[driver] Inprocess::init FAILED\n"); return 1; }
    fprintf(stderr, "[driver] Inprocess init OK\n");
    if (!hrx.load_model(model, -1, 4096)) {
        fprintf(stderr, "[driver] load_model FAILED\n");
        return 1;
    }
    fprintf(stderr, "[driver] model loaded, device=%s\n",
            hrx.has_hrx_device() ? hrx.hrx_device_name() : "(none)");

    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    fprintf(stderr, "[driver] state fd %d size %lld\n", fd, (long long)st.st_size);

    // THE REAL ENGINE SEAM (0a54070c): import from shared memory, no file.
    long imported = hrx.load_session_mem(fd);
    if (imported < 0) { fprintf(stderr, "[driver] load_session_mem FAILED\n"); return 1; }
    fprintf(stderr, "[driver] load_session_mem imported %ld tokens\n", imported);

    // Decode continuation via the engine generate() path.
    fprintf(stderr, "[driver] decoding %d tokens...\n", n_cont);
    int tok = 0;
    for (int i = 0; i < n_cont; i++) {
        int next = hrx.generate(tok);
        if (next < 0) { fprintf(stderr, "\n[driver] generate failed at %d\n", i); break; }
        printf("%d%s", next, i + 1 < n_cont ? " " : "\n");
        fflush(stdout);
        tok = next;
    }
    fprintf(stderr, "[driver] DONE\n");
    return 0;
}
