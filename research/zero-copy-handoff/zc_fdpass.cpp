// zc_fdpass.cpp — cross-process zero-copy state handoff via SCM_RIGHTS fd passing.
//   producer: prefill N tokens (default device), llama_state_get_data -> memfd,
//             send fd over a unix socket (SCM_RIGHTS), then decode+print "P .." tokens
//   consumer: recv fd, mmap, llama_state_set_data (raw state region), decode+print "C .."
// Compare C stream vs P stream == token-identical (state handoff lossless, no file).
// Usage: zc_fdpass producer <model> <n_prompt> <n_cont>   (prints "FDPASS <fd>")
//        zc_fdpass consumer <model> <fd> <n_cont>
// Simpler self-contained mode: zc_fdpass both <model> <n_prompt> <n_cont> uses a
// socketpair + fork so one binary demonstrates the real cross-process fd move.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>

static int argmax(llama_context* ctx, const llama_vocab* vocab) {
    const float* lg = llama_get_logits(ctx);
    int nv = llama_vocab_n_tokens(vocab);
    int best = 0;
    for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
    return best;
}

static int send_fd(int sock, int fd) {
    char buf[1] = {'F'};
    struct iovec iov = {buf, 1};
    char cmsgbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr msg; memset(&msg, 0, sizeof msg);
    msg.msg_iov = &iov; msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf; msg.msg_controllen = sizeof cmsgbuf;
    struct cmsghdr* c = CMSG_FIRSTHDR(&msg);
    c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), &fd, sizeof(int));
    return sendmsg(sock, &msg, 0);
}

static int recv_fd(int sock) {
    char buf[1];
    struct iovec iov = {buf, 1};
    char cmsgbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr msg; memset(&msg, 0, sizeof msg);
    msg.msg_iov = &iov; msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf; msg.msg_controllen = sizeof cmsgbuf;
    if (recvmsg(sock, &msg, 0) < 0) { perror("recvmsg"); return -1; }
    struct cmsghdr* c = CMSG_FIRSTHDR(&msg);
    int fd = -1;
    if (c && c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
        memcpy(&fd, CMSG_DATA(c), sizeof(int));
    }
    return fd;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: zc_fdpass both <model> <n_prompt> <n_cont> | producer|consumer <model> <arg> <n_cont>\n"); return 1; }
    std::string role = argv[1];
    const char* model_path = argv[2];
    const int n_cont = argc > 4 ? atoi(argv[4]) : 13;

    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { perror("socketpair"); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("RT_NGL") ? atoi(getenv("RT_NGL")) : 0;
    if (const char* dev = getenv("RT_DEV")) {
        ggml_backend_dev_t d = ggml_backend_dev_by_name(dev);
        if (d) { static ggml_backend_dev_t dl[2]; dl[0]=d; dl[1]=nullptr; mp.devices = dl; }
    }
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load FAILED\n"); return 2; }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        // ── PRODUCER: prefill, export state to memfd, send fd ──
        close(sv[0]);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096; cp.n_ubatch = 512; cp.n_batch = 2048; cp.n_threads = 8;
        llama_context* ctx = llama_init_from_model(model, cp);
        llama_set_n_threads(ctx, 8, 8);
        const llama_vocab* vocab = llama_model_get_vocab(model);
        int np = atoi(argv[3]);
        std::vector<llama_token> pt(np);
        unsigned s = 12345;
        for (int i = 0; i < np; i++) { s = s*1103515245+12345; pt[i] = (llama_token)((s>>16) % 2000 + 10); }
        for (int off = 0; off < np; off += 512) {
            int nch = (np-off) < 512 ? (np-off) : 512;
            if (llama_decode(ctx, llama_batch_get_one(pt.data()+off, nch)) != 0) return 3;
        }
        size_t sz = llama_state_get_size(ctx);
        int mfd = (int)syscall(319, "zc-state", 0); // memfd_create
        ftruncate(mfd, (off_t)sz);
        void* shm = mmap(nullptr, sz, PROT_READ|PROT_WRITE, MAP_SHARED, mfd, 0);
        size_t got = llama_state_get_data(ctx, (uint8_t*)shm, sz);
        fprintf(stderr, "[P] state %zu B in memfd; sending fd over socket\n", got);
        if (send_fd(sv[1], mfd) < 0) { perror("send_fd"); return 4; }
        // raw state has no token bookkeeping - send the resume token (last prefill
        // input) so the consumer can reproduce the continuation exactly
        int32_t resume = pt.back();
        if (write(sv[1], &resume, sizeof resume) != (ssize_t)sizeof resume) { perror("write tok"); return 4; }
        fprintf(stderr, "[P] resume token %d sent\n", resume);
        // producer continues decoding from its own ctx (reference stream)
        int t = pt.back();
        for (int i = 0; i < n_cont; i++) {
            printf("P %d\n", t); fflush(stdout);
            if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) break;
            t = argmax(ctx, vocab);
        }
        llama_free(ctx);
        _exit(0);
    } else {
        // ── CONSUMER: recv fd, import state from shared mem, decode ──
        close(sv[1]);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096; cp.n_ubatch = 512; cp.n_batch = 2048; cp.n_threads = 8;
        llama_context* ctx = llama_init_from_model(model, cp);
        llama_set_n_threads(ctx, 8, 8);
        const llama_vocab* vocab = llama_model_get_vocab(model);
        int fd = recv_fd(sv[0]);
        if (fd < 0) { fprintf(stderr, "recv fd failed\n"); return 5; }
        struct stat st; fstat(fd, &st);
        size_t fsz = (size_t)st.st_size;
        void* shm = mmap(nullptr, fsz, PROT_READ, MAP_SHARED, fd, 0);
        if (shm == MAP_FAILED) { perror("mmap"); return 6; }
        // session-file layout strip: header 12B + tokens; here producer sent RAW state
        // (get_data, no session header) so import directly.
        size_t used = llama_state_set_data(ctx, (const uint8_t*)shm, fsz);
        fprintf(stderr, "[C] imported %zu B state from received fd (set_data %zu)\n", fsz, used);
        // read the resume token the producer sent alongside the fd
        int32_t resume = 0;
        if (read(sv[0], &resume, sizeof resume) != (ssize_t)sizeof resume) { perror("read tok"); return 7; }
        fprintf(stderr, "[C] resume token %d\n", resume);
        llama_token t = (llama_token)resume;
        // print-first-then-decode (rt_session run semantics): first printed token is
        // the resume input, subsequent are true continuations - matches producer P.
        for (int i = 0; i < n_cont; i++) {
            printf("C %d\n", t); fflush(stdout);
            if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) break;
            t = argmax(ctx, vocab);
        }
        for (int i = 0; i < n_cont; i++) {
            printf("C %d\n", t); fflush(stdout);
            if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) break;
            t = argmax(ctx, vocab);
        }
        int status = 0; waitpid(pid, &status, 0);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 0;
    }
}
