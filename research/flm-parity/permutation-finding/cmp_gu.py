import numpy as np, glob, os

def load(p):
    return np.fromfile(p, dtype=np.float32)

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

oracle_dir = "/home/bcloud/zaya-decode/ffn_oracle"
dump_dir = "/tmp/prg_dump"
nt, N = 6, 4096

print(f"{'blk':>3} | {'t0':>7}{'t1':>7}{'t2':>7}{'t3':>7}{'t4':>7}{'t5':>7} | corr HRX(t)vsCPU(t): t0..t5 | best-match(u,corr) per HRX token t0..t5")
for b in range(40):
    oidx = 4 + 8 * b
    oglob = glob.glob(f"{oracle_dir}/r03_{oidx:03d}_*ffn_moe_gate_up-{b}.bin")
    hglob = glob.glob(f"{dump_dir}/*_001_ffn_moe_gate_up-{b}.bin")
    if not oglob or not hglob:
        print(f"{b:>3} | MISSING oracle={bool(oglob)} hrx={bool(hglob)}"); continue
    cands=[]
    for f in hglob:
        sz=os.path.getsize(f)
        if sz==nt*N*4: cands.append(f)
    if not cands:
        print(f"{b:>3} | no 98304B prefill dump"); continue
    O = load(oglob[0]); H = load(cands[0])
    if O.size != nt * N or H.size != nt * N:
        print(f"{b:>3} | SIZE? oracle={O.size} hrx={H.size}"); continue
    O = O.reshape(nt, N); H = H.reshape(nt, N)
    mads = [float(np.abs(H[t] - O[t]).mean()) for t in range(nt)]
    cors = [corr(H[t], O[t]) for t in range(nt)]
    best = []
    for t in range(nt):
        cs = [corr(H[t], O[u]) for u in range(nt)]
        u = int(np.argmax(cs)); best.append((u, cs[u]))
    print(f"{b:>3} | " + " ".join(f"{m:7.4f}" for m in mads) + " | "
          + " ".join(f"{c:5.2f}" for c in cors) + " | "
          + " ".join(f"({u},{c:.2f})" for u, c in best))
