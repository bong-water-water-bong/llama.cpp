import numpy as np, glob, os

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

oracle_dir = "/home/bcloud/zaya-decode/ffn_oracle"
dump_dir = "/tmp/prg_dump"
nt, N = 6, 4096

# r03_003 = GET_ROWS ffn_moe_weights = 6 i32 expert ids in assignment order (per round-64: block0 = 3,5,12,7,2,4)
print("block | experts(assign order) | HRX-vs-CPU per-token mad (t0..t5) | correct-tokens")
for b in range(40):
    oidx = 3 + 8 * b
    g = glob.glob(f"{oracle_dir}/r03_{oidx:03d}_*GET_ROWS_ffn_moe_weights-{b}.bin")
    hg = [f for f in glob.glob(f"{dump_dir}/*_001_ffn_moe_gate_up-{b}.bin") if os.path.getsize(f) == nt * N * 4]
    if not g or not hg:
        print(f"{b:>3} | MISSING"); continue
    ids = np.fromfile(g[0], dtype=np.int32)
    O = np.fromfile(hg[0], dtype=np.float32).reshape(nt, N)
    H = np.fromfile([f for f in hg if os.path.getsize(f)==nt*N*4][0], dtype=np.float32).reshape(nt, N) if False else O
    # reload properly
    O = np.fromfile([f for f in glob.glob(f"{oracle_dir}/r03_{4+8*b:03d}_*ffn_moe_gate_up-{b}.bin")][0], dtype=np.float32).reshape(nt, N)
    H = np.fromfile(hg[0], dtype=np.float32).reshape(nt, N)
    mads = [float(np.abs(H[t] - O[t]).mean()) for t in range(nt)]
    ok = [t for t in range(nt) if mads[t] < 0.05]
    print(f"{b:>3} | {list(ids)} | {' '.join(f'{m:7.4f}' for m in mads)} | {ok}")
