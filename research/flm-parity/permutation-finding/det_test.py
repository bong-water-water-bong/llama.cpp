import numpy as np, glob, os, sys, time

# wait for run2 log
while not os.path.exists("/tmp/cap-gu2.log"):
    time.sleep(10)
while True:
    with open("/tmp/cap-gu2.log") as f:
        if "EXIT=" in f.read():
            break
    time.sleep(15)
print("run2 done")

def load(d, b):
    hg = [f for f in glob.glob(f"{d}/*_001_ffn_moe_gate_up-{b}.bin") if os.path.getsize(f) == 98304]
    return np.fromfile(hg[0], dtype=np.float32).reshape(6, 4096) if hg else None

for b in [0, 3, 6, 19, 20, 35, 38, 39]:
    A = load("/tmp/prg_dump_run1", b)
    B = load("/tmp/prg_dump", b)
    if A is None or B is None:
        print(b, "missing"); continue
    for t in [0]:
        a = A[t]; bb = B[t]
        mad = float(np.abs(a - bb).mean())
        c = float(np.corrcoef(a, bb)[0, 1])
        verdict = "IDENTICAL" if mad < 1e-4 else "DIFFERENT"
        print(f"b{b} t{t}: run1-vs-run2 mad={mad:.5f} corr={c:.4f} -> {verdict}")
