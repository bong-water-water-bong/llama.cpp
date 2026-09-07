import sys
import numpy as np
import glob

sys.path.insert(0, "/tmp")
import gguf_q4nx
from gguf import GGUFReader

GGUF_PATH = "/home/bcloud/zaya-q4nx-c43.gguf"
r = GGUFReader(GGUF_PATH)
mm = np.memmap(GGUF_PATH, mode="r")

def tensor(name):
    for t in r.tensors:
        if t.name == name:
            return t
    return None

def bf16_bytes_to_f32(buf):
    u16 = np.frombuffer(buf, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32)

def dequant_tile(b):
    scales = bf16_bytes_to_f32(b[0:512])
    zeros = bf16_bytes_to_f32(b[512:1024])
    packed = b[1024:5120]
    out = np.zeros((32, 256), dtype=np.float32)
    for r in range(32):
        lane = r // 16
        lane_row = r % 16
        byte_idx = lane_row // 2
        nib = r % 2
        colbytes = packed[lane * 2048 + np.arange(256) * 8 + byte_idx].astype(np.uint8)
        q = np.where(nib == 0, colbytes & 0x0F, (colbytes >> 4) & 0x0F).astype(np.int16)
        val = np.where(q < 8, q, q - 16)
        scale = scales[r * 8 + np.arange(256) // 32]
        zp = zeros[r * 8 + np.arange(256) // 32]
        scale = np.where(np.isfinite(scale) & (np.abs(scale) <= 100.0), scale, 0.0)
        zp = np.where(np.isfinite(zp) & (np.abs(zp) <= 100.0), zp, 0.0)
        out[r] = val.astype(np.float32) * scale + zp
    return out

def load_expert(t, expert, n_tc, n_tr):
    tpe = t.shape[1]
    base = t.data_offset + expert * tpe * 5120
    W = np.zeros((n_tr * 32, n_tc * 256), dtype=np.float32)
    for tr in range(n_tr):
        for tc in range(n_tc):
            tile_idx = tr * n_tc + tc
            off = base + tile_idx * 5120
            W[tr * 32:(tr + 1) * 32, tc * 256:(tc + 1) * 256] = dequant_tile(mm[off:off + 5120])
    return W

def f16(x):
    return x.astype(np.float16).astype(np.float32)

t_gu = tensor("blk.0.ffn_gate_up_exps.weight")  # [8192, 1024, 16] -> K=2048 (n_tc=8), N=4096 (n_tr=128)
N_TC, N_TR = 8, 128
W = {}
for e in range(16):
    W[e] = f16(load_expert(t_gu, e, N_TC, N_TR))  # f16 weights as the kernel uses
print("weights ready")

e_of_tok = {0: 3, 1: 5, 2: 12, 3: 7, 4: 2, 5: 4}

# captures
raw = np.fromfile("/tmp/prg_dump/10866_001_common.moe_routing.row_debug.bin", dtype=np.uint8)
f = raw.view(np.float32)
x = np.stack([f[8192 + t * 2048:8192 + t * 2048 + 2048] for t in range(6)])  # f16-truncated, as kernel fetched
H = np.fromfile(glob.glob("/tmp/prg_dump_run1/10866_001_ffn_moe_gate_up-0.bin")[0], dtype="<f4").reshape(6, 4096)
print("x rms", [round(float(np.sqrt((v*v).mean())), 4) for v in x])

def mad(a, b):
    return float(np.abs(a - b).mean())

# 1) validate on the good row a0
out = W[e_of_tok[0]] @ x[0]
print(f"validate a0: mad(HRX)={mad(out, H[0]):.5f}  mad(oracle)={mad(out, np.fromfile(glob.glob('/home/bcloud/zaya-decode/ffn_oracle/r03_004_*ffn_moe_gate_up-0.bin')[0], dtype='<f4').reshape(6,4096)[0]):.5f}")

# 2) candidate matrix on the bad row a1
target = H[1]
print("candidate matrix for a1: best (expert, token) pairings by mad vs HRX row a1:")
res = []
for e in range(16):
    for u in range(6):
        c = W[e] @ x[u]
        res.append((mad(c, target), e, u))
res.sort()
print("top 8:", [(round(m, 4), e, u) for m, e, u in res[:8]])
print("worst 2:", [(round(m, 4), e, u) for m, e, u in res[-2:]])
# the "correct" pairing e5 x1 for reference
print("e5 x1:", round(mad(W[5] @ x[1], target), 4), " e3 x0:", round(mad(W[3] @ x[0], target), 4))

# 3) partial-K variants of the correct pairing (first m quant blocks of 8)
for m in range(1, 9):
    km = m * 256
    Wm = W[5].copy()
    Wm[:, km:] = 0.0
    c = Wm @ x[1]
    res.append((mad(c, target), f"k{km}"))
M = np.tile((np.arange(2048) < 0).astype(np.float32), (4096, 1))  # placeholder replaced below
pk = []
for m in range(1, 9):
    mask = np.zeros((4096, 2048), dtype=np.float32)
    mask[:, :m*256] = 1.0
    pk.append((mad((W[5] * mask) @ x[1], target), m))
print("partial-K (e5,x1) best:", [(round(m2, 4), m) for m2, m in sorted(pk)])
