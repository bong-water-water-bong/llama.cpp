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

def load_expert(t, expert, n_tr, n_tc):
    # per-expert contiguous planes: ne1 = tiles/expert, ne2 = experts
    tpe = t.shape[1]
    base = t.data_offset + expert * tpe * 5120
    W = np.zeros((n_tr * 32, n_tc * 256), dtype=np.float32)
    for tr in range(n_tr):
        for tc in range(n_tc):
            tile_idx = tr * n_tc + tc
            off = base + tile_idx * 5120
            W[tr * 32:(tr + 1) * 32, tc * 256:(tc + 1) * 256] = dequant_tile(mm[off:off + 5120])
    return W

# --- down mm validation (self-contained): r03_005 (swiglu oracle) -> r03_006 (down oracle)
t_dn = tensor("blk.0.ffn_moe_down_exps.weight")
print("down tensor", t_dn.shape.tolist(), "type", int(t_dn.tensor_type))
K = 2048
N_dn = 2048
n_tc = K // 256
n_tr = N_dn // 32
e_of_tok = {0: 3, 1: 5, 2: 12, 3: 7, 4: 2, 5: 4}

sw = np.fromfile(glob.glob("/home/bcloud/zaya-decode/ffn_oracle/r03_005_*ffn_moe_swiglu-0.bin")[0], dtype="<f4").reshape(6, 2048)
dn = np.fromfile(glob.glob("/home/bcloud/zaya-decode/ffn_oracle/r03_006_*ffn_moe_down-0.bin")[0], dtype="<f4").reshape(6, 2048)

for t in [0]:
    e = e_of_tok[t]
    W = load_expert(t_dn, e, n_tr, n_tc)
    out = W @ sw[t]
    mad = float(np.abs(out - dn[t]).mean())
    print(f"down validate t{t} e{e}: mad={mad:.6f} max={float(np.abs(out-dn[t]).max()):.6f} outrms={float(np.sqrt((out*out).mean())):.4f} oracle rms={float(np.sqrt((dn[t]**2).mean())):.4f}")
