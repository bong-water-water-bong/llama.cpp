#!/usr/bin/env python3
"""zaya_q4nx_convert.py — quantize a zaya f32 GGUF to the c43 Q4NX GGUF format.

Reads the FT'd f32 GGUF (all tensors f32, layout verified identical to the
known-good zaya-q4nx-f32twin.gguf), and re-emits it with the 280 mm weights
(attn_q/k, cca_val_proj1/2, attn_output, ffn_gate_up_exps, ffn_down_exps per
layer) packed as GGML_TYPE_Q4NX (43, 5120B tiles) using the engine packer
(bit-exact vs c43: verified max-err 0.0 on blk.0.attn_q). All other tensors
stay f32.

Usage: zaya_q4nx_convert.py <src_f32.gguf> <ref_c43.gguf> <out.gguf>
  ref_c43 provides the exact per-tensor (out,in) dims + the Q4NX name set.
"""
import sys
import numpy as np
import importlib.util

sys.path.insert(0, "/home/bcloud/1bit-MONSTER/third_party/llama.cpp/gguf-py")
import gguf.constants as C
_o = C.GGMLQuantizationType.__new__
C.GGMLQuantizationType.__new__ = lambda cls, v: (_o(cls, v) if v in C.GGMLQuantizationType._value2member_map_ else int.__new__(cls, v))
C.GGML_QUANT_SIZES[43] = (8192, 5120)
from gguf import GGUFReader, GGUFWriter, GGUFValueType

spec = importlib.util.spec_from_file_location("cq", "/home/bcloud/1bit-MONSTER/tools/convert_float32_bins_to_q4nx.py")
cq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cq)


def main():
    src, ref, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    r = GGUFReader(src)
    rr = GGUFReader(ref)

    # 1. Q4NX tensor logical dims from the c43 reference: tiles -> (out,in)
    #    tile grid: n_tc = in/256, n_tr = out/32, n_tiles = n_tr*n_tc.
    qdims = {}
    for t in rr.tensors:
        if int(t.tensor_type) == 43:
            n_tiles = t.data.nbytes // 5120
            ne = [int(x) for x in t.shape]
            # experts: shape (8192, tpe, n_expert) -> per-expert out=32*? compute
            # from tile count: n_tc*? — derive from the f32 source instead below.
            qdims[t.name] = ne
    print("ref Q4NX tensors:", len(qdims))

    # per-tensor f32 source data (from the FT gguf) -> logical [out,in]:
    # flat gguf data reshaped (ne0 fastest). For a 2-D f32 tensor with gguf ne
    # [a,b], flat reshape = (b, a) gives [ne1, ne0] which == [out, in] as
    # verified (attn_q flat.reshape(1024,2048) corr 1.0 vs packer [out,in]).
    src_map = {t.name: t for t in r.tensors}
    ref_map = {t.name: t for t in rr.tensors}

    # 2. Write output GGUF
    w = GGUFWriter(dst, "zaya")
    # metadata: copy scalar KVs from the source FT gguf
    for fname, f in r.fields.items():
        if fname.startswith("GGUF."):
            continue
        try:
            if f.types[0] == GGUFValueType.STRING and len(f.data) == 1:
                w.add_string(fname, str(f.parts[f.data[0]], "utf-8", "replace"))
            elif f.types[0] == GGUFValueType.UINT32 and len(f.data) == 1:
                w.add_uint32(fname, int(np.frombuffer(f.parts[f.data[0]], dtype=np.uint32)[0]))
            elif f.types[0] == GGUFValueType.INT32 and len(f.data) == 1:
                w.add_int32(fname, int(np.frombuffer(f.parts[f.data[0]], dtype=np.int32)[0]))
            elif f.types[0] == GGUFValueType.UINT64 and len(f.data) == 1:
                w.add_uint64(fname, int(np.frombuffer(f.parts[f.data[0]], dtype=np.uint64)[0]))
            elif f.types[0] == GGUFValueType.INT64 and len(f.data) == 1:
                w.add_int64(fname, int(np.frombuffer(f.parts[f.data[0]], dtype=np.int64)[0]))
            elif f.types[0] == GGUFValueType.FLOAT32 and len(f.data) == 1:
                w.add_float32(fname, float(np.frombuffer(f.parts[f.data[0]], dtype=np.float32)[0]))
            elif f.types[0] == GGUFValueType.BOOL and len(f.data) == 1:
                w.add_bool(fname, bool(np.frombuffer(f.parts[f.data[0]], dtype=np.uint8)[0]))
        except Exception as e:
            print("  skip meta", fname, e)
    # copy array-of-string KVs (tokenizer.ggml.merges etc.) - required by the loader
    for fname, f in r.fields.items():
        if fname.startswith("GGUF.") or fname in ("general.architecture",):
            continue
        try:
            if f.types[0] == GGUFValueType.ARRAY and len(f.types) > 1 and f.types[1] == GGUFValueType.STRING:
                vals = []
                for off in f.data:
                    vals.append(str(f.parts[off], "utf-8", "replace"))
                w.add_array(fname, vals)
                print(f"  copied array {fname}: {len(vals)} entries", flush=True)
        except Exception as e:
            print("  skip array", fname, e)

    n_q = n_f = 0
    for t in r.tensors:
        name = t.name
        ne = [int(x) for x in t.shape]
        raw = t.data  # numpy-shaped already: 2D [out,in]; 3D [exp,out,in]
        if name in qdims and raw.ndim == 2:
            W = np.asarray(raw, dtype=np.float32)
            tiles = cq.pack_q4nx_tile(W)
            n_tiles = tiles.shape[0]
            rt = ref_map.get(name)
            if rt is not None:
                ref_tiles = rt.data.nbytes // 5120
                if ref_tiles != n_tiles:
                    print(f"  WARN {name}: {n_tiles} tiles vs c43 {ref_tiles}", flush=True)
            w.add_tensor(name, tiles.reshape(-1).astype(np.uint8),
                         raw_shape=[n_tiles, 5120], raw_dtype=43)
            n_q += 1
        elif name in qdims and raw.ndim == 3:
            W3 = np.asarray(raw, dtype=np.float32)  # [exp, out, in]
            all_tiles = [cq.pack_q4nx_tile(W3[e]) for e in range(W3.shape[0])]
            tiles = np.concatenate(all_tiles, axis=0)
            n_tiles = tiles.shape[0]
            rt = ref_map.get(name)
            if rt is not None:
                ref_tiles = rt.data.nbytes // 5120
                if ref_tiles != n_tiles:
                    print(f"  WARN {name}: {n_tiles} tiles vs c43 {ref_tiles}", flush=True)
            w.add_tensor(name, tiles.reshape(-1).astype(np.uint8),
                         raw_shape=[n_tiles, 5120], raw_dtype=43)
            n_q += 1
        else:
            arr = np.asarray(raw, dtype=np.float32)
            w.add_tensor(name, arr, raw_dtype=0)
            n_f += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"done: q4nx={n_q} f32={n_f} -> {dst}")


if __name__ == "__main__":
    main()
