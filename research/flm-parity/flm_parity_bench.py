#!/usr/bin/env python3
"""flm-bench-equivalent parity harness — same HTTP shape against any OpenAI-compatible server.

Measures, per (model, prompt_len, max_tokens):
  - TTFT       (s)   time to first token (streamed)
  - prefill    (tok/s) prompt_tokens / ttft        [blended, documented]
  - decode     (tok/s) (n_out-1) / (wall - ttft)   [streamed token timing]
  - wall       (s)

Usage:
  flm_parity_bench.py --base http://127.0.0.1:8099/v1 --model qwen3:0.6b \
      --ppl 32 256 1024 --ngen 64 128 --reps 3
Stdlib only.
"""
import argparse, json, sys, time, urllib.request

def stream_chat(base, model, prompt, max_tokens, temp=0.0):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp, "stream": True}
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n_out = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            ch = chunk.get("choices") or []
            if not ch:
                continue
            delta = ch[0].get("delta") or {}
            tok = delta.get("content") or delta.get("reasoning_content")
            if tok:
                if ttft is None:
                    ttft = time.time() - t0
                n_out += 1
    wall = time.time() - t0
    return ttft, n_out, wall

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--model", default="qwen3:0.6b")
    ap.add_argument("--ppl", type=int, nargs="*", default=[32, 256, 1024])
    ap.add_argument("--ngen", type=int, nargs="*", default=[128])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    filler = ("The quick brown fox jumps over the lazy dog near the river bank while "
              "counting the stars in the clear night sky above the quiet town. ")
    print(f"# parity bench  base={a.base} model={a.model}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("prompt_tokens\tmax_tokens\trep\tttft_s\tprefill_tps\tdecode_tps\twall_s")
    for plen in a.ppl:
        # crude ~4 chars/token prompt; token counts reported by server via usage when avail.
        prompt = (filler * ((plen * 4) // len(filler) + 1))[: plen * 4]
        for ng in a.ngen:
            for rep in range(a.reps):
                try:
                    ttft, n_out, wall = stream_chat(a.base, a.model, prompt, ng)
                    if ttft and wall - ttft > 0 and n_out > 1:
                        dec = (n_out - 1) / (wall - ttft)
                        pre = plen / ttft if ttft > 0 else 0
                        print(f"{plen}\t{ng}\t{rep}\t{ttft:.3f}\t{pre:.1f}\t{dec:.1f}\t{wall:.3f}", flush=True)
                    else:
                        print(f"{plen}\t{ng}\t{rep}\tERR ttft={ttft} n_out={n_out} wall={wall:.3f}", flush=True)
                except Exception as e:
                    print(f"{plen}\t{ng}\t{rep}\tEXC {e}", flush=True)

if __name__ == "__main__":
    main()
