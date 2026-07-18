from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence
from math import log2, ceil

from numpy.typing import DTypeLike

from .constants import GGML_QUANT_SIZES, GGMLQuantizationType, QK_K
from .lazy import LazyNumpyTensor

# Block sizes for simple quant types (from ggml-common.h)
QK1_0 = 128
QK2_0 = 64

import numpy as np


def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})")
    return (*shape[:-1], shape[-1] // block_size * type_size)


def quant_shape_from_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % type_size != 0:
        raise ValueError(f"Quantized tensor bytes per row ({shape[-1]}) is not a multiple of {quant_type.name} type size ({type_size})")
    return (*shape[:-1], shape[-1] // type_size * block_size)


# This is faster than np.vectorize and np.apply_along_axis because it works on more than one row at a time
def _apply_over_grouped_rows(func: Callable[[np.ndarray], np.ndarray], arr: np.ndarray, otype: DTypeLike, oshape: tuple[int, ...]) -> np.ndarray:
    rows = arr.reshape((-1, arr.shape[-1]))
    assert len(rows.shape)
    osize = 1
    for dim in oshape:
        osize *= dim
    out = np.empty(shape=osize, dtype=otype)
    # compute over groups of 16 rows (arbitrary, but seems good for performance)
    n_groups = (rows.shape[0] // 16) or 1
    np.concatenate([func(group).ravel() for group in np.array_split(rows, n_groups)], axis=0, out=out)
    return out.reshape(oshape)


# round away from zero — matches libc roundf(), used by the simple
# (non-K-quant) formats below (Q4_0/Q5_0/Q8_0/TQ1_0/TQ2_0/...).
# ref: https://stackoverflow.com/a/59143326/22827863
def np_roundf(n: np.ndarray) -> np.ndarray:
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b


# round half to even — matches ggml-quants.c's `nearest_int()` bit-trick
# (`fval + 1.5*2^23` then reinterpret), which inherits the FPU's default
# IEEE-754 round-to-nearest-even rounding mode. Used by the K-quant family
# (make_qkx2_quants, make_q3_quants, and the K-quant scale/element
# requantization loops) — a genuinely different rounding rule from
# np_roundf's round-away-from-zero, not just a style choice.
def np_nearest_int(n: np.ndarray) -> np.ndarray:
    return np.round(n)


# Port of make_qkx2_quants from ggml-quants.c — shared by Q2_K, Q4_K, Q5_K.
# x, weights: (..., n) float32, vectorized over the leading "..." batch dims
# (independent sub-blocks). Returns (scale, min, L) with shapes (...,),
# (...,), (..., n) — L already holds the final integer codes in [0, nmax].
# All internal reductions use sequential left-to-right accumulation (a
# Python loop over n, vectorized across the batch dims) rather than
# np.sum's pairwise reduction — matching the C reference's plain `for`-loop
# summation exactly avoids occasional last-bit rounding divergence that a
# different summation order can introduce (see Q3_K's make_q3_quants port
# for a case where this mattered in practice).
def _make_qkx2_quants(x: np.ndarray, weights: np.ndarray, nmax: int,
                       rmin: float, rdelta: float, nstep: int, use_mad: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[-1]

    min_val = x.min(axis=-1)
    max_val = x.max(axis=-1)
    sum_w = np.zeros(x.shape[:-1], dtype=np.float32)
    sum_x = np.zeros(x.shape[:-1], dtype=np.float32)
    for i in range(n):
        sum_w += weights[..., i]
        sum_x += weights[..., i] * x[..., i]

    min_val = np.where(min_val > 0, 0.0, min_val)
    degenerate = max_val == min_val

    scale_range = max_val - min_val
    safe_range = np.where(degenerate, 1.0, scale_range)
    iscale = nmax / safe_range
    scale = 1.0 / iscale

    L = np.clip(np_nearest_int(iscale[..., None] * (x - min_val[..., None])), 0, nmax)
    best_error = np.zeros(x.shape[:-1], dtype=np.float32)
    for i in range(n):
        diff = scale * L[..., i] + min_val - x[..., i]
        diff = np.abs(diff) if use_mad else diff * diff
        best_error += weights[..., i] * diff

    best_scale = scale.copy()
    best_min = min_val.copy()
    best_L = L.copy()

    if nstep >= 1:
        # C reassigns the *outer* `min` (not just the returned best) to
        # `this_min` on every improvement, and the next iteration's iscale
        # is computed against that evolving `(max - min)`, not the original
        # data range. `best_min` tracks exactly that same variable — reuse
        # it here instead of the fixed pre-loop `min_val`/`safe_range`.
        for is_ in range(nstep + 1):
            cur_range = np.where(degenerate, 1.0, max_val - best_min)
            iscale_try = (rmin + rdelta * is_ + nmax) / cur_range
            Laux = np.clip(np_nearest_int(iscale_try[..., None] * (x - best_min[..., None])), 0, nmax)

            sum_l = np.zeros(x.shape[:-1], dtype=np.float32)
            sum_l2 = np.zeros(x.shape[:-1], dtype=np.float32)
            sum_xl = np.zeros(x.shape[:-1], dtype=np.float32)
            for i in range(n):
                li = Laux[..., i]
                wi = weights[..., i]
                sum_l += wi * li
                sum_l2 += wi * li * li
                sum_xl += wi * li * x[..., i]

            D = sum_w * sum_l2 - sum_l * sum_l
            valid = D > 0

            safe_D = np.where(valid, D, 1.0)
            this_scale = np.where(valid, (sum_w * sum_xl - sum_x * sum_l) / safe_D, 0.0)
            this_min = np.where(valid, (sum_l2 * sum_x - sum_l * sum_xl) / safe_D, 0.0)

            pos_min = this_min > 0
            this_min = np.where(pos_min, 0.0, this_min)
            safe_sum_l2 = np.where(sum_l2 > 0, sum_l2, 1.0)
            this_scale = np.where(pos_min, np.where(sum_l2 > 0, sum_xl / safe_sum_l2, 0.0), this_scale)

            cur_error = np.zeros(x.shape[:-1], dtype=np.float32)
            for i in range(n):
                diff = this_scale * Laux[..., i] + this_min - x[..., i]
                diff = np.abs(diff) if use_mad else diff * diff
                cur_error += weights[..., i] * diff

            improve = valid & (cur_error < best_error)
            best_L = np.where(improve[..., None], Laux, best_L)
            best_error = np.where(improve, cur_error, best_error)
            best_scale = np.where(improve, this_scale, best_scale)
            best_min = np.where(improve, this_min, best_min)

    best_L = np.where(degenerate[..., None], 0, best_L)
    best_scale = np.where(degenerate, 0.0, best_scale)
    the_min = np.where(degenerate, -min_val, -best_min)

    return best_scale.astype(np.float32), the_min.astype(np.float32), best_L.astype(np.int32)


# Port of make_qx_quants(n, nmax, x, L, rmse_type=1, qw=NULL) from
# ggml-quants.c — used by Q6_K. Symmetric (no min offset), weight = x*x,
# search over 19 fixed candidate iscale values around -nmax/max (unlike
# make_qkx2_quants/make_q3_quants's per-element greedy refinement, this is
# a much simpler "try N whole candidates, keep the best" search).
def _make_qx_quants_rmse1(x: np.ndarray, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[-1]
    GROUP_MAX_EPS = 1e-15

    abs_x = np.abs(x)
    amax = abs_x.max(axis=-1)
    idx = np.argmax(abs_x, axis=-1)
    max_val = np.take_along_axis(x, idx[..., None], axis=-1)[..., 0]

    all_zero = amax < GROUP_MAX_EPS
    safe_max = np.where(all_zero, 1.0, max_val)
    w = x * x

    def sums_for(iscale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        L = np.clip(np_nearest_int(iscale[..., None] * x), -nmax, nmax - 1)
        sumlx = np.zeros(x.shape[:-1], dtype=np.float32)
        suml2 = np.zeros(x.shape[:-1], dtype=np.float32)
        for i in range(n):
            li = L[..., i]
            sumlx += w[..., i] * x[..., i] * li
            suml2 += w[..., i] * li * li
        return L, sumlx, suml2

    iscale0 = -nmax / safe_max
    best_L, sumlx, suml2 = sums_for(iscale0)
    best_scale = np.where(suml2 > 0, sumlx / suml2, 0.0)
    best = best_scale * sumlx

    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        iscale = -(nmax + 0.1 * is_) / safe_max
        Laux, sumlx2, suml22 = sums_for(iscale)
        improve = (suml22 > 0) & (sumlx2 * sumlx2 > best * suml22)
        new_scale = np.where(suml22 > 0, sumlx2 / suml22, 0.0)
        best_L = np.where(improve[..., None], Laux, best_L)
        best = np.where(improve, new_scale * sumlx2, best)
        best_scale = np.where(improve, new_scale, best_scale)

    best_L = np.where(all_zero[..., None], 0, best_L + nmax)
    best_scale = np.where(all_zero, 0.0, best_scale)
    return best_scale.astype(np.float32), best_L.astype(np.int32)


# Port of quantize_row_iq4_nl_impl's per-sub-block scale search (the
# weight[j]=x[j]^2, quant_weights=NULL path) from ggml-quants.c — shared by
# IQ4_NL and IQ4_XS. `values` is the sorted 16-entry int8 codebook. Returns
# only the per-sub-block scale (matching the C `scales[ib]` array); callers
# do their own final requantization pass against the (possibly fp16-rounded)
# scale, same "Pass 3" pattern as the K-quants. Degenerate sub-blocks
# (amax < EPS) get scale 0 — deliberately NOT specially handled beyond that,
# since with scale 0 the caller's `id = 1/d` naturally becomes 0 and
# `best_index(0)` reproduces the C reference's actual degenerate behavior
# (every element maps to whichever codebook entry is closest to zero, not
# necessarily index 0) instead of guessing "leave everything zeroed".
def _make_iq4_quants(x: np.ndarray, values: np.ndarray, ntry: int) -> np.ndarray:
    n = x.shape[-1]
    GROUP_MAX_EPS = 1e-15
    values_f = values.astype(np.float32)

    abs_x = np.abs(x)
    amax = abs_x.max(axis=-1)
    idx = np.argmax(abs_x, axis=-1)
    max_val = np.take_along_axis(x, idx[..., None], axis=-1)[..., 0]

    degenerate = amax < GROUP_MAX_EPS
    safe_max = np.where(degenerate, 1.0, max_val)
    w = x * x

    def best_index(al: np.ndarray) -> np.ndarray:
        diffs = np.abs(al[..., None] - values_f)
        return np.argmin(diffs, axis=-1)

    def sums_for(id_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        al = id_[..., None] * x
        q = values_f[best_index(al)]
        sumqx = np.zeros(x.shape[:-1], dtype=np.float32)
        sumq2 = np.zeros(x.shape[:-1], dtype=np.float32)
        for j in range(n):
            sumqx += w[..., j] * q[..., j] * x[..., j]
            sumq2 += w[..., j] * q[..., j] * q[..., j]
        return sumqx, sumq2

    d0 = (-safe_max if ntry > 0 else safe_max) / values_f[0]
    id0 = np.where(d0 != 0, 1.0 / d0, 0.0)
    sumqx, sumq2 = sums_for(id0)
    d = np.where(sumq2 > 0, sumqx / sumq2, 0.0)
    best = d * sumqx
    best_d = d.copy()

    for itry in range(-ntry, ntry + 1):
        id_try = (itry + values_f[0]) / safe_max
        sumqx2, sumq22 = sums_for(id_try)
        improve = (sumq22 > 0) & (sumqx2 * sumqx2 > best * sumq22)
        new_d = np.where(sumq22 > 0, sumqx2 / sumq22, 0.0)
        best = np.where(improve, new_d * sumqx2, best)
        best_d = np.where(improve, new_d, best_d)

    return np.where(degenerate, 0.0, best_d).astype(np.float32)


_iq2_grid_cache: dict[type, tuple] = {}


def _iq2_raw_grid_codes(cls) -> np.ndarray:
    """Decode grid_hex into raw l-codes (0..len(grid_map)-1), shape
    (grid_size, 8) — bypasses the grid_map value lookup __Quant.init_grid()
    does, since the C reference's map/neighbour-search construction and
    quantization search both operate on these raw codes (internally
    rescaled to pos=2*l+1), not on grid_map's decoded output magnitudes."""
    bits_per_elem = ceil(log2(len(cls.grid_map)))
    elems_per_byte = 8 // bits_per_elem
    grid = np.frombuffer(cls.grid_hex, dtype=np.uint8)
    grid = grid.reshape((-1, 2))
    grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
    grid = grid[..., 0] | grid[..., 1]
    grid = grid.reshape((-1, 1)) >> np.array([i for i in range(0, 8, 8 // elems_per_byte)], dtype=np.uint8).reshape((1, elems_per_byte))
    grid = (grid & ((1 << bits_per_elem) - 1)).reshape((-1, cls.grid_shape[1]))
    return grid.astype(np.int64)


def _iq2_build_tables(cls, nwant: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of iq2xs_init_impl (the map/neighbours half — grid construction
    is just _iq2_raw_grid_codes). Returns (pos, kmap, neigh_padded, neigh_count):
      pos:  (grid_size, 8) int32, the C's internal 2*l+1 representation.
      kmap: (65536,) int64, codepoint -> grid row index, or -1.
      neigh_padded: (65536, max_neighbours) int32, candidate grid row
        indices for codepoints with no direct grid match (-1 padding past
        neigh_count[code]); rows for codes that ARE in kmap are unused.
      neigh_count: (65536,) int32, valid entry count per row of neigh_padded.
    Built once per class and cached (mirrors ggml_quantize_init()'s
    one-time table construction, not something done per quantize() call)."""
    cached = _iq2_grid_cache.get(cls)
    if cached is not None:
        return cached

    codes = _iq2_raw_grid_codes(cls)  # (grid_size, 8)
    grid_size = codes.shape[0]
    pos = (2 * codes + 1).astype(np.int32)  # (grid_size, 8)

    idx = np.zeros(grid_size, dtype=np.int64)
    for k in range(8):
        idx |= (codes[:, k] << (2 * k))
    kmap = -np.ones(65536, dtype=np.int64)
    kmap[idx] = np.arange(grid_size)

    invalid = np.where(kmap < 0)[0]
    inv_codes = np.zeros((invalid.shape[0], 8), dtype=np.int64)
    for k in range(8):
        inv_codes[:, k] = (invalid >> (2 * k)) & 0x3
    inv_pos = (2 * inv_codes + 1).astype(np.int32)  # (n_invalid, 8)

    neighbours: dict[int, np.ndarray] = {}
    batch = 1024
    for start in range(0, invalid.shape[0], batch):
        end = min(start + batch, invalid.shape[0])
        diff = inv_pos[start:end, None, :].astype(np.int32) - pos[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)  # (b, grid_size)
        order = np.argsort(d2, axis=-1, kind="stable")
        d2_sorted = np.take_along_axis(d2, order, axis=-1)
        # Vectorized tie-inclusive top-`nwant`-distinct-distance-level cut:
        # rank[j] = number of distinct distance values strictly less than
        # d2_sorted[j] (0-based) via a "new value" marker + cumulative sum.
        is_new = np.empty_like(d2_sorted, dtype=bool)
        is_new[:, 0] = True
        is_new[:, 1:] = d2_sorted[:, 1:] != d2_sorted[:, :-1]
        level = np.cumsum(is_new, axis=-1) - 1  # 0-based distinct-value rank
        keep = level < nwant
        counts = keep.sum(axis=-1)
        for bi in range(end - start):
            i = int(invalid[start + bi])
            n = int(counts[bi])
            neighbours[i] = order[bi, :n].astype(np.int32)

    max_n = max((len(v) for v in neighbours.values()), default=0)
    neigh_padded = np.full((65536, max(max_n, 1)), -1, dtype=np.int32)
    for code, cands in neighbours.items():
        neigh_padded[code, :len(cands)] = cands
    neigh_count = np.zeros((65536,), dtype=np.int32)
    for code, cands in neighbours.items():
        neigh_count[code] = len(cands)

    result = (pos, kmap, neigh_padded, neigh_count)
    _iq2_grid_cache[cls] = result
    return result


def _iq1m_grid_lookup(u: np.ndarray, xval_g: np.ndarray, weight_g: np.ndarray, scale: np.ndarray,
                       use_p: np.ndarray, x_p: np.ndarray, x_m: np.ndarray,
                       pos: np.ndarray, kmap: np.ndarray, neigh_padded: np.ndarray, neigh_count: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of iq1_find_best_neighbour2's dispatch (used by IQ1_M): like
    _iq2_grid_lookup, but candidates are scored via the x_p/x_m
    reconstruction-value table (selected per-group by `use_p`) instead of
    raw pos values, and weight is used directly (not sqrt)."""
    grid_idx = kmap[u]
    on_grid = grid_idx >= 0
    max_n = neigh_padded.shape[1]

    safe_u = np.where(on_grid, 0, u)
    cand = neigh_padded[safe_u]  # (..., max_n)
    cand_cnt = neigh_count[safe_u]  # (...)
    cand_valid = np.arange(max_n) < cand_cnt[..., None]
    safe_cand = np.clip(cand, 0, pos.shape[0] - 1)
    pg = pos[safe_cand]  # (..., max_n, 8) values in {1, 3, 5}
    levels = (pg - 1) // 2  # (..., max_n, 8) in {0, 1, 2}
    q_p = x_p[levels]
    q_m = x_m[levels]
    q = np.where(use_p[..., None, None], q_p, q_m)
    diff = scale[..., None, None] * q - xval_g[..., None, :]
    d2 = np.sum(weight_g[..., None, :] * diff * diff, axis=-1)  # (..., max_n)
    d2 = np.where(cand_valid, d2, np.float32(np.inf))
    best_n = np.argmin(d2, axis=-1)
    neigh_grid_idx = np.take_along_axis(cand, best_n[..., None], axis=-1)[..., 0]

    final_idx = np.where(on_grid, grid_idx, neigh_grid_idx)
    final_pos = pos[np.clip(final_idx, 0, pos.shape[0] - 1)]  # (..., 8)
    L = (final_pos - 1) // 2
    return final_idx, L, on_grid


_iq3_grid_cache: dict[type, tuple] = {}


def _iq3_build_tables(cls, nwant: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of iq3xs_init_impl (4-element groups, 3-bit codes, kmap_size=4096)
    — same shape as _iq2_build_tables, see that function's docstring."""
    cached = _iq3_grid_cache.get(cls)
    if cached is not None:
        return cached

    codes = _iq2_raw_grid_codes(cls)  # (grid_size, 4)
    grid_size = codes.shape[0]
    pos = (2 * codes + 1).astype(np.int32)  # (grid_size, 4)
    kmap_size = 4096

    idx = np.zeros(grid_size, dtype=np.int64)
    for k in range(4):
        idx |= (codes[:, k] << (3 * k))
    kmap = -np.ones(kmap_size, dtype=np.int64)
    kmap[idx] = np.arange(grid_size)

    invalid = np.where(kmap < 0)[0]
    inv_codes = np.zeros((invalid.shape[0], 4), dtype=np.int64)
    for k in range(4):
        inv_codes[:, k] = (invalid >> (3 * k)) & 0x7
    inv_pos = (2 * inv_codes + 1).astype(np.int32)

    neighbours: dict[int, np.ndarray] = {}
    batch = 1024
    for start in range(0, invalid.shape[0], batch):
        end = min(start + batch, invalid.shape[0])
        diff = inv_pos[start:end, None, :].astype(np.int32) - pos[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)  # (b, grid_size)
        order = np.argsort(d2, axis=-1, kind="stable")
        d2_sorted = np.take_along_axis(d2, order, axis=-1)
        is_new = np.empty_like(d2_sorted, dtype=bool)
        is_new[:, 0] = True
        is_new[:, 1:] = d2_sorted[:, 1:] != d2_sorted[:, :-1]
        level = np.cumsum(is_new, axis=-1) - 1
        keep = level < nwant
        counts = keep.sum(axis=-1)
        for bi in range(end - start):
            i = int(invalid[start + bi])
            n = int(counts[bi])
            neighbours[i] = order[bi, :n].astype(np.int32)

    max_n = max((len(v) for v in neighbours.values()), default=0)
    neigh_padded = np.full((kmap_size, max(max_n, 1)), -1, dtype=np.int32)
    for code, cands in neighbours.items():
        neigh_padded[code, :len(cands)] = cands
    neigh_count = np.zeros((kmap_size,), dtype=np.int32)
    for code, cands in neighbours.items():
        neigh_count[code] = len(cands)

    result = (pos, kmap, neigh_padded, neigh_count)
    _iq3_grid_cache[cls] = result
    return result


def _iq3_pack_u(codes4: np.ndarray) -> np.ndarray:
    """codes4: (..., 4) int, values 0..7 -> (...,) packed code."""
    u = np.zeros(codes4.shape[:-1], dtype=np.int64)
    for i in range(4):
        u |= (codes4[..., i].astype(np.int64) << (3 * i))
    return u


def _iq3_grid_lookup(u: np.ndarray, xval_g: np.ndarray, weight_g: np.ndarray, scale: np.ndarray,
                      pos: np.ndarray, kmap: np.ndarray, neigh_padded: np.ndarray, neigh_count: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of iq3_find_best_neighbour dispatch, see IQ2_S._grid_lookup."""
    grid_idx = kmap[u]
    on_grid = grid_idx >= 0
    max_n = neigh_padded.shape[1]

    safe_u = np.where(on_grid, 0, u)
    cand = neigh_padded[safe_u]  # (..., max_n)
    cand_cnt = neigh_count[safe_u]  # (...)
    cand_valid = np.arange(max_n) < cand_cnt[..., None]
    safe_cand = np.clip(cand, 0, pos.shape[0] - 1)
    pg = pos[safe_cand]  # (..., max_n, 4)
    diff = scale[..., None, None] * pg - xval_g[..., None, :]
    d2 = np.sum(weight_g[..., None, :] * diff * diff, axis=-1)  # (..., max_n)
    d2 = np.where(cand_valid, d2, np.float32(np.inf))
    best_n = np.argmin(d2, axis=-1)
    neigh_grid_idx = np.take_along_axis(cand, best_n[..., None], axis=-1)[..., 0]

    final_idx = np.where(on_grid, grid_idx, neigh_grid_idx)
    final_pos = pos[np.clip(final_idx, 0, pos.shape[0] - 1)]  # (..., 4)
    L = (final_pos - 1) // 2
    return final_idx, L, on_grid


class QuantError(Exception): ...


_type_traits: dict[GGMLQuantizationType, type[__Quant]] = {}


def quantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if qtype == GGMLQuantizationType.F32:
        return data.astype(np.float32, copy=False)
    elif qtype == GGMLQuantizationType.F16:
        return data.astype(np.float16, copy=False)
    elif (q := _type_traits.get(qtype)) is not None:
        return q.quantize(data)
    else:
        raise NotImplementedError(f"Quantization for {qtype.name} is not yet implemented")


def dequantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if qtype == GGMLQuantizationType.F32:
        return data.view(np.float32)
    elif qtype == GGMLQuantizationType.F16:
        return data.view(np.float16).astype(np.float32)
    elif (q := _type_traits.get(qtype)) is not None:
        return q.dequantize(data)
    else:
        raise NotImplementedError(f"Dequantization for {qtype.name} is not yet implemented")


class __Quant(ABC):
    qtype: GGMLQuantizationType
    block_size: int
    type_size: int

    grid: np.ndarray[Any, np.dtype[np.float32]] | None = None
    grid_shape: tuple[int, int] = (0, 0)
    grid_map: tuple[int | float, ...] = ()
    grid_hex: bytes | None = None

    def __init__(self):
        return TypeError("Quant conversion classes can't have instances")

    def __init_subclass__(cls, qtype: GGMLQuantizationType) -> None:
        cls.qtype = qtype
        cls.block_size, cls.type_size = GGML_QUANT_SIZES[qtype]
        cls.__quantize_lazy: Any = LazyNumpyTensor._wrap_fn(
            cls.__quantize_array,
            meta_noop=(np.uint8, cls.__shape_to_bytes)
        )
        cls.__dequantize_lazy: Any = LazyNumpyTensor._wrap_fn(
            cls.__dequantize_array,
            meta_noop=(np.float32, cls.__shape_from_bytes)
        )
        assert qtype not in _type_traits
        _type_traits[qtype] = cls

    @classmethod
    def init_grid(cls):
        if cls.grid is not None or cls.grid_hex is None:
            return

        bits_per_elem = ceil(log2(len(cls.grid_map)))
        assert bits_per_elem != 0, cls.qtype.name
        elems_per_byte = 8 // bits_per_elem

        grid = np.frombuffer(cls.grid_hex, dtype=np.uint8)
        # decode hexadecimal chars from grid
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        # unpack the grid values
        grid = grid.reshape((-1, 1)) >> np.array([i for i in range(0, 8, 8 // elems_per_byte)], dtype=np.uint8).reshape((1, elems_per_byte))
        grid = (grid & ((1 << bits_per_elem) - 1)).reshape((-1, 1))
        grid_map = np.array(cls.grid_map, dtype=np.float32).reshape((1, -1))
        grid = np.take_along_axis(grid_map, grid, axis=-1)
        cls.grid = grid.reshape((1, 1, *cls.grid_shape))

    @classmethod
    @abstractmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def quantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.astype(np.float32, copy=False)
        shape = rows.shape
        n_blocks = rows.size // cls.block_size
        blocks = rows.reshape((n_blocks, cls.block_size))
        blocks = cls.quantize_blocks(blocks)
        assert blocks.dtype == np.uint8
        assert blocks.shape[-1] == cls.type_size
        return blocks.reshape(cls.__shape_to_bytes(shape))

    @classmethod
    def dequantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.view(np.uint8)
        shape = rows.shape
        n_blocks = rows.size // cls.type_size
        blocks = rows.reshape((n_blocks, cls.type_size))
        blocks = cls.dequantize_blocks(blocks)
        assert blocks.dtype == np.float32
        assert blocks.shape[-1] == cls.block_size
        return blocks.reshape(cls.__shape_from_bytes(shape))

    @classmethod
    def __shape_to_bytes(cls, shape: Sequence[int]):
        return quant_shape_to_byte_shape(shape, cls.qtype)

    @classmethod
    def __shape_from_bytes(cls, shape: Sequence[int]):
        return quant_shape_from_byte_shape(shape, cls.qtype)

    @classmethod
    def __quantize_array(cls, array: np.ndarray) -> np.ndarray:
        return _apply_over_grouped_rows(cls.quantize_rows, arr=array, otype=np.uint8, oshape=cls.__shape_to_bytes(array.shape))

    @classmethod
    def __dequantize_array(cls, array: np.ndarray) -> np.ndarray:
        cls.init_grid()
        return _apply_over_grouped_rows(cls.dequantize_rows, arr=array, otype=np.float32, oshape=cls.__shape_from_bytes(array.shape))

    @classmethod
    def __quantize_lazy(cls, lazy_tensor: LazyNumpyTensor, /) -> Any:
        pass

    @classmethod
    def __dequantize_lazy(cls, lazy_tensor: LazyNumpyTensor, /) -> Any:
        pass

    @classmethod
    def can_quantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> bool:
        return tensor.shape[-1] % cls.block_size == 0

    @classmethod
    def quantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        if not cls.can_quantize(tensor):
            raise QuantError(f"Can't quantize tensor with shape {tensor.shape} to {cls.qtype.name}")
        if isinstance(tensor, LazyNumpyTensor):
            return cls.__quantize_lazy(tensor)
        else:
            return cls.__quantize_array(tensor)

    @classmethod
    def dequantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        if isinstance(tensor, LazyNumpyTensor):
            return cls.__dequantize_lazy(tensor)
        else:
            return cls.__dequantize_array(tensor)


class BF16(__Quant, qtype=GGMLQuantizationType.BF16):
    @classmethod
    # same as ggml_compute_fp32_to_bf16 in ggml-impl.h
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n = blocks.view(np.uint32)
        # force nan to quiet
        n = np.where((n & 0x7fffffff) > 0x7f800000, (n & np.uint32(0xffff0000)) | np.uint32(64 << 16), n)
        # round to nearest even
        n = (np.uint64(n) + (0x7fff + ((n >> 16) & 1))) >> 16
        return n.astype(np.uint16).view(np.uint8)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return (blocks.view(np.int16).astype(np.int32) << 16).view(np.float32)


class Q4_0(__Quant, qtype=GGMLQuantizationType.Q4_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)

        d = max / -8
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np.trunc((blocks * id) + np.float32(8.5), dtype=np.float32).astype(np.uint8).clip(0, 15)

        qs = qs.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([d, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)

        return (d * qs.astype(np.float32))


class Q4_1(__Quant, qtype=GGMLQuantizationType.Q4_1):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)

        d = (max - min) / 15
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32).astype(np.uint8).clip(0, 15)

        qs = qs.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

        d = d.astype(np.float16).view(np.uint8)
        m = min.astype(np.float16).view(np.uint8)

        return np.concatenate([d, m, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        m, qs = np.hsplit(rest, [2])

        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.float32)

        return (d * qs) + m


class Q5_0(__Quant, qtype=GGMLQuantizationType.Q5_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)

        d = max / -16
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        q = np.trunc((blocks * id) + np.float32(16.5), dtype=np.float32).astype(np.uint8).clip(0, 31)

        qs = q.reshape((n_blocks, 2, cls.block_size // 2))
        qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))

        qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([d, qh, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qh, qs = np.hsplit(rest, [4])

        d = d.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)

        qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(0x01)).astype(np.uint8)
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))

        qs = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)

        return (d * qs.astype(np.float32))


class Q5_1(__Quant, qtype=GGMLQuantizationType.Q5_1):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)

        d = (max - min) / 31
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        q = np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32).astype(np.uint8).clip(0, 31)

        qs = q.reshape((n_blocks, 2, cls.block_size // 2))
        qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))

        qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)

        d = d.astype(np.float16).view(np.uint8)
        m = min.astype(np.float16).view(np.uint8)

        return np.concatenate([d, m, qh, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        m, rest = np.hsplit(rest, [2])
        qh, qs = np.hsplit(rest, [4])

        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)

        qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(0x01)).astype(np.uint8)
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))

        qs = (ql | (qh << np.uint8(4))).astype(np.float32)

        return (d * qs) + m


class Q8_0(__Quant, qtype=GGMLQuantizationType.Q8_0):
    @classmethod
    # Implementation of Q8_0 with bit-exact same results as reference implementation in ggml-quants.c
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:

        d = abs(blocks).max(axis=1, keepdims=True) / 127
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)

        # (n_blocks, 2)
        d = d.astype(np.float16).view(np.uint8)
        # (n_blocks, block_size)
        qs = qs.astype(np.int8).view(np.uint8)

        return np.concatenate([d, qs], axis=1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        d, x = np.split(blocks, [2], axis=1)
        d = d.view(np.float16).astype(np.float32)
        x = x.view(np.int8).astype(np.float32)

        return (x * d)


class Q2_K(__Quant, qtype=GGMLQuantizationType.Q2_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        n_sub = QK_K // 16
        sub_size = 16
        q4scale = 15.0

        blocks_3d = blocks.reshape((n_blocks, n_sub, sub_size)).astype(np.float32)
        weights = np.abs(blocks_3d)

        # Port of quantize_row_q2_K_ref.
        scales, mins, _ = _make_qkx2_quants(blocks_3d, weights, nmax=3, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=True)

        max_scale = scales.max(axis=-1)
        max_min = mins.max(axis=-1)

        has_scale = max_scale > 0
        d_all = np.where(has_scale, max_scale / q4scale, 0.0).astype(np.float16).astype(np.float32)
        l_scales = np.where(has_scale[:, None],
                             np.clip(np_nearest_int(np.where(has_scale, q4scale / np.where(has_scale, max_scale, 1.0), 0.0)[:, None] * scales), 0, 15),
                             0).astype(np.uint8)

        has_min = max_min > 0
        dmin_all = np.where(has_min, max_min / q4scale, 0.0).astype(np.float16).astype(np.float32)
        l_mins = np.where(has_min[:, None],
                           np.clip(np_nearest_int(np.where(has_min, q4scale / np.where(has_min, max_min, 1.0), 0.0)[:, None] * mins), 0, 15),
                           0).astype(np.uint8)

        packed_scales = (l_scales | (l_mins << 4)).astype(np.uint8)  # (n_blocks, 16)

        dl = (d_all[:, None] * l_scales.astype(np.float32))[..., None]  # (n_blocks,16,1)
        ml = (dmin_all[:, None] * l_mins.astype(np.float32))[..., None]
        dl_nz = dl != 0
        safe_dl = np.where(dl_nz, dl, 1.0)

        L = np.clip(np_nearest_int((blocks_3d + ml) / safe_dl), 0, 3).astype(np.uint8)
        L = np.where(dl_nz, L, 0).reshape(n_blocks, QK_K)

        qs = np.zeros((n_blocks, QK_K // 4), dtype=np.uint8)
        for j in range(0, QK_K, 128):
            for l in range(32):
                qs[:, j // 4 + l] = (L[:, j + l].astype(np.uint16) |
                    (L[:, j + l + 32].astype(np.uint16) << 2) |
                    (L[:, j + l + 64].astype(np.uint16) << 4) |
                    (L[:, j + l + 96].astype(np.uint16) << 6)).astype(np.uint8)

        d_bytes = d_all.astype(np.float16).view(np.uint8)
        dmin_bytes = dmin_all.astype(np.float16).view(np.uint8)

        return np.concatenate([packed_scales, qs, d_bytes.reshape((n_blocks, 2)), dmin_bytes.reshape((n_blocks, 2))], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        scales, rest = np.hsplit(blocks, [QK_K // 16])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        d, dmin = np.hsplit(rest, [2])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        # (n_blocks, 16, 1)
        dl = (d * (scales & 0xF).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))
        ml = (dmin * (scales >> 4).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))

        shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))

        qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & np.uint8(3)

        qs = qs.reshape((n_blocks, QK_K // 16, 16)).astype(np.float32)

        qs = dl * qs - ml

        return qs.reshape((n_blocks, -1))


class Q3_K(__Quant, qtype=GGMLQuantizationType.Q3_K):
    GROUP_MAX_EPS = 1e-15  # matches ggml-quants.c's #define GROUP_MAX_EPS 1e-15f

    # Port of make_q3_quants(n, nmax, x, L, do_rmse=true) from ggml-quants.c.
    # x: (..., n) float32. Returns per-row scale (...,) — matches the C
    # function's own return value (sumlx/suml2 from the RMSE-refined fit,
    # NOT the naive max/nmax estimate) — callers only need this scale for
    # the outer max-scale search; the actual per-element L codes get
    # recomputed later from the final 6-bit-quantized scale, exactly like
    # the C reference does in its own second pass.
    @classmethod
    def _make_q3_quants(cls, x: np.ndarray, nmax: int = 4) -> np.ndarray:
        n = x.shape[-1]
        abs_x = np.abs(x)
        amax = abs_x.max(axis=-1)
        idx = np.argmax(abs_x, axis=-1)
        max_val = np.take_along_axis(x, idx[..., None], axis=-1)[..., 0]

        all_zero = amax < cls.GROUP_MAX_EPS
        safe_max = np.where(all_zero, 1.0, max_val)
        iscale = -nmax / safe_max

        L = np.clip(np_nearest_int(iscale[..., None] * x), -nmax, nmax - 1)
        w = x * x
        # Sequential left-to-right accumulation (matches the C reference's
        # `for i: sumlx += w*x[i]*l` loop exactly) rather than np.sum's
        # pairwise reduction, which can round differently in the last bit
        # and occasionally flip a borderline element's rounded code.
        sumlx = np.zeros(x.shape[:-1], dtype=np.float32)
        suml2 = np.zeros(x.shape[:-1], dtype=np.float32)
        for i in range(n):
            sumlx += w[..., i] * x[..., i] * L[..., i]
            suml2 += w[..., i] * L[..., i] * L[..., i]

        for _ in range(5):
            for i in range(n):
                wi = w[..., i]
                xi = x[..., i]
                Li = L[..., i]
                slx = sumlx - wi * xi * Li
                pos = slx > 0
                sl2 = suml2 - wi * Li * Li
                safe_slx = np.where(pos, slx, 1.0)
                new_l = np.clip(np_nearest_int(xi * sl2 / safe_slx), -nmax, nmax - 1)
                changed = new_l != Li
                new_slx = slx + wi * xi * new_l
                new_sl2 = sl2 + wi * new_l * new_l
                improves = (new_sl2 > 0) & (new_slx * new_slx * suml2 > sumlx * sumlx * new_sl2)
                apply = pos & changed & improves
                L[..., i] = np.where(apply, new_l, Li)
                sumlx = np.where(apply, new_slx, sumlx)
                suml2 = np.where(apply, new_sl2, suml2)

        scale = np.where(all_zero, 0.0, np.where(suml2 > 0, sumlx / suml2, 0.0))
        return scale.astype(np.float32)

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        n_sub = QK_K // 16  # 16 sub-blocks
        sub_size = 16

        blocks_3d = blocks.reshape((n_blocks, n_sub, sub_size)).astype(np.float32)

        # Pass 1 (matches quantize_row_q3_K_ref): per-subblock RMSE-optimal
        # scale estimate, used only to find the block's overall max |scale|.
        scales = cls._make_q3_quants(blocks_3d, nmax=4)  # (n_blocks, 16)

        abs_scales = np.abs(scales)
        amax_idx = np.argmax(abs_scales, axis=-1)
        max_scale = np.take_along_axis(scales, amax_idx[:, None], axis=-1)[:, 0]  # (n_blocks,) signed

        has_scale = max_scale != 0
        safe_max_scale = np.where(has_scale, max_scale, 1.0)
        iscale = np.where(has_scale, -32.0 / safe_max_scale, 0.0)
        d_all = np.where(has_scale, 1.0 / np.where(has_scale, iscale, 1.0), 0.0)  # 1/iscale, 0 where no scale
        # C stores y.d as fp16 THEN reads it back (already-rounded) for pass
        # 3's element requantization — match that precision loss exactly,
        # not the full fp32 d_all, or borderline elements round differently.
        d_all = d_all.astype(np.float16).astype(np.float32)

        l_scales = np.clip(np_nearest_int(iscale[:, None] * scales), -32, 31).astype(np.int32) + 32  # (n_blocks,16) in [0,63]
        l_scales = np.where(has_scale[:, None], l_scales, 0)  # C: memset(scales,0,12) when max_scale==0

        # Pack 6-bit scales into 12 bytes exactly like the C reference.
        packed_scales = np.zeros((n_blocks, 12), dtype=np.uint8)
        for j in range(16):
            l = l_scales[:, j].astype(np.uint8)
            if j < 8:
                packed_scales[:, j] |= (l & 0x0F)
            else:
                packed_scales[:, j - 8] |= ((l & 0x0F) << 4)
            packed_scales[:, 8 + (j % 4)] |= (((l >> 4) & 0x03) << (2 * (j // 4)))

        # Pass 3 (matches C): decode the just-packed 6-bit scales back out
        # and re-quantize every element against that FINAL rounded scale
        # (not the raw make_q3_quants scale from pass 1).
        sc = np.zeros((n_blocks, 16), dtype=np.float32)
        for j in range(16):
            if j < 8:
                lo = packed_scales[:, j] & 0x0F
            else:
                lo = (packed_scales[:, j - 8] >> 4) & 0x0F
            hi = (packed_scales[:, 8 + (j % 4)] >> (2 * (j // 4))) & 0x03
            sc[:, j] = ((lo | (hi << 4)).astype(np.int32) - 32).astype(np.float32)

        dl = (d_all[:, None] * sc)  # (n_blocks, 16)
        dl_nz = dl != 0
        safe_dl = np.where(dl_nz, dl, 1.0)[..., None]  # (n_blocks, 16, 1)

        L = np.clip(np_nearest_int(blocks_3d / safe_dl), -4, 3).astype(np.int32) + 4  # (n_blocks,16,16) in [0,7]
        L = np.where(dl_nz[..., None], L, 0)
        L = L.reshape(n_blocks, QK_K)  # flat element order matches C's `j` index (0..255)

        high = L > 3
        L = np.where(high, L - 4, L).astype(np.uint8)  # now in [0,3]

        # hmask: element (bitplane*32 + byte) -> hmask[byte] bit `bitplane`.
        hmask = np.packbits(high.reshape(n_blocks, 8, 32), axis=1, bitorder="little").reshape(n_blocks, 32)

        # qs: for each 128-element half, 4 elements 32 apart share a byte
        # (2 bits each): qs[half*32+l] = L[half*128+l] | L[+32]<<2 | L[+64]<<4 | L[+96]<<6.
        Lg = L.reshape(n_blocks, 2, 4, 32)
        qs = (Lg[:, :, 0, :] | (Lg[:, :, 1, :] << 2) | (Lg[:, :, 2, :] << 4) | (Lg[:, :, 3, :] << 6))
        qs = qs.astype(np.uint8).reshape(n_blocks, 64)

        d_bytes = d_all.astype(np.float16).view(np.uint8).reshape(n_blocks, 2)

        return np.concatenate([hmask, qs, packed_scales, d_bytes], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        hmask, rest = np.hsplit(blocks, [QK_K // 8])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [12])

        d = d.view(np.float16).astype(np.float32)

        lscales, hscales = np.hsplit(scales, [8])
        lscales = lscales.reshape((n_blocks, 1, 8)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
        lscales = lscales.reshape((n_blocks, 16))
        hscales = hscales.reshape((n_blocks, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
        hscales = hscales.reshape((n_blocks, 16))
        scales = (lscales & np.uint8(0x0F)) | ((hscales & np.uint8(0x03)) << np.uint8(4))
        scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)

        dl = (d * scales).reshape((n_blocks, 16, 1))

        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = ql.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(3)
        qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(1))
        qh = qh ^ np.uint8(1)
        q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)

        return (dl * q).reshape((n_blocks, QK_K))


class Q4_K(__Quant, qtype=GGMLQuantizationType.Q4_K):
    K_SCALE_SIZE = 12

    @staticmethod
    def get_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_blocks = scales.shape[0]
        scales = scales.view(np.uint8)
        scales = scales.reshape((n_blocks, 3, 4))
        d, m, m_d = np.split(scales, 3, axis=-2)

        sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
        min = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)

        return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        n_sub = QK_K // 32  # 8 sub-blocks
        sub_size = 32

        blocks_3d = blocks.reshape((n_blocks, n_sub, sub_size)).astype(np.float32)

        # weights[l] = av_x + |x[l]|, av_x = sqrt(mean(x^2)) per sub-block —
        # matches quantize_row_q4_K_ref's weighting exactly (different from
        # Q2_K's plain |x| weights).
        sum_x2 = np.zeros((n_blocks, n_sub), dtype=np.float32)
        for l in range(sub_size):
            sum_x2 += blocks_3d[..., l] * blocks_3d[..., l]
        av_x = np.sqrt(sum_x2 / sub_size)
        weights = av_x[..., None] + np.abs(blocks_3d)

        scales, mins, _ = _make_qkx2_quants(blocks_3d, weights, nmax=15, rmin=-1.0, rdelta=0.1, nstep=20, use_mad=False)

        max_scale = scales.max(axis=-1)
        max_min = mins.max(axis=-1)

        has_scale = max_scale > 0
        inv_scale = np.where(has_scale, 63.0 / np.where(has_scale, max_scale, 1.0), 0.0)
        has_min = max_min > 0
        inv_min = np.where(has_min, 63.0 / np.where(has_min, max_min, 1.0), 0.0)

        ls = np.clip(np_nearest_int(inv_scale[:, None] * scales), 0, 63).astype(np.uint8)
        lm = np.clip(np_nearest_int(inv_min[:, None] * mins), 0, 63).astype(np.uint8)

        d_all = np.where(has_scale, max_scale / 63.0, 0.0).astype(np.float16).astype(np.float32)
        dmin_all = np.where(has_min, max_min / 63.0, 0.0).astype(np.float16).astype(np.float32)

        # Port of quantize_row_q4_K_ref's scale-packing loop (j=0..7):
        #   j<4:  scales[j]=ls[j];  scales[j+4]=lm[j]
        #   j>=4: scales[j+4] = (ls[j]&0xF)|((lm[j]&0xF)<<4)
        #         scales[j-4] |= (ls[j]>>4)<<6;  scales[j] |= (lm[j]>>4)<<6
        # (here "scales[j-4]"/"scales[j]" for j>=4 mean bytes 0-3/4-7, already
        # written by the j<4 iterations — vectorized directly below instead
        # of replaying the loop's incremental OR-ing.)
        packed = np.zeros((n_blocks, cls.K_SCALE_SIZE), dtype=np.uint8)
        for j in range(4):
            packed[:, j]     = (ls[:, j] & 0x3F) | (((ls[:, j + 4] >> 4) & 0x3) << 6)
            packed[:, j + 4] = (lm[:, j] & 0x3F) | (((lm[:, j + 4] >> 4) & 0x3) << 6)
            packed[:, j + 8] = (ls[:, j + 4] & 0xF) | ((lm[:, j + 4] & 0xF) << 4)

        # Pass 3 (matches C): decode sc/m for each of the 8 sub-blocks back
        # out of the just-packed bytes (get_scale_min_k4) and re-quantize
        # every element against those final rounded values.
        sc = np.zeros((n_blocks, n_sub), dtype=np.float32)
        m = np.zeros((n_blocks, n_sub), dtype=np.float32)
        for j in range(n_sub):
            if j < 4:
                sc[:, j] = (packed[:, j] & 0x3F).astype(np.float32)
                m[:, j] = (packed[:, j + 4] & 0x3F).astype(np.float32)
            else:
                sc[:, j] = ((packed[:, j + 4] & 0xF) | (((packed[:, j - 4] >> 6) & 0x3) << 4)).astype(np.float32)
                m[:, j] = ((packed[:, j + 4] >> 4) | (((packed[:, j] >> 6) & 0x3) << 4)).astype(np.float32)

        dl = (d_all[:, None] * sc)[..., None]  # (n_blocks, 8, 1)
        ml = (dmin_all[:, None] * m)[..., None]
        dl_nz = dl != 0
        safe_dl = np.where(dl_nz, dl, 1.0)

        L = np.clip(np_nearest_int((blocks_3d + ml) / safe_dl), 0, 15).astype(np.uint8)
        L = np.where(dl_nz, L, 0).reshape(n_blocks, QK_K)

        # qs: for each 64-element half, low/high nibble = elements l and l+32.
        qs = np.zeros((n_blocks, QK_K // 2), dtype=np.uint8)
        idx = 0
        for j in range(0, QK_K, 64):
            qs[:, idx:idx + 32] = L[:, j:j + 32] | (L[:, j + 32:j + 64] << 4)
            idx += 32

        d_bytes = d_all.astype(np.float16).view(np.uint8)
        dmin_bytes = dmin_all.astype(np.float16).view(np.uint8)

        return np.concatenate([d_bytes.reshape((n_blocks, 2)), dmin_bytes.reshape((n_blocks, 2)), packed, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, qs = np.hsplit(rest, [cls.K_SCALE_SIZE])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        sc, m = Q4_K.get_scale_min(scales)

        d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))

        qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32)).astype(np.float32)

        return (d * qs - dm).reshape((n_blocks, QK_K))


class Q5_K(__Quant, qtype=GGMLQuantizationType.Q5_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        n_sub = QK_K // 32  # 8 sub-blocks
        sub_size = 32

        blocks_3d = blocks.reshape((n_blocks, n_sub, sub_size)).astype(np.float32)

        sum_x2 = np.zeros((n_blocks, n_sub), dtype=np.float32)
        for l in range(sub_size):
            sum_x2 += blocks_3d[..., l] * blocks_3d[..., l]
        av_x = np.sqrt(sum_x2 / sub_size)
        weights = av_x[..., None] + np.abs(blocks_3d)

        scales, mins, _ = _make_qkx2_quants(blocks_3d, weights, nmax=31, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=False)

        max_scale = scales.max(axis=-1)
        max_min = mins.max(axis=-1)

        has_scale = max_scale > 0
        inv_scale = np.where(has_scale, 63.0 / np.where(has_scale, max_scale, 1.0), 0.0)
        has_min = max_min > 0
        inv_min = np.where(has_min, 63.0 / np.where(has_min, max_min, 1.0), 0.0)

        ls = np.clip(np_nearest_int(inv_scale[:, None] * scales), 0, 63).astype(np.uint8)
        lm = np.clip(np_nearest_int(inv_min[:, None] * mins), 0, 63).astype(np.uint8)

        d_all = np.where(has_scale, max_scale / 63.0, 0.0).astype(np.float16).astype(np.float32)
        dmin_all = np.where(has_min, max_min / 63.0, 0.0).astype(np.float16).astype(np.float32)

        # Same scale/min packing as Q4_K (see the comment there).
        packed = np.zeros((n_blocks, Q4_K.K_SCALE_SIZE), dtype=np.uint8)
        for j in range(4):
            packed[:, j]     = (ls[:, j] & 0x3F) | (((ls[:, j + 4] >> 4) & 0x3) << 6)
            packed[:, j + 4] = (lm[:, j] & 0x3F) | (((lm[:, j + 4] >> 4) & 0x3) << 6)
            packed[:, j + 8] = (ls[:, j + 4] & 0xF) | ((lm[:, j + 4] & 0xF) << 4)

        sc = np.zeros((n_blocks, n_sub), dtype=np.float32)
        m = np.zeros((n_blocks, n_sub), dtype=np.float32)
        for j in range(n_sub):
            if j < 4:
                sc[:, j] = (packed[:, j] & 0x3F).astype(np.float32)
                m[:, j] = (packed[:, j + 4] & 0x3F).astype(np.float32)
            else:
                sc[:, j] = ((packed[:, j + 4] & 0xF) | (((packed[:, j - 4] >> 6) & 0x3) << 4)).astype(np.float32)
                m[:, j] = ((packed[:, j + 4] >> 4) | (((packed[:, j] >> 6) & 0x3) << 4)).astype(np.float32)

        dl = (d_all[:, None] * sc)[..., None]
        ml = (dmin_all[:, None] * m)[..., None]
        dl_nz = dl != 0
        safe_dl = np.where(dl_nz, dl, 1.0)

        L = np.clip(np_nearest_int((blocks_3d + ml) / safe_dl), 0, 31).astype(np.uint8)
        L = np.where(dl_nz, L, 0).reshape(n_blocks, QK_K)

        # qh: element (k*32 + j), k=0..7 -> qh[j] bit k. For 64-element group
        # g (n=g*64), (l1,l2) = (L[n+j], L[n+j+32]) map to k=2g,2g+1 — i.e.
        # exactly the natural (8,32) row-major reshape of the flat 256 array
        # (same bit-plane trick as Q3_K's hmask).
        hi = L > 15
        L_low = np.where(hi, L - 16, L).astype(np.uint8)
        qh = np.packbits(hi.reshape(n_blocks, 8, 32), axis=1, bitorder="little").reshape(n_blocks, 32)

        # qs: for each 64-element group g, byte j = low-nibble(L[g*64+j]) |
        # high-nibble(L[g*64+32+j]).
        Lg = L_low.reshape(n_blocks, 4, 2, 32)
        qs = (Lg[:, :, 0, :] | (Lg[:, :, 1, :] << 4)).astype(np.uint8).reshape(n_blocks, 128)

        d_bytes = d_all.astype(np.float16).view(np.uint8)
        dmin_bytes = dmin_all.astype(np.float16).view(np.uint8)

        return np.concatenate([d_bytes.reshape((n_blocks, 2)), dmin_bytes.reshape((n_blocks, 2)), packed, qh, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, rest = np.hsplit(rest, [Q4_K.K_SCALE_SIZE])
        qh, qs = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        sc, m = Q4_K.get_scale_min(scales)

        d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))

        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
        qh = (qh & np.uint8(0x01)).reshape((n_blocks, -1, 32))
        q = (ql | (qh << np.uint8(4))).astype(np.float32)

        return (d * q - dm).reshape((n_blocks, QK_K))


class Q6_K(__Quant, qtype=GGMLQuantizationType.Q6_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        n_sub = QK_K // 16  # 16 sub-blocks
        sub_size = 16
        GROUP_MAX_EPS = 1e-15

        blocks_3d = blocks.reshape((n_blocks, n_sub, sub_size)).astype(np.float32)

        # Port of quantize_row_q6_K_ref: per-subblock symmetric scale search
        # via make_qx_quants(rmse_type=1), then pick the sub-block with max
        # |scale| (keeping its sign) for the block-level d.
        scales, _ = _make_qx_quants_rmse1(blocks_3d, nmax=32)  # (n_blocks, 16)

        abs_scales = np.abs(scales)
        amax_idx = np.argmax(abs_scales, axis=-1)
        max_scale = np.take_along_axis(scales, amax_idx[:, None], axis=-1)[:, 0]  # signed
        has_scale = np.abs(max_scale) >= GROUP_MAX_EPS

        iscale = np.where(has_scale, -128.0 / np.where(has_scale, max_scale, 1.0), 0.0)
        d_all = np.where(has_scale, 1.0 / np.where(has_scale, iscale, 1.0), 0.0).astype(np.float16).astype(np.float32)

        l_scales = np.clip(np_nearest_int(iscale[:, None] * scales), None, 127).astype(np.int32)
        l_scales = np.where(has_scale[:, None], np.clip(l_scales, -128, 127), 0).astype(np.int8)

        dl = (d_all[:, None] * l_scales.astype(np.float32))[..., None]  # (n_blocks, 16, 1)
        dl_nz = dl != 0
        safe_dl = np.where(dl_nz, dl, 1.0)

        L = np.clip(np_nearest_int(blocks_3d / safe_dl), -32, 31).astype(np.int32) + 32  # [0,63]
        L = np.where(dl_nz, L, 0).astype(np.uint8).reshape(n_blocks, QK_K)

        # 128-element groups; within each, 4 "quarters" 32 apart pack into
        # ql (4-bit low nibbles, 2 quarters/byte) and qh (2-bit high bits,
        # 4 quarters/byte).
        Lg = L.reshape(n_blocks, 2, 4, 32)
        q_low = Lg & 0x0F
        q_high = (Lg >> 4) & 0x03

        ql = np.zeros((n_blocks, 2, 64), dtype=np.uint8)
        ql[:, :, 0:32] = q_low[:, :, 0, :] | (q_low[:, :, 2, :] << 4)
        ql[:, :, 32:64] = q_low[:, :, 1, :] | (q_low[:, :, 3, :] << 4)
        ql = ql.reshape(n_blocks, QK_K // 2)

        qh = (q_high[:, :, 0, :] | (q_high[:, :, 1, :] << 2) | (q_high[:, :, 2, :] << 4) | (q_high[:, :, 3, :] << 6))
        qh = qh.astype(np.uint8).reshape(n_blocks, QK_K // 4)

        d_bytes = d_all.astype(np.float16).view(np.uint8)

        return np.concatenate([ql, qh, l_scales.view(np.uint8), d_bytes.reshape((n_blocks, 2))], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        ql, rest = np.hsplit(blocks, [QK_K // 2])
        qh, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [QK_K // 16])

        scales = scales.view(np.int8).astype(np.float32)
        d = d.view(np.float16).astype(np.float32)
        d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

        ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = (qh & np.uint8(0x03)).reshape((n_blocks, -1, 32))
        q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
        q = q.reshape((n_blocks, QK_K // 16, -1)).astype(np.float32)

        return (d * q).reshape((n_blocks, QK_K))


class TQ1_0(__Quant, qtype=GGMLQuantizationType.TQ1_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)
        qs = (qs.astype(np.int8) + np.int8(1)).astype(np.uint8)

        qs0, qs1, qh = qs[..., :(32 * 5)], qs[..., (32 * 5):(48 * 5)], qs[..., (48 * 5):]
        qs0 = qs0.reshape((n_blocks, -1, 5, 32)) * np.array([81, 27, 9, 3, 1], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs0 = np.sum(qs0, axis=-2).reshape((n_blocks, -1))
        qs1 = qs1.reshape((n_blocks, -1, 5, 16)) * np.array([81, 27, 9, 3, 1], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs1 = np.sum(qs1, axis=-2).reshape((n_blocks, -1))
        qh = qh.reshape((n_blocks, -1, 4, 4)) * np.array([81, 27, 9, 3], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = np.sum(qh, axis=-2).reshape((n_blocks, -1))
        qs = np.concatenate([qs0, qs1, qh], axis=-1)
        qs = (qs.astype(np.uint16) * 256 + (243 - 1)) // 243

        qs = qs.astype(np.uint8)
        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([qs, d], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        qs, rest = np.hsplit(blocks, [(QK_K - 4 * QK_K // 64) // 5])
        qh, d = np.hsplit(rest, [QK_K // 64])

        d = d.view(np.float16).astype(np.float32)

        qs0, qs1 = qs[..., :32], qs[..., 32:]
        qs0 = qs0.reshape((n_blocks, -1, 1, 32)) * np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs0 = qs0.reshape((n_blocks, -1))
        qs1 = qs1.reshape((n_blocks, -1, 1, 16)) * np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs1 = qs1.reshape((n_blocks, -1))
        qh = qh.reshape((n_blocks, -1, 1, 4)) * np.array([1, 3, 9, 27], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = qh.reshape((n_blocks, -1))
        qs = np.concatenate([qs0, qs1, qh], axis=-1)
        qs = ((qs.astype(np.uint16) * 3) >> 8).astype(np.int8) - np.int8(1)

        return (d * qs.astype(np.float32))


class TQ2_0(__Quant, qtype=GGMLQuantizationType.TQ2_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)
        qs = (qs.astype(np.int8) + np.int8(1)).astype(np.uint8)

        qs = qs.reshape((n_blocks, -1, 4, 32)) << np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qs = qs[..., 0, :] | qs[..., 1, :] | qs[..., 2, :] | qs[..., 3, :]
        qs = qs.reshape((n_blocks, -1))

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([qs, d], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        qs, d = np.hsplit(blocks, [QK_K // 4])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qs = (qs & 0x03).reshape((n_blocks, -1)).astype(np.int8) - np.int8(1)

        return (d * qs.astype(np.float32))


class MXFP4(__Quant, qtype=GGMLQuantizationType.MXFP4):
    # e2m1 values (doubled)
    # ref: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    kvalues = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

    @staticmethod
    # see ggml_e8m0_to_fp32_half in ggml-impl.h
    def e8m0_to_fp32_half(x: np.ndarray) -> np.ndarray:
        bits = np.where(x < 2, np.uint32(0x00200000) << np.uint32(x), np.uint32(x - 1) << np.uint32(23))
        return bits.view(np.float32)

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d = abs(blocks).max(axis=-1, keepdims=True)

        with np.errstate(divide="ignore"):
            e = np.where(d > 0, np.floor(np.log2(d)) - 2 + 127, 0).astype(np.uint8)

        d = cls.e8m0_to_fp32_half(e)

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape((1, 1, 16))

        errs = np.abs(d.reshape((n_blocks, 1, 1)) * kvalues.astype(np.float32) - blocks.reshape((n_blocks, cls.block_size, 1)))
        best = np.argmin(errs, axis=-1, keepdims=True)

        qs = best.reshape(n_blocks, 2, cls.block_size // 2).astype(np.uint8)
        qs = qs[:, 0] | (qs[:, 1] << np.uint8(4))

        qs = qs.reshape((n_blocks, cls.block_size // 2))

        return np.concatenate([e, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        e, qs = np.hsplit(blocks, [1])

        d = cls.e8m0_to_fp32_half(e)

        qs = qs.reshape((n_blocks, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
        qs = (qs & np.uint8(0x0F)).view(np.int8)

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape(1, 1, 16)
        qs = np.take_along_axis(kvalues, qs, axis=-1).reshape((n_blocks, cls.block_size))

        return (d * qs.astype(np.float32))


class NVFP4(__Quant, qtype=GGMLQuantizationType.NVFP4):
    # E2M1 values doubled (kvalues_mxfp4 convention)
    kvalues = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

    @staticmethod
    def ue4m3_to_fp32(x: np.ndarray) -> np.ndarray:
        """Decode unsigned E4M3 (bias=7) to float, with 0.5 factor for kvalues convention."""
        exp = (x >> 3).astype(np.int32) & 0xF
        man = (x & 0x7).astype(np.float32)
        raw = np.where(
            exp == 0,
            man * 2**-9,
            (1.0 + man / 8.0) * (2.0 ** (exp.astype(np.float32) - 7)))
        return np.where((x == 0) | (x == 0x7F), 0.0, raw * 0.5)

    @staticmethod
    def fp32_to_ue4m3(x: np.ndarray) -> np.ndarray:
        """Vectorized float32 to unsigned E4M3, matching ggml_fp32_to_ue4m3 in C."""
        x = np.clip(x, 0.0, 448.0).astype(np.float32)
        bits = x.view(np.uint32)
        fp32_exp = ((bits >> 23) & 0xFF).astype(np.int32) - 127
        fp32_man = ((bits >> 20) & 0x7).astype(np.int32)
        ue4m3_exp = fp32_exp + 7

        sub_man = np.clip((x * 512.0 + 0.5).astype(np.int32), 0, 7)
        sub_result = np.where(sub_man >= 1, sub_man, 0).astype(np.uint8)

        round_bit = ((bits >> 19) & 1).astype(np.int32)
        man = fp32_man + round_bit
        exp = ue4m3_exp.copy()
        overflow = man > 7
        man = np.where(overflow, 0, man)
        exp = np.where(overflow, exp + 1, exp)
        normal_result = np.where(exp >= 15, np.uint8(0x7E), ((exp << 3) | man).astype(np.uint8))

        return np.where(x <= 0.0, np.uint8(0),
                        np.where(ue4m3_exp <= 0, sub_result,
                        np.where(ue4m3_exp >= 15, np.uint8(0x7E), normal_result)))

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        # NVFP4: 4 super-blocks of 16 elements each, each with its own E4M3 scale
        # block_size = 64, type_size = 4 + 32 = 36
        super_blocks = blocks.reshape((n_blocks, 4, 16))

        # Compute scale per super-block as max absolute value. kvalues are
        # 2x the true E2M1 values (max magnitude 6.0, stored doubled as 12)
        # — the UE4M3 scale maps amax to the *undoubled* max (6.0), matching
        # quantize_row_nvfp4_ref's `amax / 6.0f`.
        amax = abs(super_blocks).max(axis=-1)  # (n_blocks, 4)
        d = cls.fp32_to_ue4m3(amax / 6.0)
        d_f32 = cls.ue4m3_to_fp32(d)  # (n_blocks, 4)

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape(1, 1, 16)

        # For each super-block, find best kvalue match
        errs = np.abs(d_f32[..., None, None] * kvalues.astype(np.float32) - super_blocks[..., None])
        best = np.argmin(errs, axis=-1)  # (n_blocks, 4, 16)

        # Pack 4-bit values: 2 per byte
        lo = best[..., :8].astype(np.uint8)
        hi = best[..., 8:].astype(np.uint8)
        qs = lo | (hi << np.uint8(4))
        qs = qs.reshape((n_blocks, 32))

        return np.concatenate([d.reshape((n_blocks, 4)), qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_super = blocks.shape[0]

        d_bytes, qs = np.hsplit(blocks, [4])
        d = cls.ue4m3_to_fp32(d_bytes).reshape(n_super, 4, 1)

        qs = qs.reshape(n_super, 4, 8)
        lo = (qs & np.uint8(0x0F)).view(np.int8)
        hi = (qs >> np.uint8(4)).view(np.int8)
        vals = np.concatenate([lo, hi], axis=-1)

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape(1, 1, 16)
        vals = np.take_along_axis(kvalues, vals, axis=-1)

        return (d * vals.astype(np.float32)).reshape(n_super, 64)


class IQ2_XXS(__Quant, qtype=GGMLQuantizationType.IQ2_XXS):
    ksigns: bytes = (
        b"\x00\x81\x82\x03\x84\x05\x06\x87\x88\x09\x0a\x8b\x0c\x8d\x8e\x0f"
        b"\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f"
        b"\xa0\x21\x22\xa3\x24\xa5\xa6\x27\x28\xa9\xaa\x2b\xac\x2d\x2e\xaf"
        b"\x30\xb1\xb2\x33\xb4\x35\x36\xb7\xb8\x39\x3a\xbb\x3c\xbd\xbe\x3f"
        b"\xc0\x41\x42\xc3\x44\xc5\xc6\x47\x48\xc9\xca\x4b\xcc\x4d\x4e\xcf"
        b"\x50\xd1\xd2\x53\xd4\x55\x56\xd7\xd8\x59\x5a\xdb\x5c\xdd\xde\x5f"
        b"\x60\xe1\xe2\x63\xe4\x65\x66\xe7\xe8\x69\x6a\xeb\x6c\xed\xee\x6f"
        b"\xf0\x71\x72\xf3\x74\xf5\xf6\x77\x78\xf9\xfa\x7b\xfc\x7d\x7e\xff"
    )

    # iq2xxs_grid, but with each byte of the original packed in 2 bits,
    # by mapping 0x08 to 0, 0x19 to 1, and 0x2b to 2.
    grid_shape = (256, 8)
    grid_map = (0x08, 0x19, 0x2b)
    grid_hex = (
        b"00000200050008000a00110014002000220028002a0041004400500058006100"
        b"6400800082008a00a20001010401100115014001840198010002020222028202"
        b"010404041004210424044004420448046004810484049004a404000502050805"
        b"200546056905800591050906100640068406a406000805080808140828084108"
        b"440850085208880804094009020a140a01100410101021104010601084109010"
        b"951000110811201150115a118011241245120014081420142514491480141815"
        b"6215001616160118041810184018811800190519a019511a002002200a204420"
        b"6120802082202921482100220222012404241024402456240025412564259026"
        b"082820289428442a014004401040184021402440404048405640604081408440"
        b"9040004120416141804185410142104248425642684200440844204480449944"
        b"124524450046014804481048404845480049584961498249454a904a00500850"
        b"1150195020508050885004514251a4519152905492540a550156545600581158"
        b"195864584059085a046010604060686000615561186260620064056410651265"
        b"84654268008002800a8041808280048118814081118201840484108415844084"
        b"608400854685948509864086608602880489118a0490109024904090a1901691"
        b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
    )

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        n_blocks = blocks.shape[0]
        n_sub = QK_K // 32  # 8 sub-blocks of 32 elements
        sub_size = 32
        ks = ksigns_table = np.frombuffer(cls.ksigns, dtype=np.uint8).reshape(128)

        blocks_3d = blocks.reshape((n_blocks, n_sub, sub_size))

        # IQ2_XXS: each block of 256 has:
        #   d[2] (f16 scale)
        #   qs[64] = 32 x uint16 packed: low 8 bits = grid idx, high 7 bits = sign idx (bits 8-14), high 3 bits = scale (bits 28-30)
        # But actually looking at the dequant: qs is 32 uint16 values (64 bytes)
        # Each uint16: bits 0-7 = grid idx, bits 9-15 = sign idx (7 bits into 128-entry ksigns table)
        # Upper 3 bits of second uint16 (bits 28-30) = scale adjustment

        # For each sub-block of 32 elements, we find 4 groups of 8 grid elements
        sub_sub = blocks_3d.reshape((n_blocks, n_sub, 4, 8))  # (n, 8, 4, 8)

        # Find optimal grid+sign per group of 8
        grid = cls.grid[0, 0]  # (256, 8)
        n_grid = grid.shape[0]

        # Try all grid entries for all sign combinations (2^8 = 256)
        # This is expensive. Simplified approach: find best scale then nearest grid match.
        amax = abs(sub_sub).max(axis=-1, keepdims=True)  # (n, 8, 4, 1)
        d_scale = amax * 4.0  # rough scale factor for IQ2_XXS

        # For each group, find best grid entry by normalized correlation
        sub_sub_n = sub_sub / np.maximum(amax, 1e-10)

        # Dot product with all grid entries
        grid_flat = grid.reshape(1, 1, 1, n_grid, 8)  # (1, 1, 1, 256, 8)
        dots = (sub_sub_n[..., None, :] * grid_flat).sum(axis=-1)  # (n, 8, 4, 256)
        best_grid = dots.argmax(axis=-1)  # (n, 8, 4)

        # Determine signs from the grid match direction
        matched = np.take_along_axis(grid_flat, best_grid[..., None, None].astype(np.intp), axis=-2)  # (n, 8, 4, 1, 8)
        scale_est = amax
        # Signs: compare actual values to matched grid * scale
        # This is simplified - proper sign determination is more involved

        # Pack: qs as uint16 per pair of 8-element groups
        # Each pair of groups -> one uint16: low 8 bits = grid idx, bits 9-15 = sign idx
        qs_out = np.zeros((n_blocks, QK_K // 8), dtype=np.uint16)
        qs_out = best_grid.astype(np.uint16).reshape(n_blocks, -1)

        # d: f16 scale (per 256-element block)
        d_out = d_scale.reshape((n_blocks, 8, 4)).mean(axis=-1).mean(axis=-1)  # (n_blocks,)
        d_bytes = d_out.astype(np.float16).view(np.uint8)

        return np.concatenate([d_bytes.reshape((n_blocks, 2)), qs_out.view(np.uint8).reshape((n_blocks, QK_K // 4))], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.view(np.uint32).reshape(n_blocks, -1, 2)

        db = d * (np.float32(0.5) + (qs[..., 1] >> 28).astype(np.float32)) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))

        # get the sign indices and unpack the bits
        signs = qs[..., 1].reshape((n_blocks, -1, 1)) >> np.array([0, 7, 14, 21], dtype=np.uint32).reshape((1, 1, 4))
        ksigns = np.frombuffer(cls.ksigns, dtype=np.uint8).reshape((1, 1, 1, 128))
        signs = (signs & np.uint32(0x7F)).reshape((n_blocks, -1, 4, 1))
        signs = np.take_along_axis(ksigns, signs, axis=-1)
        signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs[..., 0].copy().view(np.uint8).reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ2_XS(__Quant, qtype=GGMLQuantizationType.IQ2_XS):
    grid_shape = (512, 8)
    grid_map = (0x08, 0x19, 0x2b)
    grid_hex = (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"49005000520055005800610064008000820085008800910094009900a0000101"
        b"04010601090110011201150118011a0121012401400142014501480151015401"
        b"6001680181018401900100020202050208021102140220024102440250025502"
        b"80028a0201040404060409041004120415041804210424044004420445044804"
        b"5104540456046004810484049004000502050505080511051405200541054405"
        b"500561058005010604061006260640064206840600080208050808080a081108"
        b"14082008250841084408500858088008a008aa08010904091009400981098909"
        b"000a200a280a960aa00a01100410061009101010121015101810211024104010"
        b"4210451048105110541060106a10811084109010001102110511081111111411"
        b"2011411144115011801194119611011204120612101240126012001402140514"
        b"0814111414142014411444144914501464148014011504151015401500161416"
        b"49160118041810181218401854188618001905196619511aa91a002002200520"
        b"08200a201120142020204120442050208020a020012104211021402148216521"
        b"002222228022a82201240424102429244024002541255225992501261a26a626"
        b"002808280a28202855288828a22868299029082a202a822a882a8a2a01400440"
        b"0640094010401240154018402140244040404240454048404a40514054406040"
        b"6540814084409040004102410541084111411441204141414441504180418541"
        b"a241014204421042124229424042004402440544084411441444194420444144"
        b"4444504480449444014504451045244540459a4500460a464446504601480448"
        b"1048404845485448624800491149444950496949044a00500250055008501150"
        b"145020502850415044505050805001510451105115514051425100524452aa52"
        b"0154045410542154405460548154a154005508558055885521566856a1560058"
        b"14584158505899581a5940594259855a0160046010604060546062608660a960"
        b"006124624a62926200641664106540654565a46501686a682569066a546a626a"
        b"00800280058008801180148020802a8041804480508080808280a880aa800181"
        b"0481068110814081518159810082208280828282a082a8820184048410841284"
        b"158440846084898400854485a58518866a860088088825885a8880888288a888"
        b"0689228a808a888a968aa88a0190049010904090569084900091229164915692"
        b"89920094059444945094589429959095929541965198a6984999159a609a00a0"
        b"02a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a4"
        b"95a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa"
    )

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        n_blocks = blocks.shape[0]

        groups_8 = blocks.reshape((n_blocks, 32, 8))
        grid = cls.grid[0, 0]

        amax = abs(groups_8).max(axis=-1, keepdims=True)
        grid_flat = grid.reshape(1, 1, grid.shape[0], 8)
        corr = (groups_8[..., None, :] * grid_flat).sum(axis=-1)
        best_grid = corr.argmax(axis=-1).astype(np.uint16)

        d = abs(blocks).max() * np.float32(0.25) / (np.float32(0.5) + 15)
        d = np.full((n_blocks,), d, dtype=np.float32)
        d_bytes = d.astype(np.float16).view(np.uint8).reshape((n_blocks, 2))

        qs_out = best_grid.reshape((n_blocks, QK_K // 8))
        scales_out = np.zeros((n_blocks, QK_K // 32), dtype=np.uint8)

        return np.concatenate([d_bytes, qs_out.view(np.uint8), scales_out], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, scales = np.hsplit(rest, [2 * QK_K // 8])

        d = d.view(np.float16).astype(np.float32)
        qs = qs.view(np.uint16)

        scales = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        scales = (scales & 0x0F).reshape((n_blocks, -1))
        db = d * (np.float32(0.5) + scales) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))

        signs = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).reshape(1, 1, 128)
        signs = np.take_along_axis(signs, (qs >> 9).reshape((n_blocks, -1, 1)), axis=-1)
        signs = signs.reshape((n_blocks, -1, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 2, 8))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, (qs & np.uint16(511)).reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ2_S(__Quant, qtype=GGMLQuantizationType.IQ2_S):
    grid_shape = (1024, 8)
    grid_map = (0x08, 0x19, 0x2b)
    grid_hex = (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"490050005200550058006100640066006900800082008500880091009400a000"
        b"a500aa0001010401060109011001120115011801210124014001420145014801"
        b"510154015601590160016501680181018401900192019501a101a40100020202"
        b"050208021102140220022a02410244024602490250025502800285028a029402"
        b"a202010404040604090410041204150418042104240426042904400442044504"
        b"48044a0451045404560459046004620465048104840486048904900495049804"
        b"a104a40400050205050508050a05110514051605190520052505280541054405"
        b"46054905500552055505580561056405800582058505880591059405a0050106"
        b"0406060609061006150640064506480651065406600681068406900600080208"
        b"050808081108140816081908200825082a084108440846084908500852085508"
        b"580861086408800885089408aa08010904091009120915091809210940094509"
        b"480951095409600981099009000a110a140a220a280a2a0a500a990a01100410"
        b"0610091010101210151018102110241026104010421045104810511054105610"
        b"59106010621065106810811084108610901095109810a110a410001102110511"
        b"08110a1111111411161119112011221125112811411144114611491150115211"
        b"5511581161116411801182118511881191119411011204120912101215122112"
        b"2412401245125112541281128412901200140214051408141114141416141914"
        b"2014251428144114441446144914501452145514581461146414801482148514"
        b"881491149414a014011504150615091510151215151518152115241540154215"
        b"4515481551155415601581158415901500160516081611161416201641164416"
        b"50168016aa160118041806180918101815181818211840184218451848185118"
        b"541860188118841800190219051908191119141920194119441950196919a219"
        b"041a101a401a561a00200220052008201120142016201920202025202a204120"
        b"4420502052205520642080208a209420aa200121042110211221152121214021"
        b"4221452151215421602181218421902100220a22222228222a22442250228822"
        b"8a22a82201240424062409241024152418242124242440244224452448245124"
        b"5424602481248424902400250525082511251425202541254425502566258025"
        b"0126042610264026592600280528112814284128442850288a28aa2801290429"
        b"102995290a2a222a642a882a8a2a014004400640094010401240154018401a40"
        b"21402440264040404240454048404a4051405440564059406040624065408140"
        b"8440904095409840a140a4400041024105410841114114411641194120412241"
        b"2541414144414641494150415241554158416141644180418241854188419141"
        b"9441a04101420442104212421542184224424042454248425142544260428142"
        b"844200440244054408440a441144144416441944204422442544284441444444"
        b"46444944504452445544584461446444804482448544884491449444a0440145"
        b"0445064509451045124515451845214524454045424545454845514554456045"
        b"6a4581458445904500460246054608461146144620464146444650468046a546"
        b"0148044809481048124815481848214824484048424845484848514854486048"
        b"84489048004902490549084911491449204941494449504980499649014a044a"
        b"104a404a00500250055008501150145016501950205022502550285041504450"
        b"4650495050505250555058506150645080508250855088509150945001510451"
        b"0651095110511251155118512151245140514251455148515151545160518151"
        b"8451905100520552085211521452205241524452505269528052015404540654"
        b"0954105412541554185421542454405442544554485451545454605481548454"
        b"9054005502550555085511551455205541554455505580550156045610562656"
        b"405600580258055808581158145820584158445850585a588058015904591059"
        b"4059005a195a855aa85a01600460066010601260156018602160246040604560"
        b"4860516054606060846090600061026105610861116114612061416144615061"
        b"806199610462106240625662a162006405640864116414642064416444645064"
        b"806401650465106540654a656865926500669466016804681068656898680069"
        b"2a69426aa16a0080028005800880118014801980208025804180448050805280"
        b"5580588061808080858091809480018104810981108112811581188121812481"
        b"408142814581488151815481818184819081a981008205820a82118214824182"
        b"4482508201840484068409841084128415841884218440844284458448845184"
        b"5484608481848484908400850285058508851185148520854185448550858085"
        b"8a85018604861086298640860088058811881488418844885088a28801890489"
        b"40896589228a588a5a8a828aa28a019004900990109012901590189024904090"
        b"4290459048905190549060908190849090900091059111911491419144915091"
        b"5a910192049210924092a6920094029405940894119414942094419444945094"
        b"8094969401950495109540959895a19500964696649601980498109826984098"
        b"a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa0"
        b"40a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4"
        b"a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa"
    )

    @staticmethod
    def _pack_u(codes8: np.ndarray) -> np.ndarray:
        """codes8: (..., 8) int, values 0..3 -> (...,) packed 16-bit code."""
        u = np.zeros(codes8.shape[:-1], dtype=np.int64)
        for i in range(8):
            u |= (codes8[..., i].astype(np.int64) << (2 * i))
        return u

    @classmethod
    def _grid_lookup(cls, u: np.ndarray, xval_g: np.ndarray, weight_g: np.ndarray, scale: np.ndarray,
                      pos: np.ndarray, kmap: np.ndarray, neigh_padded: np.ndarray, neigh_count: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """u, scale: (...); xval_g, weight_g: (..., 8). Returns (grid_index,
        L (..., 8) int, on_grid bool (...)) — port of the direct-kmap-hit /
        iq2_find_best_neighbour fallback dispatch."""
        grid_idx = kmap[u]
        on_grid = grid_idx >= 0
        max_n = neigh_padded.shape[1]

        safe_u = np.where(on_grid, 0, u)
        cand = neigh_padded[safe_u]  # (..., max_n)
        cand_cnt = neigh_count[safe_u]  # (...)
        cand_valid = np.arange(max_n) < cand_cnt[..., None]
        safe_cand = np.clip(cand, 0, pos.shape[0] - 1)
        pg = pos[safe_cand]  # (..., max_n, 8)
        diff = scale[..., None, None] * pg - xval_g[..., None, :]
        d2 = np.sum(weight_g[..., None, :] * diff * diff, axis=-1)  # (..., max_n)
        d2 = np.where(cand_valid, d2, np.float32(np.inf))
        best_n = np.argmin(d2, axis=-1)
        neigh_grid_idx = np.take_along_axis(cand, best_n[..., None], axis=-1)[..., 0]

        final_idx = np.where(on_grid, grid_idx, neigh_grid_idx)
        final_pos = pos[np.clip(final_idx, 0, pos.shape[0] - 1)]  # (..., 8)
        L = (final_pos - 1) // 2
        return final_idx, L, on_grid

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        n_blocks = blocks.shape[0]
        n_sub = QK_K // 16  # 16
        kMaxQ = 3
        GROUP_MAX_EPS_IQ2_S = 1e-8

        pos, kmap, neigh_padded, neigh_count = _iq2_build_tables(cls, nwant=1)

        blocks_3d = blocks.reshape((n_blocks, n_sub, 16)).astype(np.float32)

        sum_x2 = np.zeros((n_blocks,), dtype=np.float32)
        for j in range(QK_K):
            sum_x2 += blocks[:, j].astype(np.float32) ** 2
        sigma2 = 2.0 * sum_x2 / QK_K

        weight = 0.25 * sigma2[:, None, None] + blocks_3d * blocks_3d  # (n_blocks, 16, 16)
        waux = np.sqrt(weight)

        xval = np.abs(blocks_3d)
        neg = blocks_3d < 0
        block_signs = np.zeros((n_blocks, n_sub, 2), dtype=np.uint8)
        for k in range(2):
            s = np.zeros((n_blocks, n_sub), dtype=np.uint8)
            for i in range(8):
                s |= np.where(neg[:, :, 8 * k + i], np.uint8(1 << i), np.uint8(0))
            block_signs[:, :, k] = s

        max_sub = xval.max(axis=-1)  # (n_blocks, 16)
        degenerate = max_sub < GROUP_MAX_EPS_IQ2_S
        safe_max = np.where(degenerate, 1.0, max_sub)

        best_scale = np.zeros((n_blocks, n_sub), dtype=np.float32)
        best = np.zeros((n_blocks, n_sub), dtype=np.float32)
        best_L = np.zeros((n_blocks, n_sub, 16), dtype=np.int64)
        best_on_grid = np.zeros((n_blocks, n_sub, 2), dtype=bool)

        for is_ in range(-9, 10):
            id_ = (2 * kMaxQ - 1 + is_ * 0.1) / safe_max  # (n_blocks, 16)
            this_scale = 1.0 / id_
            Laux = np.clip(np_nearest_int(0.5 * (id_[..., None] * xval - 1)), 0, kMaxQ - 1).astype(np.int64)

            L_groups = []
            on_grid_groups = []
            for k in range(2):
                xg = xval[:, :, 8 * k:8 * k + 8]
                wg = waux[:, :, 8 * k:8 * k + 8]
                u = cls._pack_u(Laux[:, :, 8 * k:8 * k + 8])
                _, Lg, og = cls._grid_lookup(u, xg, wg, this_scale, pos, kmap, neigh_padded, neigh_count)
                L_groups.append(Lg)
                on_grid_groups.append(og)
            Laux_final = np.concatenate(L_groups, axis=-1)  # (n_blocks, 16, 16)
            on_grid = np.stack(on_grid_groups, axis=-1)  # (n_blocks, 16, 2)

            q = (2 * Laux_final + 1).astype(np.float32)
            sumqx = np.zeros((n_blocks, n_sub), dtype=np.float32)
            sumq2 = np.zeros((n_blocks, n_sub), dtype=np.float32)
            for i in range(16):
                sumqx += weight[:, :, i] * xval[:, :, i] * q[:, :, i]
                sumq2 += weight[:, :, i] * q[:, :, i] * q[:, :, i]

            improve = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            new_scale = np.where(sumq2 > 0, sumqx / sumq2, 0.0)
            best_L = np.where(improve[..., None], Laux_final, best_L)
            best = np.where(improve, new_scale * sumqx, best)
            best_scale = np.where(improve, new_scale, best_scale)
            best_on_grid = np.where(improve[..., None], on_grid, best_on_grid)

        # Requant pass for groups that never landed on-grid across the search.
        n_not_ongrid = (~best_on_grid).sum(axis=-1)  # (n_blocks, 16)
        need_requant = (n_not_ongrid > 0) & (best_scale > 0)
        id_final = np.where(best_scale != 0, 1.0 / best_scale, 0.0)

        L_final = best_L.copy()
        for k in range(2):
            redo = need_requant & (~best_on_grid[:, :, k])
            xg = xval[:, :, 8 * k:8 * k + 8]
            wg = waux[:, :, 8 * k:8 * k + 8]
            Laux_k = np.clip(np_nearest_int(0.5 * (id_final[..., None] * xg - 1)), 0, kMaxQ - 1).astype(np.int64)
            u = cls._pack_u(Laux_k)
            _, Lg, _ = cls._grid_lookup(u, xg, wg, best_scale, pos, kmap, neigh_padded, neigh_count)
            L_final[:, :, 8 * k:8 * k + 8] = np.where(redo[..., None], Lg, L_final[:, :, 8 * k:8 * k + 8])

        q = (2 * L_final + 1).astype(np.float32)
        sumqx = np.zeros((n_blocks, n_sub), dtype=np.float32)
        sumq2 = np.zeros((n_blocks, n_sub), dtype=np.float32)
        for i in range(16):
            sumqx += weight[:, :, i] * xval[:, :, i] * q[:, :, i]
            sumq2 += weight[:, :, i] * q[:, :, i] * q[:, :, i]
        rescaled = np.where(sumq2 > 0, sumqx / sumq2, best_scale)
        best_scale = np.where(need_requant, rescaled, best_scale)

        flip = best_scale < 0
        best_scale = np.abs(best_scale)
        block_signs = np.where(flip[..., None], (~block_signs) & np.uint8(0xFF), block_signs)

        scales = np.where(degenerate, 0.0, best_scale)
        L_final = np.where(degenerate[..., None], 0, L_final)

        max_scale = scales.max(axis=-1)  # (n_blocks,)
        has_scale = max_scale > 0
        d = np.where(has_scale, max_scale / 31.0, 0.0)
        d_all = (d * 0.9875).astype(np.float16).astype(np.float32)
        id_d = np.where(d != 0, 1.0 / d, 0.0)

        l6 = np.clip(np_nearest_int(0.5 * (id_d[:, None] * scales - 1)), 0, 15).astype(np.uint8)
        l6 = np.where(has_scale[:, None], l6, 0)
        scale_bytes = np.zeros((n_blocks, n_sub // 2), dtype=np.uint8)
        for ib in range(0, n_sub, 2):
            scale_bytes[:, ib // 2] = l6[:, ib] | (l6[:, ib + 1] << 4)

        u_final = np.zeros((n_blocks, n_sub, 2), dtype=np.int64)
        for k in range(2):
            u_final[:, :, k] = cls._pack_u(L_final[:, :, 8 * k:8 * k + 8])
        grid_idx_final = kmap[u_final]  # (n_blocks, 16, 2)
        grid_idx_final = np.where(has_scale[:, None, None], grid_idx_final, 0)

        qs_low = (grid_idx_final & 0xFF).astype(np.uint8).reshape(n_blocks, QK_K // 8)
        qh_bits = ((grid_idx_final >> 8) & 0x03).astype(np.uint8)  # (n_blocks, 16, 2)
        qh_bits_flat = qh_bits.reshape(n_blocks, QK_K // 8)  # index i8 = 2*ib+k
        qh_out = np.zeros((n_blocks, QK_K // 32), dtype=np.uint8)
        for i8 in range(QK_K // 8):
            qh_out[:, i8 // 4] |= (qh_bits_flat[:, i8] << (2 * (i8 % 4)))

        signs_out = np.where(has_scale[:, None, None], block_signs, 0).reshape(n_blocks, QK_K // 8).astype(np.uint8)

        d_bytes = d_all.astype(np.float16).view(np.uint8)

        return np.concatenate([
            d_bytes.reshape((n_blocks, 2)),
            qs_low,
            signs_out,
            qh_out,
            scale_bytes,
        ], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, rest = np.hsplit(rest, [QK_K // 8])
        signs, rest = np.hsplit(rest, [QK_K // 8])
        qh, scales = np.hsplit(rest, [QK_K // 32])

        d = d.view(np.float16).astype(np.float32)

        scales = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        scales = (scales & 0x0F).reshape((n_blocks, -1))
        db = d * (np.float32(0.5) + scales) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))

        signs = signs.reshape((n_blocks, -1, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 2, 8))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4))
        qs = qs.astype(np.uint16) | ((qh & 0x03).astype(np.uint16) << 8).reshape((n_blocks, -1))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ3_XXS(__Quant, qtype=GGMLQuantizationType.IQ3_XXS):
    grid_shape = (256, 4)
    grid_map = (0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e)
    grid_hex = (
        b"0000020004001100130017002000220031004200730075000101030110011201"
        b"2101250130013201410154017001000202020402110220022202310233023702"
        b"5102570275020103070310031203250370031304370444045704730475040105"
        b"0705320552053506640610071407160743076107011003101010121021102310"
        b"3010321034104710501000110211111120112211011203121012121221123012"
        b"7212001302132013311346136613011405145014201524154615711505162217"
        b"4017002002201120132020202220262031204220012103210521102112212121"
        b"3021632167217021002202221122172220222222372240225522012310231423"
        b"7023742335245324032527254125742501270327162745270130103012302130"
        b"2330503065307230003102312031313144314631013203321032253252327232"
        b"1133333330344734723400350635223555351436363663363337603704401740"
        b"3540374053405740744120423742404260426642074345430444514464442545"
        b"4345704505471047124730471250415070500051065126515551145232527252"
        b"0253535310542354275472540255315550562457425724604460466064602161"
        b"6161176264623063366344640565526533660367216703700570077010703270"
        b"5270267140711272457252720073157333736073217441740075027524753076"
    )

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        n_blocks = blocks.shape[0]
        n_ib = QK_K // 32  # 8
        kMaxQ = 8
        GROUP_MAX_EPS_IQ3_XXS = 1e-8

        pos, kmap, neigh_padded, neigh_count = _iq3_build_tables(cls, nwant=2)

        blocks_3d = blocks.reshape((n_blocks, n_ib, 32)).astype(np.float32)

        weight = blocks_3d * blocks_3d  # (n_blocks, n_ib, 32) -- no imatrix branch
        waux = np.sqrt(weight)

        xval = np.abs(blocks_3d).copy()
        neg = blocks_3d < 0
        block_signs = np.zeros((n_blocks, n_ib, 4), dtype=np.uint8)
        for ks in range(4):
            s = np.zeros((n_blocks, n_ib), dtype=np.uint8)
            nflip = np.zeros((n_blocks, n_ib), dtype=np.int32)
            for i in range(8):
                s |= np.where(neg[:, :, 8 * ks + i], np.uint8(1 << i), np.uint8(0))
                nflip += neg[:, :, 8 * ks + i].astype(np.int32)
            odd = (nflip % 2) == 1
            ax = np.stack([weight[:, :, 8 * ks + i] * blocks_3d[:, :, 8 * ks + i] ** 2 for i in range(8)], axis=-1)
            imin = np.argmin(ax, axis=-1)  # (n_blocks, n_ib)
            for i in range(8):
                sel = odd & (imin == i)
                xval[:, :, 8 * ks + i] = np.where(sel, -xval[:, :, 8 * ks + i], xval[:, :, 8 * ks + i])
                s = np.where(sel, s ^ np.uint8(1 << i), s)
            block_signs[:, :, ks] = s & 0x7F

        max_sub = xval.max(axis=-1)  # (n_blocks, n_ib)
        degenerate = max_sub < GROUP_MAX_EPS_IQ3_XXS
        safe_max = np.where(degenerate, 1.0, max_sub)

        best_scale = np.zeros((n_blocks, n_ib), dtype=np.float32)
        best = np.zeros((n_blocks, n_ib), dtype=np.float32)
        best_L = np.zeros((n_blocks, n_ib, 32), dtype=np.int64)
        best_on_grid = np.ones((n_blocks, n_ib, 8), dtype=bool)

        for is_ in range(-15, 16):
            id_ = (2 * kMaxQ - 1 + is_ * 0.2) / safe_max
            this_scale = 1.0 / id_
            Laux = np.clip(np_nearest_int(0.5 * (id_[..., None] * xval - 1)), 0, kMaxQ - 1).astype(np.int64)

            L_groups = []
            on_grid_groups = []
            for kg in range(8):
                xg = xval[:, :, 4 * kg:4 * kg + 4]
                wg = waux[:, :, 4 * kg:4 * kg + 4]
                u = _iq3_pack_u(Laux[:, :, 4 * kg:4 * kg + 4])
                _, Lg, og = _iq3_grid_lookup(u, xg, wg, this_scale, pos, kmap, neigh_padded, neigh_count)
                L_groups.append(Lg)
                on_grid_groups.append(og)
            Laux_final = np.concatenate(L_groups, axis=-1)  # (n_blocks, n_ib, 32)
            on_grid = np.stack(on_grid_groups, axis=-1)  # (n_blocks, n_ib, 8)

            q = (2 * Laux_final + 1).astype(np.float32)
            sumqx = np.zeros((n_blocks, n_ib), dtype=np.float32)
            sumq2 = np.zeros((n_blocks, n_ib), dtype=np.float32)
            for i in range(32):
                sumqx += weight[:, :, i] * xval[:, :, i] * q[:, :, i]
                sumq2 += weight[:, :, i] * q[:, :, i] * q[:, :, i]

            improve = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            new_scale = np.where(sumq2 > 0, sumqx / sumq2, 0.0)
            best_L = np.where(improve[..., None], Laux_final, best_L)
            best = np.where(improve, new_scale * sumqx, best)
            best_scale = np.where(improve, new_scale, best_scale)
            best_on_grid = np.where(improve[..., None], on_grid, best_on_grid)

        n_not_ongrid = (~best_on_grid).sum(axis=-1)
        need_requant = (n_not_ongrid > 0) & (best_scale > 0)
        id_final = np.where(best_scale != 0, 1.0 / best_scale, 0.0)

        L_final = best_L.copy()
        for kg in range(8):
            redo = need_requant & (~best_on_grid[:, :, kg])
            xg = xval[:, :, 4 * kg:4 * kg + 4]
            wg = waux[:, :, 4 * kg:4 * kg + 4]
            Laux_k = np.clip(np_nearest_int(0.5 * (id_final[..., None] * xg - 1)), 0, kMaxQ - 1).astype(np.int64)
            u = _iq3_pack_u(Laux_k)
            _, Lg, _ = _iq3_grid_lookup(u, xg, wg, best_scale, pos, kmap, neigh_padded, neigh_count)
            L_final[:, :, 4 * kg:4 * kg + 4] = np.where(redo[..., None], Lg, L_final[:, :, 4 * kg:4 * kg + 4])

        q = (2 * L_final + 1).astype(np.float32)
        sumqx = np.zeros((n_blocks, n_ib), dtype=np.float32)
        sumq2 = np.zeros((n_blocks, n_ib), dtype=np.float32)
        for i in range(32):
            sumqx += weight[:, :, i] * xval[:, :, i] * q[:, :, i]
            sumq2 += weight[:, :, i] * q[:, :, i] * q[:, :, i]
        rescaled = np.where(sumq2 > 0, sumqx / sumq2, best_scale)
        best_scale = np.where(need_requant, rescaled, best_scale)

        flip = best_scale < 0
        best_scale = np.abs(best_scale)
        block_signs = np.where(flip[..., None], (~block_signs) & np.uint8(0x7F), block_signs)

        scales = np.where(degenerate, 0.0, best_scale)
        L_final = np.where(degenerate[..., None], 0, L_final)

        max_scale = scales.max(axis=-1)  # (n_blocks,)
        has_scale = max_scale > 0
        d = np.where(has_scale, max_scale / 31.0, 0.0)
        d_all = (d * 1.0125).astype(np.float16).astype(np.float32)
        id_d = np.where(d != 0, 1.0 / d, 0.0)

        l4 = np.clip(np_nearest_int(0.5 * (id_d[:, None] * scales - 1)), 0, 15).astype(np.uint32)
        l4 = np.where(has_scale[:, None], l4, 0)

        u_final = np.zeros((n_blocks, n_ib, 8), dtype=np.int64)
        for kg in range(8):
            u_final[:, :, kg] = _iq3_pack_u(L_final[:, :, 4 * kg:4 * kg + 4])
        grid_idx_final = kmap[u_final]  # (n_blocks, n_ib, 8)
        grid_idx_final = np.where(has_scale[:, None, None], grid_idx_final, 0)
        qs_out = grid_idx_final.astype(np.uint8).reshape(n_blocks, QK_K // 4)

        block_signs_masked = np.where(has_scale[:, None, None], block_signs, 0).astype(np.uint32)
        scales_and_signs = (block_signs_masked[:, :, 0] | (block_signs_masked[:, :, 1] << 7) |
                             (block_signs_masked[:, :, 2] << 14) | (block_signs_masked[:, :, 3] << 21) |
                             (l4 << 28)).astype(np.uint32)
        scas_bytes = scales_and_signs.view(np.uint8).reshape(n_blocks, n_ib * 4)

        d_bytes = d_all.astype(np.float16).view(np.uint8)

        return np.concatenate([
            d_bytes.reshape((n_blocks, 2)),
            qs_out,
            scas_bytes,
        ], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, scales = np.hsplit(rest, [QK_K // 4])

        d = d.view(np.float16).astype(np.float32)
        scales = scales.view(np.uint32)

        db = d * (np.float32(0.5) + (scales >> 28).astype(np.float32)) * np.float32(0.5)
        db = db.reshape((n_blocks, -1, 1, 1))

        signs = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 7, 14, 21], dtype=np.uint32).reshape((1, 1, 4))
        ksigns = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).reshape((1, 1, 1, 128))
        signs = (signs & np.uint32(0x7F)).reshape((n_blocks, -1, 4, 1))
        signs = np.take_along_axis(ksigns, signs, axis=-1)
        signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ3_S(__Quant, qtype=GGMLQuantizationType.IQ3_S):
    grid_shape = (512, 4)
    grid_map = (0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f)
    grid_hex = (
        b"0000010002000500070010001100120014001600200021002500330040004200"
        b"4500470051005300600062007100740077000001010102010401100111011501"
        b"2001230127013101350144016101650172010002010205020702100213021602"
        b"2102250230023402420245024702510253027002730203031103150320032203"
        b"3103330336034403500352036703710375030004130417042104240432044004"
        b"4304510470040205040520052205260533054105450547056605730506061106"
        b"1306310652067106000702070407200722072607330750075407001001100210"
        b"0410101011101310151017102010221031103410361054105610611072100011"
        b"0111031106111011141121113011331141115011521170117611001212121512"
        b"1712201224123212401243125512601272120113041307131013131321132713"
        b"3013341341136213701303140514121414143114331442144614501454140115"
        b"1015131521153015321551152016241627164416461601170317101712172117"
        b"3517411762177017002001200320052007201020122014201620212023202720"
        b"3020322041204320452050205220672070207320752000210221102113211721"
        b"2221252131213421422151210122042207222122232230223722412253225722"
        b"7122742200230223052311232223242331233323422350236623012407242024"
        b"2324322435244124722475240425112522253725402553257025002602260726"
        b"2126552661260527112726273027432750270230113013301530173022303130"
        b"3330353042304430473051306330713001310331053114312131233140316031"
        b"7231763100321232203232323432503201331033143321332333273330334133"
        b"4333473355337333033411341634223431345234603464340135103512352535"
        b"3235443556357335163641360137033720372237353700400440124020402440"
        b"2740324041405040704002410741114113412241304135414341514155410142"
        b"0342104215422142334240425742624270420443114313432043224331433543"
        b"0044024424443744404471440545074521456245134634466046104715473047"
        b"4347514702501050145022504050445047505250665074500151035105511251"
        b"2151325172510052115223523052365253520253075310532753445351536553"
        b"7353015404542054325446541255265551555355425602570457225711601360"
        b"1560316033606060006120612761646112623462426255626262706200631463"
        b"2163406325644364626400650365346560650566406611671367007004700770"
        b"2070227036704070547062700271117124714371457101720472107216722172"
        b"3072517202733273357353730174057413742074507422754275027631760077"
    )

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        n_blocks = blocks.shape[0]
        n_ib = QK_K // 32  # 8
        kMaxQ = 8

        pos, kmap, neigh_padded, neigh_count = _iq3_build_tables(cls, nwant=3)

        blocks_3d = blocks.reshape((n_blocks, n_ib, 32)).astype(np.float32)

        weight = blocks_3d * blocks_3d  # (n_blocks, n_ib, 32) -- no imatrix branch
        waux = np.sqrt(weight)

        xval = np.abs(blocks_3d)
        neg = blocks_3d < 0
        block_signs = np.zeros((n_blocks, n_ib, 4), dtype=np.uint8)
        for ks in range(4):
            s = np.zeros((n_blocks, n_ib), dtype=np.uint8)
            for i in range(8):
                s |= np.where(neg[:, :, 8 * ks + i], np.uint8(1 << i), np.uint8(0))
            block_signs[:, :, ks] = s

        max_sub = xval.max(axis=-1)  # (n_blocks, n_ib)
        degenerate = max_sub == 0
        safe_max = np.where(degenerate, 1.0, max_sub)

        best_scale = np.zeros((n_blocks, n_ib), dtype=np.float32)
        best = np.zeros((n_blocks, n_ib), dtype=np.float32)
        best_L = np.zeros((n_blocks, n_ib, 32), dtype=np.int64)
        best_on_grid = np.zeros((n_blocks, n_ib, 8), dtype=bool)

        for is_ in range(-9, 10):
            id_ = (2 * kMaxQ - 1 + is_ * 0.2) / safe_max
            this_scale = 1.0 / id_
            Laux = np.clip(np_nearest_int(0.5 * (id_[..., None] * xval - 1)), 0, kMaxQ - 1).astype(np.int64)

            L_groups = []
            on_grid_groups = []
            for kg in range(8):
                xg = xval[:, :, 4 * kg:4 * kg + 4]
                wg = waux[:, :, 4 * kg:4 * kg + 4]
                u = _iq3_pack_u(Laux[:, :, 4 * kg:4 * kg + 4])
                _, Lg, og = _iq3_grid_lookup(u, xg, wg, this_scale, pos, kmap, neigh_padded, neigh_count)
                L_groups.append(Lg)
                on_grid_groups.append(og)
            Laux_final = np.concatenate(L_groups, axis=-1)  # (n_blocks, n_ib, 32)
            on_grid = np.stack(on_grid_groups, axis=-1)  # (n_blocks, n_ib, 8)

            q = (2 * Laux_final + 1).astype(np.float32)
            sumqx = np.zeros((n_blocks, n_ib), dtype=np.float32)
            sumq2 = np.zeros((n_blocks, n_ib), dtype=np.float32)
            for i in range(32):
                sumqx += weight[:, :, i] * xval[:, :, i] * q[:, :, i]
                sumq2 += weight[:, :, i] * q[:, :, i] * q[:, :, i]

            improve = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            new_scale = np.where(sumq2 > 0, sumqx / sumq2, 0.0)
            best_L = np.where(improve[..., None], Laux_final, best_L)
            best = np.where(improve, new_scale * sumqx, best)
            best_scale = np.where(improve, new_scale, best_scale)
            best_on_grid = np.where(improve[..., None], on_grid, best_on_grid)

        n_not_ongrid = (~best_on_grid).sum(axis=-1)
        need_requant = (n_not_ongrid > 0) & (best_scale > 0)
        id_final = np.where(best_scale != 0, 1.0 / best_scale, 0.0)

        # unlike IQ2_S/IQ3_XXS, IQ3_S redoes ALL 8 groups (not just off-grid ones)
        # whenever any group needs it, matching the (commented-out) C condition.
        L_final = best_L.copy()
        for kg in range(8):
            xg = xval[:, :, 4 * kg:4 * kg + 4]
            wg = waux[:, :, 4 * kg:4 * kg + 4]
            Laux_k = np.clip(np_nearest_int(0.5 * (id_final[..., None] * xg - 1)), 0, kMaxQ - 1).astype(np.int64)
            u = _iq3_pack_u(Laux_k)
            _, Lg, _ = _iq3_grid_lookup(u, xg, wg, best_scale, pos, kmap, neigh_padded, neigh_count)
            L_final[:, :, 4 * kg:4 * kg + 4] = np.where(need_requant[..., None], Lg, L_final[:, :, 4 * kg:4 * kg + 4])

        q = (2 * L_final + 1).astype(np.float32)
        sumqx = np.zeros((n_blocks, n_ib), dtype=np.float32)
        sumq2 = np.zeros((n_blocks, n_ib), dtype=np.float32)
        for i in range(32):
            sumqx += weight[:, :, i] * xval[:, :, i] * q[:, :, i]
            sumq2 += weight[:, :, i] * q[:, :, i] * q[:, :, i]
        rescaled = np.where(sumq2 > 0, sumqx / sumq2, best_scale)
        best_scale = np.where(need_requant, rescaled, best_scale)

        flip = best_scale < 0
        best_scale = np.abs(best_scale)
        block_signs = np.where(flip[..., None], (~block_signs) & np.uint8(0xFF), block_signs)

        scales = np.where(degenerate, 0.0, best_scale)
        L_final = np.where(degenerate[..., None], 0, L_final)

        max_scale = scales.max(axis=-1)  # (n_blocks,)
        has_scale = max_scale > 0
        d = np.where(has_scale, max_scale / 31.0, 0.0)
        d_all = (d * 1.033).astype(np.float16).astype(np.float32)
        id_d = np.where(d != 0, 1.0 / d, 0.0)

        l4 = np.clip(np_nearest_int(0.5 * (id_d[:, None] * scales - 1)), 0, 15).astype(np.uint8)
        l4 = np.where(has_scale[:, None], l4, 0)
        scale_bytes = np.zeros((n_blocks, n_ib // 2), dtype=np.uint8)
        for ib in range(0, n_ib, 2):
            scale_bytes[:, ib // 2] = l4[:, ib] | (l4[:, ib + 1] << 4)

        u_final = np.zeros((n_blocks, n_ib, 8), dtype=np.int64)
        for kg in range(8):
            u_final[:, :, kg] = _iq3_pack_u(L_final[:, :, 4 * kg:4 * kg + 4])
        grid_idx_final = kmap[u_final]  # (n_blocks, n_ib, 8)
        grid_idx_final = np.where(has_scale[:, None, None], grid_idx_final, 0)

        qs_out = (grid_idx_final & 0xFF).astype(np.uint8).reshape(n_blocks, QK_K // 4)
        qh_bits = ((grid_idx_final >> 8) & 0x01).astype(np.uint8)  # (n_blocks, n_ib, 8)
        qh_bits_flat = qh_bits.reshape(n_blocks, n_ib * 8)  # global g = ib*8+kg
        qh_out = np.zeros((n_blocks, QK_K // 32), dtype=np.uint8)
        for g in range(n_ib * 8):
            qh_out[:, g // 8] |= (qh_bits_flat[:, g] << (g % 8))

        signs_out = np.where(has_scale[:, None, None], block_signs, 0).reshape(n_blocks, QK_K // 8).astype(np.uint8)

        d_bytes = d_all.astype(np.float16).view(np.uint8)

        return np.concatenate([
            d_bytes.reshape((n_blocks, 2)),
            qs_out,
            qh_out,
            signs_out,
            scale_bytes,
        ], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        qh, rest = np.hsplit(rest, [QK_K // 32])
        signs, scales = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32)

        scales = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        scales = (scales & 0x0F).reshape((n_blocks, -1))
        db = d * (1 + 2 * scales)
        db = db.reshape((n_blocks, -1, 1, 1))

        signs = signs.reshape((n_blocks, -1, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([i for i in range(8)], dtype=np.uint8)
        qh = (qh & 0x01).astype(np.uint16).reshape((n_blocks, -1))
        qs = qs.astype(np.uint16) | (qh << 8)

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ1_S(__Quant, qtype=GGMLQuantizationType.IQ1_S):
    # iq1s_grid, with each byte packed into 2 bits
    # -1, 0, 1 <=> 0, 1, 2
    grid_shape = (2048, 8)
    grid_map = (-1, 0, 1)
    grid_hex = (
        b"00000200050008000a00110015002000220028002a0045005100540056006500"
        b"8000820088008a009500a000a200a800aa000401050111011401160119011a01"
        b"2501410146014901520155015a0161016401660168018501910194019601a501"
        b"0002020208020a0215022002220228022a024502510259026402690280028202"
        b"88028a02910295029902a002a202a802aa021104140416042504410449045504"
        b"5a046404650491049904a5040105040505050605150518051a05290540054505"
        b"4a0550055105540555055605590560056205650568056a058105910595059805"
        b"9a05a105a405a505a605a9051406190641064406500652065506580660066106"
        b"6606690685069106940699060008020808080a0815082008220828082a084508"
        b"5108560865088008820888088a089508a008a208a808aa080509110914091909"
        b"2409250941095009510955096109640969099109940996099909a509000a020a"
        b"080a0a0a150a200a220a280a2a0a450a510a590a610a650a800a820a850a880a"
        b"8a0a950aa00aa20aa80aaa0a1010111014101910241025104110441050105510"
        b"58106110641065106910911094109610a110a510011104110611091110111211"
        b"1511181121112411291145114a11501151115211541155115611591160116511"
        b"841192119511a111a41111121412161225124012461249125212551258125a12"
        b"641266128512911294129612a512011406140914141415141814191421142614"
        b"41144514461448144a1451145414551456145914621465146814841489149014"
        b"94149514981499149a14a114a414a514a914021505150a151115141515151615"
        b"191520152215251528152a154115441545154615511552155415551556155915"
        b"5a1561156415651566156915801582158415851588158a159015911594159515"
        b"961599159a15a015a215a51501160416051606161516161618161a1621162616"
        b"401642164416451648164a165116551656165816591661166416651668166916"
        b"6a1686168a1692169516a416a916111816182518411844184618491850185518"
        b"58185a1860186118641866186918851891189418a5181019121915191a192119"
        b"25194219441945194819511954195519561959195a19601965196a1989199119"
        b"921995199819a119a619a919091a161a241a261a441a461a491a501a521a551a"
        b"581a611a661a691a851a911a961a9a1a0020022008200a201520202022202520"
        b"28202a20452051205920612065208020822088208a209520a020a220a520a820"
        b"aa2005211121142119212521422144214921552158215a216121642165216621"
        b"8521902196219921a521012208220a22112215222022222228222a2245225122"
        b"562259226522812288228a2291229522a022a222a822aa220524142416241924"
        b"252444244524462449245224552458245a2466248524912494249924a124a524"
        b"0925152521252925402545254825512554255525592562256525682589259025"
        b"9425952598259a25a125a425a625a92505261026122619262526412649265526"
        b"6026612669268426862690269a260028022808280a2815282028222828282a28"
        b"45285128542865288028822888288a28a028a228a828aa280929112914291929"
        b"2529462949295229552961296429662969298529902996299929a429a529002a"
        b"022a082a0a2a202a222a282a2a2a452a512a562a592a652a802a822a882a8a2a"
        b"952aa02aa22aa82aaa2a054011401640254049405240554058405a4061406440"
        b"664094409940a140a6400041014104410641094112411541164118411a412141"
        b"26412941454148414a41514154415541564159415a41654168416a4181418441"
        b"8641904192419541a041a141a241054211421442164225424142524255425a42"
        b"6442694289429442a5420144154419442944454448444a445144544455445644"
        b"61446244654468446a44814486448944904492449544a044a144a94401450245"
        b"05450a4511451445154516451945204525452a45414544454545464549455045"
        b"5145544555455645584559456145644565456645694582458445854588459145"
        b"94459545964599459a45a545a845aa450146054609461446154618461a462146"
        b"2446294640464246454648465046514652465546564659466246654668468146"
        b"85468a4694469546a146a446a6460548114815481a4825484248494850485548"
        b"5848614864486648694885489148944896489948a5480149054906490a491049"
        b"144915491849214924492649404945494a495149524954495549564959496049"
        b"6249654966496a49864989499249954996499849a149a449a649a949164a444a"
        b"464a494a554a584a5a4a644a694a944aa54a0150045005500650095012501550"
        b"1a50215024502950405045504850515054505550565059506550685086508950"
        b"95509850a050a150a650a9500551085109510a51115114511551165118511951"
        b"20512551265128512a5141514451455146514951505151515251545155515651"
        b"585159515a51615164516551665169518251855191519451955196519951a051"
        b"a551aa5101520652125215521a5221522452425245524a525152545255525652"
        b"595262526552855290529252955299529a52a452045405541154145415541654"
        b"185419542154255428542a54415444544554465449544a545054515454545554"
        b"5654585459545a54615462546454655466546954805488548a54915494549554"
        b"96549954a154a454a554aa540155025504550555065509551055115512551455"
        b"1555165519551a55215524552555265529554055415542554455455546554855"
        b"4955505551555255545555555655585559555a55605561556455655566556855"
        b"69556a5581558455855589558a559055915594559555965598559955a155a455"
        b"a555a655a9550056015602560456065608560956115614561556185619562056"
        b"2156225624562556265628562956415645564656485649564a56505651565256"
        b"545655565656585659565a566156645665566956825685568656885689568a56"
        b"915695569a56a256a556a656a856a95604580558065809581058155818582158"
        b"2a58455848584a58515854585558565858585958605862586458655882588958"
        b"9058925895589858a158a9580159025905590a59115914591559165919592559"
        b"41594459455946594959505951595259545955595659585959595a5961596459"
        b"655966596959815985598959915994599559965998599959a559045a085a155a"
        b"1a5a205a255a265a295a455a485a495a515a555a565a585a595a625a655a685a"
        b"6a5a815a8a5a925a955a965a985a9a5aa15a0560146016601960256044605060"
        b"5560566058605a60616064606660696081609660a56001610461066109611261"
        b"15612161226126612961456149615161556156615961656166616a6184618a61"
        b"92619561a161a661a96111621662196240624162466255625662586260628562"
        b"91629662a56211641264156416641a6421642664296440644264456448644a64"
        b"516454645564566459645a646064626465648464856489649064926494649564"
        b"966498649a64a164a464a964056508650a651165156516651965446545654665"
        b"496550655165546555655665596561656465656566656965866589658a659165"
        b"9565966599659a65a265a565a665a86502660966156620662666286629664066"
        b"456648664a66516654665566566658665a666066656668668066826685668a66"
        b"9466966698669966a066a466a666aa661668196825684168526855685a686168"
        b"6968856891689868a66801690469106915692169246926692969406941694569"
        b"4669486951695469556956695969606965696a69826984698a699569a169a469"
        b"a569a969116a166a186a416a446a496a506a556a586a5a6a646a656a696a866a"
        b"946a986a9a6aa66a0080028008800a802080228028802a804580508051805480"
        b"5680598065808080828088808a809580a080a280a880aa800581118114811681"
        b"1981258141814481498150815281558156815881598164816681698185818981"
        b"948196819981a5810082028208820a8215822082228228822a82518254825982"
        b"65828082828288828a829582a082a282a882aa82148419844184448451845584"
        b"5a846184648469849484998401850985128515851a8526852985408541854585"
        b"4885518554855585568559855a856585668568856a8581858485868589859085"
        b"928595859885a68511861686198625864186448649864a865086558659865a86"
        b"618666866a86858691869a86a4860088028808880a8815882088228828882a88"
        b"41884588518854885988658869888088828888888a889588a088a288a888aa88"
        b"05890689118914891689258941894489468949895089528955895a8961896489"
        b"858996899989a589008a028a088a0a8a158a208a228a288a2a8a458a518a548a"
        b"568a808a828a888a8a8a958aa08aa28aa88aaa8a059011901690189019902590"
        b"419046904990559058905a9069906a9085909190949096909990a59001910491"
        b"069109911091159118911a912191249126912991409145915091519154915591"
        b"569159916291659184918691929195919891a191a491a691a991059211921492"
        b"19922592449246924992509252925592589266926992859294929692a9920194"
        b"04940694109415941894269440944a9451945494559456945894599460946194"
        b"62946594849486949294949495949894a194a9940095059508950a9510951195"
        b"14951595169519952195259529952a9541954495459546954995509551955295"
        b"549555955695589559955a956195649565956695699581958595889591959295"
        b"94959595969599959a95a095a295a595a895aa95019604961096159619962096"
        b"2696299645964896499651965296559656965996659668968296849689968a96"
        b"929694969596a496a696a9960598169819982598419846985098529855985698"
        b"5a98649865988598919896989998a59804990699099910991299159918991a99"
        b"209921992499269940994299459948994a995199549955995699599962996599"
        b"66996a99819984999099929995999a99a199a699059a159a259a449a469a499a"
        b"509a559a589a619a859a919a949a959a969a00a002a008a00aa015a020a022a0"
        b"28a02aa045a051a054a056a059a080a082a088a08aa095a0a0a0a2a0a8a0aaa0"
        b"05a109a111a114a116a119a11aa146a149a151a155a158a15aa161a164a185a1"
        b"90a192a196a199a102a208a20aa210a219a222a228a22aa245a251a256a259a2"
        b"65a280a282a288a28aa295a2a0a2a2a2a8a2aaa219a425a441a444a450a454a4"
        b"55a458a45aa461a465a466a468a469a485a406a509a510a512a515a518a526a5"
        b"29a542a545a551a554a555a556a559a565a56aa581a584a585a586a589a592a5"
        b"95a598a505a611a616a61aa621a625a644a646a64aa652a655a656a658a660a6"
        b"62a686a690a695a696a699a6a1a6a4a6a6a600a802a808a80aa820a822a828a8"
        b"2aa851a854a856a859a880a882a888a88aa895a8a0a8a2a8a8a8aaa805a914a9"
        b"19a921a925a941a950a955a95aa961a966a969a990a996a900aa02aa08aa0aaa"
        b"20aa22aa28aa2aaa51aa54aa56aa80aa82aa88aa8aaa95aaa0aaa2aaa8aaaaaa"
    )

    delta = np.float32(0.125)

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        n_blocks = blocks.shape[0]

        # IQ1_S: 8-element groups, {-1, 0, 1} grid values
        # Each group finds the nearest grid entry
        groups = blocks.reshape((n_blocks, QK_K // 8, 8))
        grid = cls.grid[0, 0]  # (2048, 8)

        amax = abs(groups).max(axis=-1, keepdims=True)
        grid_flat = grid.reshape(1, 1, grid.shape[0], 8)
        corr = (groups[..., None, :] * grid_flat).sum(axis=-1)
        best_grid = corr.argmax(axis=-1).astype(np.uint16)

        # d: single f16 per block
        d = abs(blocks).max() * np.float32(0.125)
        d = np.full((n_blocks,), d, dtype=np.float32)
        d_bytes = d.astype(np.float16).view(np.uint8)

        # qs[QK_K//8]
        qs_out = (best_grid & 0xFF).astype(np.uint8).reshape((n_blocks, QK_K // 8))
        qh_raw = ((best_grid >> 8) & 0x07).astype(np.uint16).reshape((n_blocks, -1, 4))
        qh_out = (qh_raw[..., 0] | (qh_raw[..., 1] << 3) | (qh_raw[..., 2] << 6) | (qh_raw[..., 3] << 9)).astype(np.uint16)

        return np.concatenate([d_bytes.reshape((n_blocks, 2)), qs_out, qh_out.view(np.uint8)], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, qh = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint16)

        dl = d * (2 * ((qh >> 12) & 7) + 1)
        dl = dl.reshape((n_blocks, -1, 1, 1))
        delta = np.where((qh & np.uint16(0x8000)) == 0, cls.delta, -cls.delta)
        delta = delta.reshape((n_blocks, -1, 1, 1))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape((1, 1, 4))
        qs = qs.astype(np.uint16) | ((qh & 7) << 8).reshape((n_blocks, -1))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (dl * (grid + delta)).reshape((n_blocks, -1))


class IQ1_M(__Quant, qtype=GGMLQuantizationType.IQ1_M):
    grid_shape = IQ1_S.grid_shape
    grid_map = IQ1_S.grid_map
    grid_hex = IQ1_S.grid_hex

    delta = IQ1_S.delta

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        n_blocks = blocks.shape[0]
        n_ib = QK_K // 16  # 16
        block_size = 16
        GROUP_MAX_EPS_IQ1_M = 1e-7
        delta = np.float32(cls.delta)

        pos, kmap, neigh_padded, neigh_count = _iq2_build_tables(cls, nwant=3)

        blocks_3d = blocks.reshape((n_blocks, n_ib, block_size)).astype(np.float32)
        weight = blocks_3d * blocks_3d  # (n_blocks, n_ib, 16) -- no imatrix branch
        xb = blocks_3d

        x_p = np.array([-1 + delta, delta, 1 + delta], dtype=np.float32)
        x_m = np.array([-1 - delta, -delta, 1 - delta], dtype=np.float32)

        max_sub = np.abs(xb).max(axis=-1)  # (n_blocks, n_ib)
        degenerate = max_sub < GROUP_MAX_EPS_IQ1_M

        # Exact SSD search over the two split points (i1, i2) of the sorted
        # values, with 4 candidate sign assignments (k) for the two halves
        # (elements < 8 / >= 8). Uses prefix sums over the sorted order to
        # avoid an O(block_size^3) search per subblock.
        order = np.argsort(xb, axis=-1)  # ascending; ties are measure-zero for real data
        xb_sorted = np.take_along_axis(xb, order, axis=-1)
        w_sorted = np.take_along_axis(weight, order, axis=-1)
        half0 = order < (block_size // 2)  # True: original index in first half

        HALF0_PROFILE = np.array([True, True, False, False])
        HALF1_PROFILE = np.array([True, False, True, False])
        use_p_sorted = np.where(half0[..., None], HALF0_PROFILE, HALF1_PROFILE)  # (n_blocks, n_ib, 16, 4)

        CumA = np.zeros((n_blocks, n_ib, 3, 4, block_size + 1), dtype=np.float32)
        CumB = np.zeros((n_blocks, n_ib, 3, 4, block_size + 1), dtype=np.float32)
        for j in range(3):
            for k in range(4):
                val = np.where(use_p_sorted[..., k], x_p[j], x_m[j])  # (n_blocks, n_ib, 16)
                aj = w_sorted * val * xb_sorted
                bj = w_sorted * val * val
                CumA[:, :, j, k, 1:] = np.cumsum(aj, axis=-1)
                CumB[:, :, j, k, 1:] = np.cumsum(bj, axis=-1)

        best_score = np.full((n_blocks, n_ib), -np.inf, dtype=np.float32)
        best_scale = np.zeros((n_blocks, n_ib), dtype=np.float32)
        besti1 = np.full((n_blocks, n_ib), -1, dtype=np.int32)
        besti2 = np.full((n_blocks, n_ib), -1, dtype=np.int32)
        best_k = np.zeros((n_blocks, n_ib), dtype=np.int32)

        for i1 in range(block_size + 1):
            for i2 in range(i1, block_size + 1):
                sumqx4 = (CumA[:, :, 0, :, i1] +
                          (CumA[:, :, 1, :, i2] - CumA[:, :, 1, :, i1]) +
                          (CumA[:, :, 2, :, block_size] - CumA[:, :, 2, :, i2]))  # (n_blocks, n_ib, 4)
                sumq24 = (CumB[:, :, 0, :, i1] +
                          (CumB[:, :, 1, :, i2] - CumB[:, :, 1, :, i1]) +
                          (CumB[:, :, 2, :, block_size] - CumB[:, :, 2, :, i2]))
                for k in range(4):
                    sumqx = sumqx4[:, :, k]
                    sumq2 = sumq24[:, :, k]
                    improve = (sumq2 > 0) & (sumqx * sumqx > best_score * sumq2)
                    new_scale = np.where(sumq2 != 0, sumqx / sumq2, np.float32(0.0))
                    best_score = np.where(improve, new_scale * sumqx, best_score)
                    best_scale = np.where(improve, new_scale, best_scale)
                    besti1 = np.where(improve, i1, besti1)
                    besti2 = np.where(improve, i2, besti2)
                    best_k = np.where(improve, k, best_k)

        not_found = besti1 < 0
        use_default = degenerate | not_found

        pos_idx = np.arange(block_size).reshape(1, 1, block_size)
        level_sorted = np.where(pos_idx < besti1[..., None], 0,
                                 np.where(pos_idx < besti2[..., None], 1, 2)).astype(np.int64)
        L = np.zeros((n_blocks, n_ib, block_size), dtype=np.int64)
        np.put_along_axis(L, order, level_sorted, axis=-1)

        scale = best_scale
        flip = scale < 0
        L = np.where(flip[..., None], 2 - L, L)
        scale = np.abs(scale)
        best_k = np.where(flip, 3 - best_k, best_k)

        L = np.where(use_default[..., None], 1, L)
        scale = np.where(use_default, np.float32(0.0), scale)
        best_k = np.where(use_default, 0, best_k)

        use_p_kg0 = best_k < 2
        use_p_kg1 = (best_k % 2) == 0

        index_kg = np.zeros((n_blocks, n_ib, 2), dtype=np.int64)
        on_grid_kg = np.zeros((n_blocks, n_ib, 2), dtype=bool)
        L_final = L.copy()
        for kg in range(2):
            xg = xb[:, :, 8 * kg:8 * kg + 8]
            wg = weight[:, :, 8 * kg:8 * kg + 8]
            u = IQ2_S._pack_u(L_final[:, :, 8 * kg:8 * kg + 8])
            use_p_this = use_p_kg0 if kg == 0 else use_p_kg1
            idx, Lg, og = _iq1m_grid_lookup(u, xg, wg, scale, use_p_this, x_p, x_m,
                                             pos, kmap, neigh_padded, neigh_count)
            index_kg[:, :, kg] = idx
            on_grid_kg[:, :, kg] = og
            L_final[:, :, 8 * kg:8 * kg + 8] = Lg

        need_rescan = ~on_grid_kg.all(axis=-1)  # (n_blocks, n_ib)
        sumqx_f = np.zeros((n_blocks, n_ib), dtype=np.float32)
        sumq2_f = np.zeros((n_blocks, n_ib), dtype=np.float32)
        for kg in range(2):
            use_p_this = use_p_kg0 if kg == 0 else use_p_kg1
            Lg = L_final[:, :, 8 * kg:8 * kg + 8]
            q_all = np.where(use_p_this[..., None], x_p[Lg], x_m[Lg])  # (n_blocks, n_ib, 8)
            wg = weight[:, :, 8 * kg:8 * kg + 8]
            xg = xb[:, :, 8 * kg:8 * kg + 8]
            for j in range(8):
                sumqx_f += wg[:, :, j] * q_all[:, :, j] * xg[:, :, j]
                sumq2_f += wg[:, :, j] * q_all[:, :, j] * q_all[:, :, j]
        rescan_valid = (sumqx_f > 0) & (sumq2_f > 0)
        scale = np.where(need_rescan & rescan_valid, sumqx_f / sumq2_f, scale)

        valid_ib = ~use_default
        index_kg_masked = np.where(valid_ib[..., None], index_kg, 0)
        qs_out = (index_kg_masked & 0xFF).astype(np.uint8).reshape(n_blocks, QK_K // 8)

        qh_low = ((index_kg_masked[:, :, 0] >> 8) & 0x07).astype(np.uint8)
        qh_high = ((index_kg_masked[:, :, 1] >> 8) & 0x07).astype(np.uint8)
        qh_byte = qh_low | (qh_high << 4)  # (n_blocks, n_ib)

        scales_masked = np.where(valid_ib, scale, np.float32(0.0))
        max_scale = scales_masked.max(axis=-1)  # (n_blocks,)
        has_block_scale = max_scale > 0
        d0 = np.where(has_block_scale, max_scale / 15.0, np.float32(0.0))
        id_d0 = np.where(d0 != 0, 1.0 / d0, np.float32(0.0))

        l_level = np.clip(np_nearest_int(0.5 * (id_d0[:, None] * scales_masked - 1)), 0, 7).astype(np.int64)

        mask_table = np.array([0x00, 0x80, 0x08, 0x88], dtype=np.uint8)
        shift_mask = mask_table[best_k]  # (n_blocks, n_ib)
        qh_byte_final = (qh_byte | shift_mask).astype(np.uint8)

        sc = np.zeros((n_blocks, 4), dtype=np.uint16)
        for ib in range(n_ib):
            sc[:, ib // 4] |= (l_level[:, ib].astype(np.uint16) << np.uint16(3 * (ib % 4)))

        sumqx_final = np.zeros((n_blocks,), dtype=np.float32)
        sumq2_final = np.zeros((n_blocks,), dtype=np.float32)
        for ib in range(n_ib):
            lf = l_level[:, ib].astype(np.float32)
            qmul = 2 * lf + 1
            for kg in range(2):
                use_p_this = use_p_kg0[:, ib] if kg == 0 else use_p_kg1[:, ib]
                Lg = L_final[:, ib, 8 * kg:8 * kg + 8]
                qx = np.where(use_p_this[:, None], x_p[Lg], x_m[Lg])  # (n_blocks, 8)
                wg = weight[:, ib, 8 * kg:8 * kg + 8]
                xg = xb[:, ib, 8 * kg:8 * kg + 8]
                for j in range(8):
                    q = qx[:, j] * qmul
                    sumqx_final += wg[:, j] * q * xg[:, j]
                    sumq2_final += wg[:, j] * q * q
        d_refined = np.where(sumq2_final > 0, sumqx_final / sumq2_final, d0)
        s_f16 = (d_refined * np.float32(1.1125)).astype(np.float16)
        s_u16 = s_f16.view(np.uint16)
        s_u16 = np.where(has_block_scale, s_u16, np.uint16(0))

        sc[:, 0] |= ((s_u16 & np.uint16(0x000F)) << np.uint16(12))
        sc[:, 1] |= ((s_u16 & np.uint16(0x00F0)) << np.uint16(8))
        sc[:, 2] |= ((s_u16 & np.uint16(0x0F00)) << np.uint16(4))
        sc[:, 3] |= (s_u16 & np.uint16(0xF000))

        qh_out = qh_byte_final.reshape(n_blocks, QK_K // 16)
        scales_bytes = sc.view(np.uint8).reshape(n_blocks, QK_K // 32)

        return np.concatenate([qs_out, qh_out, scales_bytes], axis=-1)

    # Okay *this* type is weird. It's the only one which stores the f16 scales in multiple parts.
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        qs, rest = np.hsplit(blocks, [QK_K // 8])
        qh, scales = np.hsplit(rest, [QK_K // 16])

        # The f16 scale is packed across multiple bytes
        scales = scales.view(np.uint16)
        d = (scales.reshape((n_blocks, 4)) & np.uint16(0xF000)) >> np.array([12, 8, 4, 0], dtype=np.uint16).reshape((1, 4))
        d = d[..., 0] | d[..., 1] | d[..., 2] | d[..., 3]
        d = d.view(np.float16).astype(np.float32).reshape((n_blocks, 1))

        scales = scales.reshape(n_blocks, -1, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape((1, 1, 4))
        scales = (scales & 0x07).reshape((n_blocks, -1))
        dl = d * (2 * scales + 1)
        dl = dl.reshape((n_blocks, -1, 2, 1, 1))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        qs = qs.astype(np.uint16) | ((qh & 0x07).astype(np.uint16) << 8).reshape((n_blocks, -1))

        delta = np.where(qh & 0x08 == 0, cls.delta, -cls.delta)
        delta = delta.reshape((n_blocks, -1, 2, 2, 1))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 2, 8))

        return (dl * (grid + delta)).reshape((n_blocks, -1))


class IQ4_NL(__Quant, qtype=GGMLQuantizationType.IQ4_NL):
    kvalues = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        block_size = cls.block_size  # 32

        # Port of quantize_row_iq4_nl_impl called with
        # (super_block_size=block_size=32, ntry=7) — matches quantize_iq4_nl,
        # the function ggml_quantize_chunk actually dispatches to (NOT
        # quantize_row_iq4_nl_ref, which uses ntry=-1 and is unused by the
        # normal quantize path).
        blocks_f = blocks.astype(np.float32).reshape(n_blocks, 1, block_size)
        kvals = np.array(cls.kvalues, dtype=np.int8)

        scale = _make_iq4_quants(blocks_f, kvals, ntry=7)[:, 0]  # (n_blocks,)

        # Final requant uses the *full-precision* scale directly (verified
        # against the C source — unlike the K-quants, no fp16 round-trip
        # before this pass).
        kvals_f = kvals.astype(np.float32)
        id_ = np.where(scale != 0, 1.0 / scale, 0.0)
        al = id_[:, None] * blocks.astype(np.float32)
        best = np.argmin(np.abs(al[..., None] - kvals_f), axis=-1).astype(np.uint8)  # (n_blocks, 32)

        # C: q4[j] = L[j] | (L[16+j] << 4) — split-half pairing, not adjacent.
        half = block_size // 2
        qs = best[:, :half] | (best[:, half:] << np.uint8(4))

        d = scale.astype(np.float16).reshape(n_blocks, 1).view(np.uint8)

        return np.concatenate([d, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))

        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 1))

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape(1, 1, 16)
        qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32).reshape((n_blocks, -1))

        return (d * qs)


class IQ4_XS(__Quant, qtype=GGMLQuantizationType.IQ4_XS):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        n_sub = QK_K // 32  # 8 sub-blocks
        sub_size = 32

        blocks_3d = blocks.reshape((n_blocks, n_sub, sub_size)).astype(np.float32)
        kvals = np.array(IQ4_NL.kvalues, dtype=np.int8)
        kvals_f = kvals.astype(np.float32)

        # Port of quantize_row_iq4_nl_impl(super_block_size=QK_K, block_size=32,
        # ntry=7) — matches quantize_iq4_xs.
        scales = _make_iq4_quants(blocks_3d, kvals, ntry=7)  # (n_blocks, 8)

        abs_scales = np.abs(scales)
        amax_idx = np.argmax(abs_scales, axis=-1)
        max_scale = np.take_along_axis(scales, amax_idx[:, None], axis=-1)[:, 0]  # signed

        d = -max_scale / 32.0
        d_all = d.astype(np.float16)
        id_ = np.where(d != 0, 1.0 / d, 0.0)  # full precision, no fp16 round-trip (verified vs C)

        l = np.clip(np_nearest_int(id_[:, None] * scales), -32, 31).astype(np.int32)  # (n_blocks, 8)
        dl = d[:, None] * l.astype(np.float32)
        idl = np.where(dl != 0, 1.0 / dl, 0.0)[..., None]  # (n_blocks, 8, 1)

        al = idl * blocks_3d
        best = np.argmin(np.abs(al[..., None] - kvals_f), axis=-1).astype(np.uint8)  # (n_blocks, 8, 32)

        l6 = (l + 32).astype(np.uint8)  # (n_blocks, 8), 6-bit in [0,63]
        l_l = l6 & 0x0F
        l_h = l6 >> 4

        scales_l = np.zeros((n_blocks, n_sub // 2), dtype=np.uint8)
        for ib in range(0, n_sub, 2):
            scales_l[:, ib // 2] = l_l[:, ib] | (l_l[:, ib + 1] << 4)
        scales_h = np.zeros((n_blocks,), dtype=np.uint16)
        for ib in range(n_sub):
            scales_h |= (l_h[:, ib].astype(np.uint16) << np.uint16(2 * ib))

        # qs: within each 32-element sub-block, split-half pairing (element j
        # with element 16+j), same as IQ4_NL.
        Bg = best.reshape(n_blocks, n_sub, 2, sub_size // 2)
        qs = (Bg[:, :, 0, :] | (Bg[:, :, 1, :] << 4)).astype(np.uint8).reshape(n_blocks, QK_K // 2)

        d_bytes = d_all.view(np.uint8)

        return np.concatenate([d_bytes.reshape((n_blocks, 2)), scales_h.view(np.uint8).reshape((n_blocks, 2)), scales_l, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        scales_h, rest = np.hsplit(rest, [2])
        scales_l, qs = np.hsplit(rest, [QK_K // 64])

        d = d.view(np.float16).astype(np.float32)
        scales_h = scales_h.view(np.uint16)

        scales_l = scales_l.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        scales_h = scales_h.reshape((n_blocks, 1, -1)) >> np.array([2 * i for i in range(QK_K // 32)], dtype=np.uint16).reshape((1, -1, 1))
        scales_l = scales_l.reshape((n_blocks, -1)) & np.uint8(0x0F)
        scales_h = scales_h.reshape((n_blocks, -1)).astype(np.uint8) & np.uint8(0x03)

        scales = (scales_l | (scales_h << np.uint8(4))).astype(np.int8) - np.int8(32)
        dl = (d * scales.astype(np.float32)).reshape((n_blocks, -1, 1))

        qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = qs.reshape((n_blocks, -1, 32, 1)) & np.uint8(0x0F)

        kvalues = np.array(IQ4_NL.kvalues, dtype=np.int8).reshape((1, 1, 1, -1))
        qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32).reshape((n_blocks, -1, 32))

        return (dl * qs).reshape((n_blocks, -1))


# =============================================================================
# Newly added quant types
# =============================================================================

class Q1_0(__Quant, qtype=GGMLQuantizationType.Q1_0):
    """1-bit binary quantization. block_size=128, type_size=18 (2 bytes d + 16 bytes qs).
    Each weight is either +d or -d based on sign."""
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        # d = mean absolute value per block
        d = np.abs(blocks).mean(axis=-1, keepdims=True).astype(np.float32)
        d = np.where(d == 0, np.float32(1e-10), d)

        # Sign: 1 if w >= 0, 0 if w < 0
        bits = (blocks >= 0).astype(np.uint8)

        # Pack bits: 8 bits per byte
        qs = np.packbits(bits.reshape((n_blocks, QK1_0)), axis=-1, bitorder="little")

        d_bytes = d.astype(np.float16).view(np.uint8)

        return np.concatenate([d_bytes.reshape((n_blocks, 2)), qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])
        d = d.view(np.float16).astype(np.float32)
        neg_d = -d

        bits = np.unpackbits(qs, axis=-1, bitorder="little").reshape((n_blocks, QK1_0))

        # 1 bit -> +d or -d
        result = np.where(bits.astype(np.bool), d, neg_d)
        return result.astype(np.float32)


class Q2_0(__Quant, qtype=GGMLQuantizationType.Q2_0):
    """2-bit quantization. block_size=64, type_size=18 (2 bytes d + 16 bytes qs).
    Values: {-1, 0, +1, +2} * d, encoded as 2-bit {0, 1, 2, 3}."""
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        # d = max absolute value
        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            id = np.where(d != 0, 1.0 / d, 0)

        # Quantize: round(w/d), clamp to [-1, 2], shift to [0, 3]
        q = np_roundf(blocks * id).astype(np.int8)
        q = np.clip(q, -1, 2).astype(np.int8) + 1

        # Pack 4 values per byte (2 bits each)
        qs = q.reshape((n_blocks, -1, 4)).astype(np.uint8)
        qs = qs[..., 0] | (qs[..., 1] << np.uint8(2)) | (qs[..., 2] << np.uint8(4)) | (qs[..., 3] << np.uint8(6))

        d_bytes = d.astype(np.float16).view(np.uint8)

        return np.concatenate([d_bytes.reshape((n_blocks, 2)), qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])
        d = d.view(np.float16).astype(np.float32)

        # Unpack 2-bit values
        qs = qs.reshape((n_blocks, -1, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        q_val = (qs & np.uint8(0x03)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(1)

        return (d * q_val.astype(np.float32))


class Q8_K(__Quant, qtype=GGMLQuantizationType.Q8_K):
    """8-bit K-block quantization. block_size=256, type_size=292 (4 bytes d + 256 bytes qs + 32 bytes bsums).
    
    Note: Q8_K has a float d instead of fp16, making its type_size 4 + QK_K + QK_K // 8.
    bsums stores the sum of int8 values in each group of 16 elements (used for efficient dot products).
    """
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        # Find max absolute value per block
        amax = abs(blocks).max(axis=-1, keepdims=True)  # (n_blocks, 1)

        # Scale to int8 range [-127, 127]
        iscale = np.where(amax > 0, 127.0 / amax, 1.0)
        qs = np_roundf(blocks * iscale).astype(np.int8)
        qs = np.clip(qs, -127, 127)

        # Compute d as the inverse of iscale (note: sign convention differs from Q8_0)
        d = np.where(amax > 0, amax / 127.0, 0.0).astype(np.float32)

        # Compute block sums: sum of int8 values in groups of 16
        bsums = qs.reshape((n_blocks, -1, 16)).sum(axis=-1).astype(np.int16)

        # Pack: d (float32 = 4 bytes) + qs (256 bytes) + bsums (16 * 2 = 32 bytes)
        d_bytes = d.view(np.uint8)
        qs_bytes = qs.view(np.uint8)
        bsums_bytes = bsums.view(np.uint8)

        return np.concatenate([d_bytes.reshape((n_blocks, 4)), qs_bytes, bsums_bytes], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d_bytes, rest = np.hsplit(blocks, [4])
        qs_bytes, bsums_bytes = np.hsplit(rest, [QK_K])

        d = d_bytes.view(np.float32)
        qs = qs_bytes.view(np.int8).astype(np.float32)

        return (d * qs)


class Q8_1(__Quant, qtype=GGMLQuantizationType.Q8_1):
    """8-bit quantization with sum. block_size=32, type_size=36 (2 bytes d + 2 bytes s + 32 bytes qs).
    Q8_1 is primarily an intermediate format used for efficient dot products.
    d = amax / 127, qs = round(x / d), s = d * sum(qs) stored as f16.
    """
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        amax = abs(blocks).max(axis=-1, keepdims=True)  # (n, 1)
        d = amax / 127.0
        with np.errstate(divide="ignore", invalid="ignore"):
            id = np.where(d != 0, 1.0 / d, 0)

        qs = np_roundf(blocks * id).astype(np.int8)

        # s = d * sum(qs)
        s = (d.reshape((n_blocks,)) * qs.sum(axis=-1).astype(np.float32))

        d_bytes = d.astype(np.float16).view(np.uint8).reshape((n_blocks, 2))
        s_bytes = s.astype(np.float16).view(np.uint8).reshape((n_blocks, 2))
        qs_bytes = qs.view(np.uint8)

        return np.concatenate([d_bytes, s_bytes, qs_bytes], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        s, qs = np.hsplit(rest, [2])

        d = d.view(np.float16).astype(np.float32)
        qs = qs.view(np.int8).astype(np.float32)

        return (d * qs)
