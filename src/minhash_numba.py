from __future__ import annotations

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

from core import ShingleDB, MinHashParams, MERSENNE_P


@njit(parallel=True, fastmath=True, nogil=True)
def _kernel_parallel_dmajor(vals, offsets, a, b, P, N):
    D = offsets.shape[0] - 1
    sig = np.empty((D, N), dtype=np.int64)
    for d in prange(D):
        lo = offsets[d]
        hi = offsets[d + 1]
        for i in range(N):
            sig[d, i] = P
        for idx in range(lo, hi):
            x = vals[idx]
            for i in range(N):
                v = (a[i] * x + b[i]) % P
                if v < sig[d, i]:
                    sig[d, i] = v
    return sig


@njit(fastmath=True, nogil=True)
def _kernel_serial_dmajor(vals, offsets, a, b, P, N):
    D = offsets.shape[0] - 1
    sig = np.empty((D, N), dtype=np.int64)
    for d in range(D):
        lo = offsets[d]
        hi = offsets[d + 1]
        for i in range(N):
            sig[d, i] = P
        for idx in range(lo, hi):
            x = vals[idx]
            for i in range(N):
                v = (a[i] * x + b[i]) % P
                if v < sig[d, i]:
                    sig[d, i] = v
    return sig


@njit(parallel=True, fastmath=False, nogil=True)
def _kernel_parallel_dmajor_nofast(vals, offsets, a, b, P, N):
    D = offsets.shape[0] - 1
    sig = np.empty((D, N), dtype=np.int64)
    for d in prange(D):
        lo = offsets[d]
        hi = offsets[d + 1]
        for i in range(N):
            sig[d, i] = P
        for idx in range(lo, hi):
            x = vals[idx]
            for i in range(N):
                v = (a[i] * x + b[i]) % P
                if v < sig[d, i]:
                    sig[d, i] = v
    return sig


@njit(parallel=True, fastmath=True, nogil=True)
def _kernel_parallel_hmajor(vals, offsets, a, b, P, N):
    D = offsets.shape[0] - 1
    sig = np.empty((N, D), dtype=np.int64)
    for d in prange(D):
        lo = offsets[d]
        hi = offsets[d + 1]
        for i in range(N):
            sig[i, d] = P
        for idx in range(lo, hi):
            x = vals[idx]
            for i in range(N):
                v = (a[i] * x + b[i]) % P
                if v < sig[i, d]:
                    sig[i, d] = v
    return sig


def _prep(db: ShingleDB, params: MinHashParams):
    return (db.vals, db.offsets, params.a, params.b,
            np.int64(MERSENNE_P), np.int64(params.n_hashes))


def minhash_numba_serial(db: ShingleDB, params: MinHashParams) -> np.ndarray:
    v, o, a, b, P, N = _prep(db, params)
    return _kernel_serial_dmajor(v, o, a, b, P, N).T


def minhash_numba_parallel(db: ShingleDB, params: MinHashParams,
                           n_threads: int | None = None,
                           layout: str = "dmajor",
                           fastmath: bool = True) -> np.ndarray:
    if n_threads is not None:
        set_num_threads(n_threads)
    v, o, a, b, P, N = _prep(db, params)
    if layout == "hmajor":
        return _kernel_parallel_hmajor(v, o, a, b, P, N)
    if not fastmath:
        return _kernel_parallel_dmajor_nofast(v, o, a, b, P, N).T
    return _kernel_parallel_dmajor(v, o, a, b, P, N).T


def warmup(db: ShingleDB, params: MinHashParams) -> None:
    minhash_numba_serial(db, params)
    minhash_numba_parallel(db, params, layout="dmajor")
    minhash_numba_parallel(db, params, layout="hmajor")
    minhash_numba_parallel(db, params, layout="dmajor", fastmath=False)


if __name__ == "__main__":
    import core as C
    print("Validating Numba kernels against the NumPy reference...")
    docs, _ = C.generate_corpus(200, vocab_size=5000, doc_len=200,
                                n_dup_clusters=5, cluster_size=3, seed=4)
    db = C.build_shingle_db(docs, k=4)
    params = C.MinHashParams.create(96, seed=8)

    ref = C.minhash_signatures_numpy(db, params)
    checks = {
        "serial            ": minhash_numba_serial(db, params),
        "parallel dmajor   ": minhash_numba_parallel(db, params, layout="dmajor"),
        "parallel hmajor   ": minhash_numba_parallel(db, params, layout="hmajor"),
        "parallel no-fastm ": minhash_numba_parallel(db, params, fastmath=False),
    }
    ok_all = True
    for name, sig in checks.items():
        ok = np.array_equal(sig, ref)
        ok_all &= ok
        print(f"  {name}: shape={sig.shape}  identical to reference: {ok}")
    print(f"\nMax threads Numba can use here: {get_num_threads()}")
    print("ALL PASS" if ok_all else "FAILURE")
