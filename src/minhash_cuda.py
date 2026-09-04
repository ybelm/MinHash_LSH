from __future__ import annotations
import numpy as np
from numba import cuda, int64

from core import ShingleDB, MinHashParams, MERSENNE_P

TILE = 256 # shared-memory tile size


# Kernels
@cuda.jit
def _kernel_global(vals, offsets, a, b, P, sig, D, N):
    t = cuda.grid(1)
    if t >= D * N:
        return
    d = t // N
    i = t % N
    lo = offsets[d]
    hi = offsets[d + 1]
    ai = a[i]
    bi = b[i]
    m = P
    for idx in range(lo, hi):
        v = (ai * vals[idx] + bi) % P
        if v < m:
            m = v
    sig[d, i] = m


@cuda.jit
def _kernel_shared(vals, offsets, a, b, P, sig, N):
    d = cuda.blockIdx.x
    i = cuda.threadIdx.x  # hash index
    D = offsets.shape[0] - 1
    if d >= D:
        return

    tile = cuda.shared.array(shape=TILE, dtype=int64)
    lo = offsets[d]
    hi = offsets[d + 1]

    active = i < N
    ai = a[i] if active else 0
    bi = b[i] if active else 0
    m = P

    pos = lo
    while pos < hi:
        n = hi - pos
        if n > TILE:
            n = TILE
        # cooperative load of this tile into shared memory
        j = cuda.threadIdx.x
        while j < n:
            tile[j] = vals[pos + j]
            j += cuda.blockDim.x
        cuda.syncthreads()

        if active:
            for jj in range(n):
                v = (ai * tile[jj] + bi) % P
                if v < m:
                    m = v
        cuda.syncthreads()
        pos += n

    if active:
        sig[d, i] = m


# Host wrappers
def gpu_available() -> bool:
    try:
        return cuda.is_available()
    except Exception:
        return False


def minhash_cuda(db: ShingleDB, params: MinHashParams, block_size: int = 256, use_shared: bool = False) -> np.ndarray:

    D, N = db.n_docs, params.n_hashes
    P = np.int64(MERSENNE_P)

    d_vals = cuda.to_device(db.vals)
    d_off = cuda.to_device(db.offsets)
    d_a = cuda.to_device(params.a)
    d_b = cuda.to_device(params.b)
    d_sig = cuda.device_array((D, N), dtype=np.int64)

    if use_shared:
        _kernel_shared[D, N](d_vals, d_off, d_a, d_b, P, d_sig, N)
    else:
        total = D * N
        blocks = (total + block_size - 1) // block_size
        _kernel_global[blocks, block_size](d_vals, d_off, d_a, d_b, P, d_sig, D, N)

    cuda.synchronize()
    return d_sig.copy_to_host().T


def make_gpu_runner(db: ShingleDB, params: MinHashParams, block_size: int = 256, use_shared: bool = False):
    D, N = db.n_docs, params.n_hashes
    P = np.int64(MERSENNE_P)
    d_vals = cuda.to_device(db.vals)
    d_off = cuda.to_device(db.offsets)
    d_a = cuda.to_device(params.a)
    d_b = cuda.to_device(params.b)
    d_sig = cuda.device_array((D, N), dtype=np.int64)
    total = D * N
    blocks = (total + block_size - 1) // block_size

    def run():
        if use_shared:
            _kernel_shared[D, N](d_vals, d_off, d_a, d_b, P, d_sig, N)
        else:
            _kernel_global[blocks, block_size](d_vals, d_off, d_a, d_b, P, d_sig, D, N)
        cuda.synchronize()
        return d_sig.copy_to_host().T

    return run


if __name__ == "__main__":
    import os
    import core as C
    sim = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
    if not (gpu_available() or sim):
        print("No CUDA GPU and simulator disabled.")
        print("Validate logic with:  NUMBA_ENABLE_CUDASIM=1 python3 minhash_cuda.py")
        raise SystemExit(0)

    n_docs = 20 if sim else 400
    N = 8 if sim else 128
    docs, _ = C.generate_corpus(n_docs, vocab_size=2000, doc_len=60 if sim else 200,
                                n_dup_clusters=2, cluster_size=3, seed=4)
    db = C.build_shingle_db(docs, k=3)
    params = C.MinHashParams.create(N, seed=8)
    ref = C.minhash_signatures_numpy(db, params)

    print(f"{'[SIMULATOR]' if sim else '[GPU]'} validating CUDA kernels "
          f"(docs={n_docs}, N={N})...")
    g = minhash_cuda(db, params, block_size=64, use_shared=False)
    s = minhash_cuda(db, params, use_shared=True)
    print(f"  global kernel identical to reference: {np.array_equal(g, ref)}")
    print(f"  shared kernel identical to reference: {np.array_equal(s, ref)}")
    print("ALL PASS" if np.array_equal(g, ref) and np.array_equal(s, ref) else "FAILURE")
