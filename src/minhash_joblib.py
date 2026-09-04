from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from core import ShingleDB, MinHashParams, MERSENNE_P


def slice_payload(db: ShingleDB, lo: int, hi: int):
    v0 = db.offsets[lo]
    v1 = db.offsets[hi]
    vals = db.vals[v0:v1]
    local_off = (db.offsets[lo:hi + 1] - v0).astype(np.int64)
    return vals, local_off


def _worker(vals, offsets, a, b, P, N):
    n_docs = offsets.shape[0] - 1
    sig = np.empty((N, n_docs), dtype=np.int64)
    a = a[:, None]
    b = b[:, None]
    for d in range(n_docs):
        X = vals[offsets[d]:offsets[d + 1]][None, :]
        sig[:, d] = ((a * X + b) % P).min(axis=1)
    return sig


def _chunk_bounds(n_docs: int, n_chunks: int):
    edges = np.linspace(0, n_docs, n_chunks + 1, dtype=np.int64)
    return [(int(edges[c]), int(edges[c + 1])) for c in range(n_chunks)
            if edges[c + 1] > edges[c]]


def minhash_joblib(db: ShingleDB, params: MinHashParams, n_jobs: int = -1, n_chunks: int | None = None, backend: str = "loky") -> np.ndarray:
    N = params.n_hashes
    P = int(MERSENNE_P)
    if n_chunks is None:
        base = n_jobs if n_jobs and n_jobs > 0 else 4
        n_chunks = max(1, base * 4)
    n_chunks = min(n_chunks, db.n_docs)

    bounds = _chunk_bounds(db.n_docs, n_chunks)
    payloads = [slice_payload(db, lo, hi) for (lo, hi) in bounds]

    blocks = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_worker)(vals, off, params.a, params.b, P, N)
        for (vals, off) in payloads
    )
    return np.concatenate(blocks, axis=1)


if __name__ == "__main__":
    import core as C
    print("Validating joblib MinHash against the NumPy reference...")
    docs, _ = C.generate_corpus(300, vocab_size=6000, doc_len=180, n_dup_clusters=8, cluster_size=3, seed=6)
    db = C.build_shingle_db(docs, k=4)
    params = C.MinHashParams.create(80, seed=2)
    ref = C.minhash_signatures_numpy(db, params)

    ok_all = True
    for backend in ("loky", "threading"):
        for n_chunks in (4, 16, 64):
            sig = minhash_joblib(db, params, n_jobs=2, n_chunks=n_chunks, backend=backend)
            ok = np.array_equal(sig, ref)
            ok_all &= ok
            print(f"  backend={backend:<11} n_chunks={n_chunks:<3} "
                  f"shape={sig.shape}  identical: {ok}")
    print("\nALL PASS" if ok_all else "\nFAILURE")
