from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core import POLY_BASE


def _band_keys(sig: np.ndarray, band: int, r: int) -> np.ndarray:
    rows = sig[band * r:(band + 1) * r, :]
    D = sig.shape[1]
    keys = np.zeros(D, dtype=np.uint64)
    base = POLY_BASE
    for row in range(r):
        keys = keys * base + rows[row].astype(np.uint64)
    return keys


def _pairs_from_band(keys: np.ndarray) -> set:
    buckets: dict = {}
    for d in range(keys.shape[0]):
        buckets.setdefault(keys[d], []).append(d)
    out = set()
    for members in buckets.values():
        m = len(members)
        if m < 2:
            continue
        for x in range(m):
            for y in range(x + 1, m):
                i, j = members[x], members[y]
                out.add((i, j) if i < j else (j, i))
    return out


def lsh_reduction(sig: np.ndarray, n_bands: int, n_threads: int) -> set:
    N = sig.shape[0]
    r = N // n_bands
    band_ids = list(range(n_bands))

    def work(my_bands):
        local = set()
        for band in my_bands:
            local |= _pairs_from_band(_band_keys(sig, band, r))
        return local

    # Split bands across threads (static, contiguous partition)
    chunks = [band_ids[t::n_threads] for t in range(n_threads)]
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        partials = list(ex.map(work, chunks))
    result = set()
    for p in partials:
        result |= p
    return result


def lsh_shared_lock(sig: np.ndarray, n_bands: int, n_threads: int) -> set:
    N = sig.shape[0]
    r = N // n_bands
    shared: set = set()
    lock = threading.Lock()
    band_ids = list(range(n_bands))

    def work(my_bands):
        for band in my_bands:
            keys = _band_keys(sig, band, r)
            buckets: dict = {}
            for d in range(keys.shape[0]):
                buckets.setdefault(keys[d], []).append(d)
            for members in buckets.values():
                m = len(members)
                if m < 2:
                    continue
                for x in range(m):
                    for y in range(x + 1, m):
                        i, j = members[x], members[y]
                        pair = (i, j) if i < j else (j, i)
                        with lock:   # lock per insert
                            shared.add(pair)

    chunks = [band_ids[t::n_threads] for t in range(n_threads)]
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(work, chunks))
    return shared


def lsh_coarse_lock(sig: np.ndarray, n_bands: int, n_threads: int) -> set:
    N = sig.shape[0]
    r = N // n_bands
    shared: set = set()
    lock = threading.Lock()
    band_ids = list(range(n_bands))

    def work(my_bands):
        local = set()
        for band in my_bands:
            local |= _pairs_from_band(_band_keys(sig, band, r))
        with lock:   # one merge per thread
            shared.update(local)

    chunks = [band_ids[t::n_threads] for t in range(n_threads)]
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(work, chunks))
    return shared


STRATEGIES = {
    "reduction": lsh_reduction,
    "shared_lock": lsh_shared_lock,
    "coarse_lock": lsh_coarse_lock,
}


if __name__ == "__main__":
    import core as C
    print("Validating parallel LSH strategies against the reference...")
    docs, _ = C.generate_corpus(1000, vocab_size=8000, doc_len=200, n_dup_clusters=30, cluster_size=4, seed=1)
    db = C.build_shingle_db(docs, k=4)
    params = C.MinHashParams.create(128, seed=2)
    sig = C.minhash_signatures_numpy(db, params)
    n_bands = C.optimal_bands(128, 0.5)

    ref = C.lsh_candidate_pairs(sig, n_bands)
    print(f"  reference candidates: {len(ref)}  (bands={n_bands})")
    ok_all = True
    for name, fn in STRATEGIES.items():
        for th in (1, 2, 4):
            got = fn(sig, n_bands, th)
            ok = (got == ref)
            ok_all &= ok
            print(f"  {name:<12} threads={th}: {len(got)} pairs  identical: {ok}")
    print("\nALL PASS" if ok_all else "\nFAILURE")
