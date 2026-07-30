from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


MERSENNE_P = np.int64((1 << 31) - 1)
POLY_BASE = np.uint64(1099511628211)
SIG_DTYPE = np.int64


@dataclass
class ShingleDB:
    vals: np.ndarray
    offsets: np.ndarray
    n_docs: int
    ground_truth: set = field(default_factory=set)

    def doc(self, d: int) -> np.ndarray:
        return self.vals[self.offsets[d]:self.offsets[d + 1]]

    @property
    def nnz(self) -> int:
        return int(self.offsets[-1])

@dataclass
class MinHashParams:
    a: np.ndarray
    b: np.ndarray
    n_hashes: int

    @staticmethod
    def create(n_hashes: int, seed: int = 12345) -> "MinHashParams":
        rng = np.random.default_rng(seed)
        p = int(MERSENNE_P)
        a = rng.integers(1, p, size=n_hashes, dtype=np.int64)
        b = rng.integers(0, p, size=n_hashes, dtype=np.int64)
        return MinHashParams(a=a, b=b, n_hashes=n_hashes)

def _hash_shingles(tokens: np.ndarray, k: int) -> np.ndarray:
    L = tokens.shape[0]
    if L < k:

        k = L
    n_sh = L - k + 1
    t = tokens.astype(np.uint64)
    h = np.zeros(n_sh, dtype=np.uint64)
    base = POLY_BASE
    for j in range(k):
        h = h * base + t[j:j + n_sh]
    ids = (h % np.uint64(int(MERSENNE_P))).astype(np.int64)
    return np.unique(ids)

def build_shingle_db(docs: list[np.ndarray], k: int,
                     ground_truth: set | None = None) -> ShingleDB:
    per_doc = [_hash_shingles(d, k) for d in docs]
    offsets = np.zeros(len(per_doc) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([s.shape[0] for s in per_doc])
    vals = (np.concatenate(per_doc) if per_doc
            else np.zeros(0, dtype=np.int64))
    return ShingleDB(vals=vals, offsets=offsets, n_docs=len(docs),
                     ground_truth=ground_truth or set())

def minhash_signatures_numpy(db: ShingleDB, params: MinHashParams) -> np.ndarray:
    N = params.n_hashes
    sig = np.empty((N, db.n_docs), dtype=SIG_DTYPE)
    a = params.a[:, None]
    b = params.b[:, None]
    p = MERSENNE_P
    for d in range(db.n_docs):
        X = db.doc(d)[None, :]
        hashed = (a * X + b) % p
        sig[:, d] = hashed.min(axis=1)
    return sig

def minhash_signatures_python(db: ShingleDB, params: MinHashParams) -> np.ndarray:
    N = params.n_hashes
    a = params.a.tolist()
    b = params.b.tolist()
    p = int(MERSENNE_P)
    offsets = db.offsets
    vals = db.vals
    sig = np.empty((N, db.n_docs), dtype=SIG_DTYPE)
    for d in range(db.n_docs):
        lo, hi = int(offsets[d]), int(offsets[d + 1])
        for i in range(N):
            ai, bi = a[i], b[i]
            m = p
            for idx in range(lo, hi):
                v = (ai * int(vals[idx]) + bi) % p
                if v < m:
                    m = v
            sig[i, d] = m
    return sig

def lsh_candidate_pairs(sig: np.ndarray, n_bands: int) -> set:
    N, D = sig.shape
    assert N % n_bands == 0, "n_hashes must be divisible by n_bands"
    r = N // n_bands
    candidates: set = set()
    for band in range(n_bands):
        rows = sig[band * r:(band + 1) * r, :]

        keys = np.zeros(D, dtype=np.uint64)
        base = POLY_BASE
        for row in range(r):
            keys = keys * base + rows[row].astype(np.uint64)
        buckets: dict = {}
        for d in range(D):
            buckets.setdefault(keys[d], []).append(d)
        for members in buckets.values():
            m = len(members)
            if m < 2:
                continue
            for x in range(m):
                for y in range(x + 1, m):
                    i, j = members[x], members[y]
                    candidates.add((i, j) if i < j else (j, i))
    return candidates

def verify_pairs(sig: np.ndarray, candidates, threshold: float) -> set:
    N = sig.shape[0]
    out = set()
    for (i, j) in candidates:
        est = np.count_nonzero(sig[:, i] == sig[:, j]) / N
        if est >= threshold:
            out.add((i, j))
    return out

def exact_jaccard(db: ShingleDB, i: int, j: int) -> float:
    A = db.doc(i)
    B = db.doc(j)
    sa, sb = set(A.tolist()), set(B.tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def optimal_bands(n_hashes: int, threshold: float) -> int:
    best_b, best_err = 1, 1e9
    for b in range(1, n_hashes + 1):
        if n_hashes % b:
            continue
        r = n_hashes // b
        t_mid = (1.0 / b) ** (1.0 / r)
        err = abs(t_mid - threshold)
        if err < best_err:
            best_err, best_b = err, b
    return best_b

def generate_corpus(n_docs: int, vocab_size: int = 20000,
                    doc_len: int = 300, doc_len_jitter: int = 60,
                    n_dup_clusters: int = 0, cluster_size: int = 3,
                    mutation_rate: float = 0.10, zipf_s: float = 1.2,
                    seed: int = 0):
    rng = np.random.default_rng(seed)

    ranks = np.arange(1, vocab_size + 1)
    weights = 1.0 / np.power(ranks, zipf_s)
    weights /= weights.sum()

    docs: list[np.ndarray] = []

    def sample_doc() -> np.ndarray:
        L = max(10, int(doc_len + rng.integers(-doc_len_jitter, doc_len_jitter + 1)))
        return rng.choice(vocab_size, size=L, p=weights).astype(np.int64)

    n_base = n_docs - n_dup_clusters * cluster_size
    for _ in range(max(0, n_base)):
        docs.append(sample_doc())

    ground_truth: set = set()
    for _ in range(n_dup_clusters):
        base = sample_doc()
        cluster_ids = []
        for _c in range(cluster_size):
            variant = base.copy()
            n_mut = int(mutation_rate * variant.shape[0])
            if n_mut:
                pos = rng.choice(variant.shape[0], size=n_mut, replace=False)
                variant[pos] = rng.choice(vocab_size, size=n_mut, p=weights)
            cluster_ids.append(len(docs))
            docs.append(variant)
        for x in range(len(cluster_ids)):
            for y in range(x + 1, len(cluster_ids)):
                i, j = cluster_ids[x], cluster_ids[y]
                ground_truth.add((i, j) if i < j else (j, i))

    perm = rng.permutation(len(docs))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    docs = [docs[p] for p in perm]
    ground_truth = {(int(min(inv[i], inv[j])), int(max(inv[i], inv[j])))
                    for (i, j) in ground_truth}
    return docs, ground_truth
