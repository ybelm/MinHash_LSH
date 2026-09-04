import numpy as np
import core as C

def test_estimator_unbiased(tol=0.03):
    print("\n--- [1] MinHash estimator vs true Jaccard ---")
    rng = np.random.default_rng(1)
    # Build pairs of documents with a varied overlap
    docs = []
    for _ in range(40):
        base = rng.integers(0, 5000, size=400, dtype=np.int64)
        keep = rng.uniform(0.2, 0.95)
        m = base.copy()
        n_rep = int((1 - keep) * m.shape[0])
        if n_rep:
            pos = rng.choice(m.shape[0], size=n_rep, replace=False)
            m[pos] = rng.integers(0, 5000, size=n_rep, dtype=np.int64)
        docs.append(base)
        docs.append(m)

    db = C.build_shingle_db(docs, k=1)   # k=1 -> shingle set == token set
    params = C.MinHashParams.create(256, seed=7)
    sig = C.minhash_signatures_numpy(db, params)

    errs = []
    for p in range(0, len(docs), 2):
        i, j = p, p + 1
        true_j = C.exact_jaccard(db, i, j)
        est = np.count_nonzero(sig[:, i] == sig[:, j]) / params.n_hashes
        errs.append(abs(true_j - est))
    mae = float(np.mean(errs))
    ok = mae < tol
    print(f"    mean |Jaccard - estimate| = {mae:.4f}  (tol {tol})  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_impl_equivalence():
    print("\n--- [2] pure-Python vs NumPy signatures ---")
    docs, _ = C.generate_corpus(60, vocab_size=3000, doc_len=120, seed=3)
    db = C.build_shingle_db(docs, k=3)
    params = C.MinHashParams.create(64, seed=9)
    sig_np = C.minhash_signatures_numpy(db, params)
    sig_py = C.minhash_signatures_python(db, params)
    ok = np.array_equal(sig_np, sig_py)
    print(f"    identical signature matrices: {ok}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_end_to_end(min_recall=0.95):
    print("\n--- [3] LSH recovers injected near-duplicates ---")
    docs, truth = C.generate_corpus(
        n_docs=800, vocab_size=8000, doc_len=200,
        n_dup_clusters=25, cluster_size=3, mutation_rate=0.03, seed=11)
    db = C.build_shingle_db(docs, k=3)

    n_hashes = 128
    threshold = 0.5
    params = C.MinHashParams.create(n_hashes, seed=5)
    sig = C.minhash_signatures_numpy(db, params)

    n_bands = C.optimal_bands(n_hashes, threshold)
    cand = C.lsh_candidate_pairs(sig, n_bands)
    found = C.verify_pairs(sig, cand, threshold=threshold)

    recall, precision = C.recall_precision(found, truth)
    n_pairs = db.n_docs * (db.n_docs - 1) // 2
    print(f"    docs={db.n_docs}  bands={n_bands} x r={n_hashes // n_bands}  "
          f"threshold={threshold}")
    print(f"    candidate pairs={len(cand)}  (vs {n_pairs} brute-force)  "
          f"reduction={100 * (1 - len(cand) / n_pairs):.2f}%")
    print(f"    injected clusters -> ground-truth pairs={len(truth)}")
    print(f"    recall={recall:.3f}  precision={precision:.3f}")
    ok = recall >= min_recall
    print(f"    -> {'PASS' if ok else 'FAIL'} (recall >= {min_recall})")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("  MinHash + LSH  --  reference pipeline correctness")
    print("=" * 60)
    results = [test_estimator_unbiased(), test_impl_equivalence(), test_end_to_end()]
    print("\n" + "=" * 60)
    print(f"  OVERALL: {'ALL PASS' if all(results) else 'FAILURE'}")
    print("=" * 60)
