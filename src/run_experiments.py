from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import core as C
import minhash_numba as MN
import minhash_joblib as MJ
import lsh_parallel as LP
from numba import get_num_threads, set_num_threads

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


THREAD_GRID = [1, 2, 4, 8, 16, 32]


def _threads_available() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _grid(maxt: int):
    g = [t for t in THREAD_GRID if t <= maxt]
    if maxt not in g:
        g.append(maxt)
    return sorted(set(g))


def make_dataset(cfg):
    docs, truth = C.generate_corpus(
        n_docs=cfg["n_docs"], vocab_size=cfg["vocab"], doc_len=cfg["doc_len"],
        n_dup_clusters=cfg["n_dup_clusters"], cluster_size=4,
        mutation_rate=0.05, seed=1)
    db = C.build_shingle_db(docs, k=cfg["k"])
    params = C.MinHashParams.create(cfg["n_hashes"], seed=2)
    return db, params, truth


def exp_impl_comparison(db, params, cfg, n_runs):
    print("[impl_comparison]")
    rows = []


    cap = min(db.n_docs, 400)
    sub = C.ShingleDB(db.vals[:db.offsets[cap]], db.offsets[:cap + 1], cap)
    s, _ = C.benchmark(C.minhash_signatures_python, sub, params,
                       n_runs=1, warmup=0, label="python")
    py_full = s.mean * (db.n_docs / cap)
    rows.append(dict(impl="pure_python", mean_s=py_full, std_s=0.0,
                     min_s=py_full, max_s=py_full, cpu_s=py_full,
                     note=f"extrapolated x{db.n_docs/cap:.1f} from {cap} docs"))

    maxt = get_num_threads()
    impls = [
        ("numpy_vectorized", C.minhash_signatures_numpy, (db, params)),
        ("numba_serial", MN.minhash_numba_serial, (db, params)),
        ("numba_parallel", lambda d, p: MN.minhash_numba_parallel(d, p, n_threads=maxt), (db, params)),
        ("joblib_loky", lambda d, p: MJ.minhash_joblib(d, p, n_jobs=maxt, n_chunks=maxt * 4), (db, params)),
    ]
    ref = C.minhash_signatures_numpy(db, params)
    for name, fn, args in impls:
        s, out = C.benchmark(fn, *args, n_runs=n_runs, warmup=2, label=name)
        assert np.array_equal(out, ref), f"{name} disagrees with reference!"
        rows.append(dict(impl=name, mean_s=s.mean, std_s=s.std, min_s=s.min,
                         max_s=s.max, cpu_s=s.cpu_mean, note="validated==ref"))

    df = pd.DataFrame(rows)
    base = df.loc[df.impl == "pure_python", "mean_s"].values[0]
    df["speedup_vs_python"] = base / df["mean_s"]
    df.to_csv(os.path.join(RESULTS, "impl_comparison.csv"), index=False)
    print(df[["impl", "mean_s", "speedup_vs_python"]].to_string(index=False))
    return df


def exp_thread_scaling(db, params, cfg, n_runs):
    print("\n[thread_scaling]")
    maxt = get_num_threads()
    rows = []
    t1 = None
    for th in _grid(maxt):
        s, _ = C.benchmark(lambda d, p: MN.minhash_numba_parallel(d, p, n_threads=th),
                           db, params, n_runs=n_runs, warmup=2, label=f"threads={th}")
        if th == 1:
            t1 = s.mean
        rows.append(dict(threads=th, mean_s=s.mean, std_s=s.std, min_s=s.min,
                         cpu_s=s.cpu_mean,
                         speedup=(t1 / s.mean) if t1 else 1.0,
                         efficiency=((t1 / s.mean) / th) if t1 else 1.0))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "thread_scaling.csv"), index=False)
    print(df[["threads", "mean_s", "speedup", "efficiency"]].to_string(index=False))
    return df


def exp_weak_scaling(cfg, n_runs):
    print("\n[weak_scaling]  (docs grow proportionally with threads)")
    maxt = get_num_threads()
    per_thread = max(200, cfg["n_docs"] // max(1, maxt))
    rows = []
    for th in _grid(maxt):
        n_docs = per_thread * th
        docs, _ = C.generate_corpus(n_docs=n_docs, vocab_size=cfg["vocab"],
                                    doc_len=cfg["doc_len"], seed=3)
        db = C.build_shingle_db(docs, k=cfg["k"])
        params = C.MinHashParams.create(cfg["n_hashes"], seed=2)
        MN.minhash_numba_parallel(db, params, n_threads=th)
        s, _ = C.benchmark(lambda d, p: MN.minhash_numba_parallel(d, p, n_threads=th),
                           db, params, n_runs=n_runs, warmup=1, label=f"weak t={th}")
        rows.append(dict(threads=th, n_docs=n_docs, mean_s=s.mean, std_s=s.std))
    df = pd.DataFrame(rows)
    t1 = df.loc[df.threads == 1, "mean_s"].values[0]
    df["weak_efficiency"] = t1 / df["mean_s"]
    df.to_csv(os.path.join(RESULTS, "weak_scaling.csv"), index=False)
    print(df.to_string(index=False))
    return df


def exp_layout(db, params, cfg, n_runs):
    print("\n[layout / vectorisation]")
    maxt = get_num_threads()
    variants = [
        ("dmajor_fastmath", dict(layout="dmajor", fastmath=True)),
        ("hmajor_fastmath", dict(layout="hmajor", fastmath=True)),
        ("dmajor_nofastmath", dict(layout="dmajor", fastmath=False)),
    ]
    rows = []
    for name, kw in variants:
        s, _ = C.benchmark(lambda d, p: MN.minhash_numba_parallel(d, p, n_threads=maxt, **kw),
                           db, params, n_runs=n_runs, warmup=2, label=name)
        rows.append(dict(variant=name, mean_s=s.mean, std_s=s.std, min_s=s.min))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "layout.csv"), index=False)
    print(df.to_string(index=False))
    return df


def exp_joblib_chunks(db, params, cfg, n_runs):
    print("\n[joblib chunk-size / backend]")
    maxt = _threads_available()
    rows = []
    for nc in [1, 2, 4, 8, 16, 32, 64, 128]:
        if nc > db.n_docs:
            continue
        s, _ = C.benchmark(lambda d, p: MJ.minhash_joblib(d, p, n_jobs=maxt, n_chunks=nc, backend="loky"),
                           db, params, n_runs=max(3, n_runs // 2), warmup=1,
                           label=f"loky nc={nc}")
        rows.append(dict(backend="loky", n_chunks=nc, mean_s=s.mean, std_s=s.std))
    for nc in [4, 32]:
        s, _ = C.benchmark(lambda d, p: MJ.minhash_joblib(d, p, n_jobs=maxt, n_chunks=nc, backend="threading"),
                           db, params, n_runs=max(3, n_runs // 2), warmup=1,
                           label=f"threading nc={nc}")
        rows.append(dict(backend="threading", n_chunks=nc, mean_s=s.mean, std_s=s.std))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "joblib_chunks.csv"), index=False)
    print(df.to_string(index=False))
    return df


def exp_lsh_sync(db, params, cfg, n_runs):
    print("\n[lsh synchronization]")
    sig = C.minhash_signatures_numpy(db, params)
    n_bands = C.optimal_bands(params.n_hashes, 0.5)
    ref = C.lsh_candidate_pairs(sig, n_bands)
    maxt = _threads_available()
    rows = []
    for strat, fn in LP.STRATEGIES.items():
        for th in _grid(maxt):
            got, elapsed = None, []
            for _ in range(2):
                fn(sig, n_bands, th)
            for _ in range(max(3, n_runs // 2)):
                t0 = time.perf_counter()
                got = fn(sig, n_bands, th)
                elapsed.append(time.perf_counter() - t0)
            assert got == ref, f"{strat} disagrees with reference!"
            rows.append(dict(strategy=strat, threads=th,
                             mean_s=float(np.mean(elapsed)),
                             std_s=float(np.std(elapsed))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "lsh_sync.csv"), index=False)
    print(df.to_string(index=False))
    return df


def exp_vectorization(db, params, cfg, n_runs):
    print("\n[vectorization: Python loop vs NumPy]")
    cap = min(db.n_docs, 400)
    sub = C.ShingleDB(db.vals[:db.offsets[cap]], db.offsets[:cap + 1], cap)
    s_py, _ = C.benchmark(C.minhash_signatures_python, sub, params,
                          n_runs=1, warmup=0, label="python_loop")
    s_np, _ = C.benchmark(C.minhash_signatures_numpy, sub, params,
                          n_runs=n_runs, warmup=1, label="numpy_vectorized")
    s_nb, _ = C.benchmark(MN.minhash_numba_serial, sub, params,
                          n_runs=n_runs, warmup=2, label="numba_serial")
    df = pd.DataFrame([
        dict(method="python_loop", mean_s=s_py.mean),
        dict(method="numpy_vectorized", mean_s=s_np.mean),
        dict(method="numba_serial", mean_s=s_nb.mean),
    ])
    df["speedup"] = df["mean_s"].iloc[0] / df["mean_s"]
    df.to_csv(os.path.join(RESULTS, "vectorization.csv"), index=False)
    print(df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small dataset for a fast dev/CI run")
    args = ap.parse_args()

    if args.quick:
        cfg = dict(n_docs=800, vocab=6000, doc_len=150, k=4, n_hashes=64,
                   n_dup_clusters=20)
        n_runs = 5
    else:
        cfg = dict(n_docs=6000, vocab=30000, doc_len=400, k=5, n_hashes=256,
                   n_dup_clusters=60)
        n_runs = 10

    print("=" * 68)
    print("  MinHash + LSH  --  parallel benchmark suite")
    print("=" * 68)
    print(f"  mode      : {'QUICK' if args.quick else 'FULL'}")
    print(f"  config    : {cfg}")
    print(f"  numba max threads: {get_num_threads()}   affinity cores: {_threads_available()}")
    print("=" * 68)

    db, params, truth = make_dataset(cfg)
    print(f"  dataset   : docs={db.n_docs}  nnz={db.nnz}  N={params.n_hashes}")
    MN.warmup(db, params)

    pd.DataFrame([cfg]).to_csv(os.path.join(RESULTS, "config.csv"), index=False)

    exp_impl_comparison(db, params, cfg, n_runs)
    exp_thread_scaling(db, params, cfg, n_runs)
    exp_weak_scaling(cfg, n_runs)
    exp_layout(db, params, cfg, n_runs)
    exp_joblib_chunks(db, params, cfg, n_runs)
    exp_lsh_sync(db, params, cfg, n_runs)
    exp_vectorization(db, params, cfg, n_runs)

    print("\nAll CSVs written to results/. Run plot_results.py to render figures.")


if __name__ == "__main__":
    main()
