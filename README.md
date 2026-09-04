# Parallel Near-Duplicate Search with MinHash + LSH (Python)

Efficiently finding pairs of similar documents in a large collection, using
**MinHashing** to estimate Jaccard similarity and **Locality-Sensitive Hashing
(LSH)** to avoid the quadratic all-pairs comparison.

The project implements a **sequential baseline** and several **parallel Python**
variants, and benchmarks them to demonstrate the parallelisation methodologies
covered in the course: data parallelism, loop optimisation, vectorisation
(SIMD), scheduling, and synchronisation.

```
tokens --shingling--> shingle sets --MinHash--> signature matrix
      --LSH banding--> candidate pairs --verify--> similar pairs
```

The estimator identity that makes it work:
`P(minhash_i(A) == minhash_i(B)) = Jaccard(A, B)`.

---


## Project structure

```
src/
├── core.py              # data structures, reference MinHash (numpy + pure python),
│                        # shingling, LSH banding, verification, dataset generator,
│                        # benchmark harness (wall + CPU time)
├── minhash_numba.py     # Numba kernels: serial, parallel, layout & fastmath variants
├── minhash_joblib.py    # process/thread-parallel MinHash across document chunks
├── lsh_parallel.py      # parallel LSH: reduction vs shared-lock vs coarse-lock
├── minhash_cuda.py      # Numba CUDA kernels (global + shared memory), GPU wrapper
├── async_preprocess.py  # asyncio I/O demo (and the CPU-bound anti-pattern)
├── run_experiments.py   # runs the whole suite -> results/*.csv
├── plot_results.py      # results/*.csv -> results/plots/*.png
└── test_core.py         # correctness tests for the reference pipeline
results/                 # CSVs and figures (generated)
report/                  # write-up
```

---

## Requirements

Python 3.10+ and a multi-core CPU

```bash
make setup           # pip install -r requirements.txt
```

Mandatory: `numpy`, `numba`, `joblib`, `matplotlib`, `pandas`.
Optional: `tbb` (for the Numba TBB threading-layer experiment).

---

## Running

```bash
make test                # correctness of the reference pipeline
make validate            # every parallel variant produces signatures == reference
make experiments-quick   # small dataset, runs in seconds (sanity / CI)
make experiments         # full dataset (sequential baseline > 10 s), writes CSVs
make plots               # render all figures
make all                 # validate + experiments + plots
```

---

## Configuration

Dataset and benchmark parameters live at the top of `run_experiments.py`:

| Parameter | Full | Quick | Meaning |
|-----------|------|-------|---------|
| `n_docs` | 6000 | 800 | number of documents |
| `vocab` | 30000 | 6000 | vocabulary size (Zipfian) |
| `doc_len` | 400 | 150 | mean document length (tokens) |
| `k` | 5 | 4 | shingle size |
| `n_hashes` | 256 | 64 | MinHash permutations (signature length N) |
| `n_dup_clusters` | 60 | 20 | injected near-duplicate clusters (ground truth) |