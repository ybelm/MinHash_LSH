from __future__ import annotations

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
PLOTS = os.path.join(RESULTS, "plots")
os.makedirs(PLOTS, exist_ok=True)

C_SEQ = "#4C72B0"
C_PAR = "#55A868"
C_ALT = "#C44E52"
C_3 = "#8172B3"


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        print(f"[skip] {name} not found")
        return None
    return pd.read_csv(p)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name), dpi=120)
    plt.close(fig)
    print(f"[ok]  {name}")


def plot_impl():
    df = _load("impl_comparison.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = [C_ALT if "python" in i else C_PAR for i in df.impl]
    ax.bar(df.impl, df.mean_s, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Mean time (s) [log]")
    ax.set_title("MinHash: implementation comparison")
    for x, (m, s) in enumerate(zip(df.mean_s, df.speedup_vs_python)):
        ax.text(x, m, f"{s:.0f}x", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, "01_impl_comparison.png")


def plot_thread_scaling():
    df = _load("thread_scaling.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(df.threads, df.speedup, marker="o", color=C_PAR, linewidth=2, label="Measured")
    ax.plot(df.threads, df.threads, linestyle="--", color="gray", label="Ideal (linear)")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup vs 1 thread")
    ax.set_title("Numba prange: strong scaling")
    ax.set_xticks(df.threads)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, "02_thread_scaling.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(df.threads, df.efficiency, marker="s", color=C_SEQ, linewidth=2)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Parallel efficiency")
    ax.set_title("Numba prange: parallel efficiency")
    ax.set_xticks(df.threads)
    ax.set_ylim(0, 1.15)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    _save(fig, "03_efficiency.png")


def plot_weak_scaling():
    df = _load("weak_scaling.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(df.threads, df.weak_efficiency, marker="o", color=C_PAR, linewidth=2, label="Measured")
    ax.axhline(1.0, linestyle="--", color="gray", label="Ideal (constant time)")
    ax.set_xlabel("Threads (problem size grows proportionally)")
    ax.set_ylabel("Weak-scaling efficiency  T(1)/T(p)")
    ax.set_title("Numba prange: weak scaling")
    ax.set_xticks(df.threads)
    ax.set_ylim(0, 1.15)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, "04_weak_scaling.png")


def plot_layout():
    df = _load("layout.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(df.variant, df.mean_s, yerr=df.std_s, color=[C_PAR, C_ALT, C_SEQ], capsize=4)
    ax.set_ylabel("Mean time (s)")
    ax.set_title("Signature layout & vectorisation (cache / SIMD)")
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, "05_layout.png")


def plot_joblib():
    df = _load("joblib_chunks.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    loky = df[df.backend == "loky"]
    ax.plot(loky.n_chunks, loky.mean_s, marker="o", color=C_PAR, linewidth=2, label="loky (processes)")
    thr = df[df.backend == "threading"]
    if len(thr):
        ax.scatter(thr.n_chunks, thr.mean_s, color=C_ALT, zorder=5, label="threading (GIL)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of chunks [log2]")
    ax.set_ylabel("Mean time (s)")
    ax.set_title("joblib: chunk-size & backend")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, "06_joblib_chunks.png")


def plot_lsh_sync():
    df = _load("lsh_sync.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"reduction": C_PAR, "shared_lock": C_ALT, "coarse_lock": C_SEQ}
    for strat in df.strategy.unique():
        d = df[df.strategy == strat]
        ax.plot(d.threads, d.mean_s, marker="o", linewidth=2,
                color=colors.get(strat, C_3), label=strat)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Mean time (s)")
    ax.set_title("LSH build: synchronization strategy")
    ax.set_xticks(sorted(df.threads.unique()))
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, "07_lsh_sync.png")


def plot_vectorization():
    df = _load("vectorization.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(df.method, df.mean_s, color=[C_ALT, C_PAR, C_SEQ])
    ax.set_yscale("log")
    ax.set_ylabel("Mean time (s) [log]")
    ax.set_title("Vectorisation: Python loop vs NumPy vs Numba")
    for x, (m, s) in enumerate(zip(df.mean_s, df.speedup)):
        ax.text(x, m, f"{s:.0f}x", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.4)
    _save(fig, "08_vectorization.png")


if __name__ == "__main__":
    print("Rendering figures from results/ ...")
    plot_impl()
    plot_thread_scaling()
    plot_weak_scaling()
    plot_layout()
    plot_joblib()
    plot_lsh_sync()
    plot_vectorization()
    print(f"\nFigures written to {os.path.relpath(PLOTS)}")
