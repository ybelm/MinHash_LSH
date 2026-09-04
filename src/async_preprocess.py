from __future__ import annotations
import asyncio
import time
import numpy as np


# I/O-bound stage: loading many documents
def _read_one_sync(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_documents_sync(paths: list[str]) -> list[str]:
    return [_read_one_sync(p) for p in paths]


async def _read_one_async(path: str) -> str:
    return await asyncio.to_thread(_read_one_sync, path)


async def load_documents_async(paths: list[str]) -> list[str]:
    return await asyncio.gather(*(_read_one_async(p) for p in paths))


def load_documents_sync_latency(paths: list[str], latency: float) -> list[str]:
    out = []
    for p in paths:
        time.sleep(latency)             # blocking wait: latencies add up
        out.append(_read_one_sync(p))
    return out


async def _read_one_slow(path: str, latency: float) -> str:
    await asyncio.sleep(latency)        # non-blocking wait: loop starts others
    return await asyncio.to_thread(_read_one_sync, path)


async def load_documents_async_latency(paths: list[str], latency: float) -> list[str]:
    return await asyncio.gather(*(_read_one_slow(p, latency) for p in paths))


def _cpu_task(n: int) -> float:
    # A deliberately CPU-heavy pure-Python loop (no I/O, no GIL release)
    acc = 0.0
    for i in range(n):
        acc += (i * 2654435761) % 1000003
    return acc

async def _cpu_task_async(n: int) -> float:
    return _cpu_task(n)

async def run_cpu_tasks_async(n: int, k: int):
    return await asyncio.gather(*(_cpu_task_async(n) for _ in range(k)))

def run_cpu_tasks_sync(n: int, k: int):
    return [_cpu_task(n) for _ in range(k)]



# Demo / measurement
def demo(tmpdir: str = "/tmp/minhash_async_demo", n_files: int = 40):
    import os
    os.makedirs(tmpdir, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n_files):
        p = os.path.join(tmpdir, f"doc_{i}.txt")
        if not os.path.exists(p):
            words = rng.integers(0, 50000, size=4000)
            with open(p, "w") as f:
                f.write(" ".join(map(str, words.tolist())))
        paths.append(p)

    print("--- I/O-bound: loading %d files (warm local cache) ---" % n_files)
    t0 = time.perf_counter(); load_documents_sync(paths); t_sync = time.perf_counter() - t0
    t0 = time.perf_counter(); asyncio.run(load_documents_async(paths)); t_async = time.perf_counter() - t0
    print(f"  sequential reads: {t_sync*1000:8.2f} ms")
    print(f"  asyncio.gather: {t_async*1000:8.2f} ms   (speedup {t_sync/t_async:.2f}x)")
    print("  -> tiny cached files: overhead dominates, async doesn't help here.")

    lat = 0.005
    print(f"\n--- I/O-bound with {lat*1000:.0f} ms/file modelled latency (slow disk/network) ---")
    t0 = time.perf_counter(); load_documents_sync_latency(paths, lat); t_sync_l = time.perf_counter() - t0
    t0 = time.perf_counter(); asyncio.run(load_documents_async_latency(paths, lat)); t_async_l = time.perf_counter() - t0
    print(f"  sequential reads: {t_sync_l*1000:8.2f} ms")
    print(f"  asyncio.gather: {t_async_l*1000:8.2f} ms   (speedup {t_sync_l/t_async_l:.2f}x)")

    print("\n--- CPU-bound: %d tasks (asyncio cannot parallelise these) ---" % 8)
    n = 2_000_000
    t0 = time.perf_counter(); run_cpu_tasks_sync(n, 8); t_sync = time.perf_counter() - t0
    t0 = time.perf_counter(); asyncio.run(run_cpu_tasks_async(n, 8)); t_async = time.perf_counter() - t0
    print(f"  sequential: {t_sync*1000:8.2f} ms")
    print(f"  asyncio.gather: {t_async*1000:8.2f} ms   (speedup {t_sync/t_async:.2f}x)")
    print("  -> asyncio ~1x on CPU work")
    return {"io_sync": t_sync, "io_async": t_async}


if __name__ == "__main__":
    demo()
