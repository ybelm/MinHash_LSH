PY = python3
SRC = src

setup:
	$(PY) -m pip install -r requirements.txt

test:
	cd $(SRC) && $(PY) test_core.py

validate:
	cd $(SRC) && $(PY) minhash_numba.py
	cd $(SRC) && $(PY) minhash_joblib.py
	cd $(SRC) && $(PY) lsh_parallel.py
	cd $(SRC) && $(PY) async_preprocess.py


validate-gpu-sim:
	cd $(SRC) && NUMBA_ENABLE_CUDASIM=1 $(PY) minhash_cuda.py

experiments-quick:
	cd $(SRC) && $(PY) run_experiments.py --quick

experiments:
	cd $(SRC) && $(PY) run_experiments.py

plots:
	cd $(SRC) && $(PY) plot_results.py

all: validate experiments plots

clean:
	rm -rf results/*.csv results/plots
	find . -name __pycache__ -type d -exec rm -rf {} +
