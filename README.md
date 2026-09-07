# seismo-pyscamp

Scripts for using the matrix profile algorithm SCAMP for earthquake similarity search.

## Setup

```bash
conda env create -f environment.yml
conda activate seismo-pyscamp
```

This builds the latest `pyscamp` straight from https://github.com/zpzim/SCAMP and
installs `seiscamp` in editable mode, so edits under `seiscamp/` take effect
without reinstalling.

Verify:

```bash
python -c "import pyscamp, seiscamp; print(pyscamp.gpu_supported())"
```

## GPU support

SCAMP's GPU backend is **CUDA only** — there is no Metal or MPS backend, so on a
Mac (Apple Silicon or Intel) `pyscamp` builds and runs CPU-only, multithreaded.
`pyscamp.gpu_supported()` returns `False`, and passing `gpus=[...]` aborts with a
failed assertion rather than a clean error.

`scripts_frs/python_scripts/step1_pyscamp_1chan.py` checks `gpu_supported()` and
drops the `gpus` argument when there is no CUDA device, so the same script runs
on the cluster and on a laptop. `scripts_frs/python_scripts/old/call_pyscamp.py`
still hardcodes `gpus=[0, 1, 2, 3]` and will abort locally.

CPU throughput is fine for exploratory work — a 50k-sample self-join with a
200-sample window takes well under a second on an M-series chip — but full-day
traces at high sampling rates are still much faster on the CUDA cluster.

## Optional: the `picking/` scripts

`picking/` uses [PhasePApy](https://github.com/austinholland/PhasePApy), which
ships no `setup.py` and so cannot be pip-installed. Clone it and put it on the
path:

```bash
git clone https://github.com/austinholland/PhasePApy.git
export PYTHONPATH="$PWD/PhasePApy:$PYTHONPATH"
```

Some of those scripts also call APIs removed from modern ObsPy (for example
`obspy.core.util.gps2DistAzimuth`, now `obspy.geodetics.gps2dist_azimuth`).
