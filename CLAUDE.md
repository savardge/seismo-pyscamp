# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for earthquake similarity search using the SCAMP matrix profile algorithm (`pyscamp`). A small installable package (`seiscamp`) provides shared library code; the rest is dataset-specific pipeline scripts meant to run on SLURM clusters (FRS and Meager datasets). There are no tests and no linting — this is exploratory scientific code.

## Setup

```bash
conda env create -f environment.yml
conda activate seismo-pyscamp
```

This builds `pyscamp` from source (github.com/zpzim/SCAMP) and installs `seiscamp` in editable mode (`pip install -e .`), so edits under `seiscamp/` take effect without reinstalling.

Verify: `python -c "import pyscamp, seiscamp; print(pyscamp.gpu_supported())"`

## GPU constraint (important)

SCAMP's GPU backend is **CUDA only**. On macOS `pyscamp` runs CPU-only and `pyscamp.gpu_supported()` returns `False`; passing `gpus=[...]` **aborts with a failed assertion**, not a clean error. Any code calling `pyscamp.selfjoin`/`join` must gate the `gpus` kwarg on `pyscamp.gpu_supported()` — see `get_mp()` in [step1_pyscamp_1chan.py](scripts_frs/python_scripts/step1_pyscamp_1chan.py) for the pattern. `scripts_frs/python_scripts/old/call_pyscamp.py` still hardcodes `gpus=[0,1,2,3]` and will abort locally.

## Architecture

### `seiscamp/` — the installed package
- `data.py` — waveform I/O: reads daily miniSEED from a year/julian-day directory tree (`get_stream_1day`, `get_stream_days`), resamples, merges, fills gaps with noise, applies gain. Also `get_filename_root()`, which defines the **filename convention that carries pipeline metadata**: `{start}_{end}_{station}_{channel}_{fs}Hz_win{sublen}samp` — downstream steps parse station/channel/times/fs/window length back out of filenames rather than passing them explicitly.
- `MPLib.py` — `MatrixProfile` class (wraps MP values + match indices + trace metadata) and detection helpers (interval merging, group finding).
- `GraphLib.py` — DFS connected-components on an undirected graph, used to cluster detections whose matching windows link them.
- `util.py` — running statistics (std/median/mean) and zero-fill time-gap detection on ObsPy traces.

Note: modules use `from seiscamp.X import *` star-imports of each other; `data.py` re-exports `util`'s names, etc.

### Pipeline (`scripts_frs/` is the current/reference one)

Four numbered steps in `scripts_frs/python_scripts/`, chained by files on disk and the filename convention above:

1. `step1_pyscamp_1chan.py <station> <YYYYMMDD> <num_days> <sublen_sec> <channel> <fs>` — reads one channel-day(s) of waveforms, runs `pyscamp.selfjoin` (Pearson, double precision), writes `*_mp.npy` + `*_ind.npy`. Skips if output already exists.
2. `step2_extract_detections.py <mp_filename>` — thresholds the matrix profile against a running median/MAD to extract detection windows → `*_detections.csv`.
3. `step3_combine.py <prefix> <suffix> <chan_prefix>` — joins per-component (N/E/Z) detection CSVs into one.
4. `step4_group.py <combined_csv>` — clusters detections into event families via connected components (networkx + eqcorrscan alignment) → `*_times.csv`.

SLURM wrappers (`run_pyscamp.slurm`, `mpdetect.slurm`, `run_pyscamp_ARC.slurm`) drive these as job arrays reading station/day lists from `input_files/`. `check_if_*.sh` audit which array indices completed.

**Hardcoded cluster paths**: the scripts embed absolute paths like `/home/genevieve.savard/seismo-pyscamp/...` (output dirs) and `/home/gilbert_lab/cami_frs/all_daily_symlinks` (waveform archive, also `DEFAULT_WF_DIR` in `seiscamp/data.py`). These only resolve on the cluster; to run locally you must override them.

### Other directories
- `scripts_meager/` — older variant of the same pipeline for the Meager dataset (gitignored but present locally).
- `picking/` — legacy phase-picking/association workflow using PhasePApy (vendored under `picking/phasepapy/`). Requires cloning PhasePApy onto `PYTHONPATH` per the README; some scripts call APIs removed from modern ObsPy (e.g. `obspy.core.util.gps2DistAzimuth` → `obspy.geodetics.gps2dist_azimuth`).
- `notebooks/` — exploratory Jupyter notebooks (gitignored, kept out of tracking deliberately).
