# Running the FRS pipeline on Glados

SLURM scripts for the `glados-gpu` partition (24 nodes x 4 GTX 1080 Ti). The
pipeline itself is unchanged; these only set up the environment and drive
`scripts_frs/run_frs_day.sh` with the Glados paths.

1. Build once (GPU node, ~30-60 min including the conda solve):
   ```bash
   cd ~/seismo-pyscamp/scripts_frs/glados && mkdir -p logs && sbatch build_pyscamp.slurm
   ```
   Creates the `seismo-pyscamp` env from `environment-glados.yml` and compiles
   pyscamp with conda-forge's CUDA 12.0 for compute capability 6.1 only. The log ends with a
   GPU self-join test; `gpu_supported: True` is the thing to check.
2. Benchmark one day end to end:
   ```bash
   sbatch benchmark_day.slurm 20220719 100 0.5
   ```
   Writes `~/scamp_runs/bench_<day>_<fs>Hz_<jobid>/` with the detections and a
   `timing_step1.csv` (per-trace SCAMP seconds at 100 Hz single, plus one trace
   at 250 Hz single and one at 100 Hz double) for the cost model.

Never run these steps on the login node; both scripts are meant for `sbatch`.
