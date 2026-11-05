# Stage 1: from SCAMP detections to waveform families

Post-processing of a `scripts_frs` run (step 1–3 outputs) into candidate
events and waveform families, one day at a time. Library code lives in
`seiscamp.families`; these scripts only parse arguments and write files.

| Script | Reads | Writes | What it does |
|---|---|---|---|
| `s1_associate.py` | `RUN/detections_combined/<DAY>*COMBINED*.csv` | `OUT/<DAY>/occurrences.csv`, `event_stations.csv`, `events.csv` | Collapses SCAMP window pairs into per-station occurrences (5 s), associates them across ≥ 3 stations within 1 s into events. |
| `s2_cut_windows.py` | `events.csv`, raw archive | `OUT/<DAY>/windows.npz` | Reads each station-day once (decimated to 250 Hz), cuts 1 s + 3 s windows, band-pass 5–60 Hz. |
| `s3_cluster.py` | `windows.npz`, `events.csv` | `cc.npy`, `lag.npy`, `families.csv`, `family_summary_{near,tight}.csv`, `figures/` | Network correlation at a common lag (±1 s), average-linkage clustering cut at 0.7 (near) and 0.9 (tight), per-family timing statistics, correlation-matrix and per-family waveform figures. |
| `stage1_day.slurm` | | | Runs the three for one day (or one day per array task) on a compute node. |

```bash
cd ~/seismo-pyscamp/scripts_frs/stage1 && mkdir -p logs
sbatch stage1_day.slurm ~/scamp_runs/bench_20220719_100Hz_3869428 20220719
```

Never run `s2`/`s3` on the login node: `s2` holds a 3.6 GB day cache and
`s3` is CPU-bound.

## Design choices worth knowing

- **Why not step 4's groups.** Step 4 links events through any matrix-profile
  peak, which is single-linkage clustering and chains dissimilar events
  (median within-family correlation 0.45 in the discovery campaign; one
  4,731-member blob on BH015 on 2022-07-19). Here families come from the
  full pairwise correlation matrix with average linkage.
- **Greedy, anchored grouping** in both the collapse and the association:
  a group never extends more than the tolerance past its first member, so
  dense noise cannot chain.
- **One vertical string.** BH001–BH024 share one surface coordinate; the
  association tolerance (1 s) reflects SCAMP window jitter, not moveout, and
  "3 stations" means three levels of the string.
- **Band 5–60 Hz, not 15–200 Hz.** The wider band was noise-dominated for
  the first template scans on this array.
- **`context` column.** `events.csv` carries `context = "unknown"`. The
  calibrated classifier in `cmc-nonlinloc/template_matching` (quiet / survey
  / vibroseis / machinery) is not installed on Glados; run it where it exists
  and merge its labels on `event_id` before trusting any family as seismicity.
- **Families are per day** at this stage. Cross-day families need template
  matching (stage 2), because a SCAMP self-join only sees pairs within one
  trace-day.
