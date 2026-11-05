"""
seiscamp.families -- stage 1 of the post-SCAMP workflow.

Turns the per-station SCAMP detection lists written by the step 1-3 pipeline
into candidate events and then into waveform families:

1. :mod:`seiscamp.families.associate` -- collapse the per-station window
   pairs into occurrences and associate them across stations (levels of the
   borehole string) into candidate events.
2. :mod:`seiscamp.families.windows` -- cut one three-component window per
   station around every event from the raw daily archive, reading each
   station-day only once.
3. :mod:`seiscamp.families.similarity` -- network cross-correlation of every
   event pair at a common lag, and alignment of events to a reference.
4. :mod:`seiscamp.families.cluster` -- hierarchical (average-linkage)
   clustering of the correlation matrix into near-repeating and tight
   families, with per-family timing statistics.
5. :mod:`seiscamp.families.plotting` -- the correlation-matrix figure and
   one waveform figure per family.

The scripts in ``scripts_frs/stage1/`` chain these for one day; see the
README there. Every function takes plain numpy / pandas objects so the
pieces can also be used from a notebook.
"""
