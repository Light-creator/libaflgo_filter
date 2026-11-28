# MLP modes and restart checkpoints

MLP is disabled by default. Enable it with `--enable-mlp --ml-library PATH`.
The library is not loaded in PassAll mode. `--intermediate-nodes PATH` remains
required for campaign identity. SymCC is independently controlled by
`--enable-symcc --symcc-binary PATH --symcc-work-dir PATH`.
Use separate output directories for different configurations.

`data.csv` is appended, with a header only for an empty file. It is still located
in the working directory: use a separate working directory per campaign. Rows
are flushed at the end of each mutational stage. An abrupt crash can lose buffered
rows and a replayed stage can produce duplicate rows.

Runtime checkpoints are disabled by default. Enable save and automatic restore with
`--enable-checkpoint`. Without this flag, existing checkpoints are neither read nor
modified, configuration files are not hashed for checkpoint identity, and the model
and tracker start fresh after a restart. CSV append behavior is independent of this
flag. LibAFL's own restarting machinery remains active.

Runtime checkpoints live under OUTPUT/runtime-checkpoint. At stage boundaries,
once at least 30 seconds have elapsed, a new generation is written, then the
`latest` pointer is atomically replaced. The previous completed generation is
removed only after publication of the next. An interrupted write does not replace
`latest`; incomplete directories may remain after an abrupt termination.

Checkpoints contain Burn model weights, collected samples, calibration data,
phase, threshold, learning target, RNG and filter counters, plus SymCC candidate
IDs, attempted byte strings, and visited node history. Save/load requires the
new `model_checkpoint` symbol in the model library. Rebuild ml_model first.
The corpus, executor and scheduler remain managed by LibAFL's existing restarter.
Timers get a fresh grace interval on restore; downtime is not stagnation.
The diagnostic event counters are reset on restart.

Restore is attempted only when LibAFL actually supplies a restored state. A fresh
state invalidates the checkpoint pointer to avoid reusing old CorpusIds. Merely
starting a new command with an old output directory does NOT promise restoration
of the previous campaign. Full manual stop/resume is not implemented here.

Configuration identity includes MLP mode, node CSV contents and model-library
contents. Changed configuration requires a new output directory. The checkpoint
format is private, versioned by runtime-checkpoint-v2, and is not a portable
interchange format. Model/data files must come from the same completed generation.
Do not load checkpoint files from untrusted sources.

The nominal interval is 30 seconds, checked at stage boundaries, so long stages
can increase checkpoint age. A crash can repeat work since the latest checkpoint,
including some SymCC attempts. These are periodic checkpoints, not exactly-once
transactional recovery and not power-loss durability guarantees.
