> **Historical pre-ARROW artifact.** PIVOT names and schemas below identify the
> sealed implementation lineage and are intentionally preserved.

# Stage-B single-GPU serial matrix queue

`tools/run_stageb_serial_matrix_queue.py` is the durable queue layer for the
sealed Table-C and Table-B/D launchers. It accepts an explicit ordered list of
`ROW:SEED` IDs, routes each ID through the launcher that registered it, and
runs one detached job at a time.

It does not call `main.py`, change a config, resume a failed experiment, adopt
an existing output, or consider an already-existing output successful.

## Completion contract

An item can release the next item only when all of the following are true:

1. the existing launcher reports the detached orchestrator `completed`;
2. PID/start-time evidence says that orchestrator is no longer running;
3. its detached `launch.json`, `status.json`, and retained preflight plan bind
   exactly one expected root and exactly the queued run ID;
4. the preflight plan records `output_dir_fresh_at_plan=true`;
5. `sequence_manifest.json` is `completed` for the same run ID;
6. its completed phases exactly match its planned phases in order; and
7. every phase has a `launch_manifest.json` marked `completed` and a
   `postflight.json` marked `passed` for that run ID. Table-D phase IDs must
   also match.

A process exit code, a checkpoint filename, console text, or an orchestration
status alone is insufficient. Missing, malformed, unknown, or contradictory
evidence fails the current item and leaves every later item pending.

## Durable state and exclusivity

`create` writes `QUEUE_DIR/queue.json`. Its immutable plan contains:

- ordered run IDs and token/paper routing;
- exact hashes of both runner source files;
- the runner Python path;
- the selected single CUDA device;
- a snapshot of only the runtime environment variables consumed by the two
  runners; and
- the shared GPU lease path.

The plan has a canonical SHA-256 checked on every read. Queue updates use
atomic replace plus `fsync`. A queue-local advisory lock permits one supervisor
for that queue. Before the first launch, a durable per-GPU lease is written;
the same queue retains it between items and removes it only after the entire
queue passes completion verification. A second queue using the same canonical
GPU key stops before creating a job.

The lease coordinates queues created by this tool. It cannot control a
training process launched directly or by an older launcher invocation. Confirm
that any such experiment is terminal before starting `run`; do not start the
queue alongside an independently running Stage-B job.

If the queue supervisor exits, the current runner job remains detached. On the
next `run`, the queue finds the one job under that item's unique orchestration
root, or recovers the short-lived detach launcher by its exact command and
Linux process identity. It never launches a duplicate when more than one
candidate exists. A failed queue retains its GPU lease deliberately; inspect
the child status and failure evidence before any manual lease cleanup.

## Create an explicit queue

Set the complete training contract when creating the queue. Later `run`
invocations use this captured snapshot, so changing the caller's environment
does not silently change later rows.

```bash
PIVOT_BATCH_SIZE=40 \
PIVOT_MAX_TRAIN_ITERS=1000 \
PIVOT_ITER_CHECKPOINT_INTERVAL=1000 \
PIVOT_NUM_WORKERS=2 \
PIVOT_CUDA_VISIBLE_DEVICES=0 \
PIVOT_TOKEN_OUTPUT_ROOT=outputs/paper_cvpr_v1/token_ablation_b40_u1000 \
python tools/run_stageb_serial_matrix_queue.py create \
  outputs/paper_cvpr_v1/queues/table_c_remaining_b40_u1000 \
  --run-id L0:42 \
  --run-id L0:73 \
  --run-id L1:17 \
  --run-id L1:42 \
  --run-id L1:73
```

`--run-id` is repeatable and its order is the queue order. The current version
intentionally has no implicit `--all`: the exact paper run list must be visible
in the plan. Run IDs are validated from the launchers' own `list --json`
inventories.

The supported catalogs are:

- Table C: `L0` through `L10`, seeds `17`, `42`, `73` (33 runs), routed to
  `run_stageb_token_ablation_matrix.py`;
- Table B: `D0`, `D1`, `D2`, `D3`, `D2m`, `D3m`, all three seeds (18 runs);
  and
- Table D: `S0`, `S1`, `S2`, `S3`, `S2F`, all three seeds (15 runs), routed
  with Table B to `run_stageb_paper_ablation_matrices.py`.

Use separate queues when a sealed comparison block needs a different batch
size, output root, diagnostic interval, or update budget. The shared GPU lease
still serializes those queues, but one must complete before the next is run.

Never include a run whose formal output or detached job already exists. The
underlying launcher requires a fresh output and the queue does not silently
skip or import prior work.

## Run, stop, and resume

The normal supervisor runs until all items pass or one item fails:

```bash
python tools/run_stageb_serial_matrix_queue.py run \
  outputs/paper_cvpr_v1/queues/table_c_remaining_b40_u1000
```

Restart the exact same command after a terminal/session interruption. No
special resume flag is needed. For a scheduler or manual controller, one
durable transition at a time is available:

```bash
python tools/run_stageb_serial_matrix_queue.py run QUEUE_DIR --once
```

`--once` may reserve, start the detach launcher, bind its detached job, poll
and reconcile it, or complete one item. Repeating it is equivalent to the
continuous supervisor.

`run` is the only queue command that can start training. `create`, `status`,
and `verify` do not start a GPU job.

## Inspect and verify

`status` is read-only. If an item has a bound detached job, it delegates to the
correct existing runner's read-only `status` command.

```bash
python tools/run_stageb_serial_matrix_queue.py status QUEUE_DIR
```

The continuous supervisor uses that runner's `reconcile` command before acting
on a stale status. At the end, independently re-read every sequence and phase
postflight:

```bash
python tools/run_stageb_serial_matrix_queue.py verify QUEUE_DIR
```

`verify` exits nonzero for an incomplete queue, a missing phase, a failed or
changed postflight, a runner source hash drift, or any binding mismatch.

## Failure policy

The first failure atomically marks the current item and queue `failed`; later
items remain `pending` and no further runner is launched. The failure record
contains the run ID, transition phase, and exact contract error. The lease is
retained because an ambiguous failure must not permit another queue to assume
the GPU is free.

There is no automatic retry or output deletion. Diagnose the detached job,
preserve its evidence, choose a new fresh output root, and create a new explicit
queue only after proving the old child process is terminal. Lease deletion is
therefore a conscious operator action rather than a queue recovery heuristic.

## CPU-only tests

The queue tests use temporary fake launchers and never start training:

```bash
python -m unittest tests.test_stageb_serial_matrix_queue -v
```

They cover the real 66-ID catalog, ordered success including S3's three
phases, supervisor restart, cross-queue GPU lease exclusion, explicit child
failure, missing postflight, runner-source drift, and immutable-plan tampering.
