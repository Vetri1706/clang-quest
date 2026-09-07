# Distributed reference implementation

The project verifies an analytic 3D MLP, replicated data-parallel Transformer,
actual 3D MLA decoder, and automatic ZeRO Transformer across data ranks.
`rawllm/zero.py` provides the sharding primitive; `rawllm/zero_model.py` supplies
operation-level rematerialization so the autograd graph retains no gathered
weights. `rawllm/sharded_checkpoint.py` adds collective commit and fixed-membership
recovery through a shared POSIX filesystem. The 3D implementation still uses a
fixed tiny topology. Automatic ZeRO and the 3D scheduler are separate execution
paths. There is no GPU transport, RDMA, native BF16, elastic membership, or
arbitrary partition planner.

## Commands and measured scope

```sh
python distributed_demo.py --output results/distributed_demo.json
python distributed_train.py --output results/distributed_transformer.json --checkpoint results/distributed_transformer_checkpoint
python distributed_llm3d.py --output results/distributed_llm3d.json --weights results/distributed_llm3d_weights
python -m unittest tests.test_distributed -v
```

The eight-process MLP run compared each reconstructed full weight and parameter
gradient against an independent dense analytic backward pass. Its original
recorded maximum gradient discrepancy was approximately `6.94e-18`; loss,
updated weights, and data replicas agreed exactly. Runtime measurements are
observations on the development host, not throughput promises.

The real Transformer run used 8,556 parameters, two ranks, two sequences per
rank, eight predicted tokens per sequence, and three AdamW updates. Its maximum
gradient discrepancy from a combined-batch reference was approximately
`5.55e-17`; updated weights and replicas agreed exactly. FP64 model/autograd
arithmetic permits a strict comparison, while AdamW uses FP32 masters and
moments. Each local batch has equal target-token mass; mean rank gradients
therefore equal global mean token gradients. Unequal masks or batch sizes would
require a global sum of weighted gradient numerators divided by the global
target mass. The generated checkpoint contains real updated weights and
optimizer state. This fixture does not establish language or C++ tutoring skill.

The actual 3D Transformer run used 12,652 global parameters with an untied
embedding and output head. Ranks owned 5,374 or 5,390 parameters after their
temporary full initialization was discarded. These counts include explicitly
replicated latent projections and normalization scales. Eight ranks completed
two AdamW updates with two microbatches per data rank. Maximum reconstructed
gradient discrepancy was approximately `5.55e-17`; reconstructed weights,
data replicas, and tensor replicas agreed exactly. The measured runtime was
approximately 0.95 seconds on the development host. The resulting model export
contains trained weights and configuration, without optimizer-resume state.

## Transformer attention, FFN, and pipeline partitioning

The actual Transformer path splits attention heads across tensor ranks. Query,
content-key, and value up-projection matrices are split by output columns. The
attention output projection is split by its matching input rows, and the
partial output projections are summed. SwiGLU gate/up matrices split by hidden
output features, with the down projection split by matching input rows. All
attention heads and FFN features participate in the computation.

Two explicit autograd maps express the communication derivatives:

\[
\operatorname{copy}_{TP}:\quad x\mapsto x,\quad
\bar x=\sum_{t=0}^{T-1}\bar x_t;
\qquad
\operatorname{reduce}_{TP}:\quad y=\sum_{t=0}^{T-1}y_t,\quad
\bar y_t=\bar y.
\]

The copy map occurs after attention RMSNorm, before the replicated query/key/
value latent projection paths, and after FFN RMSNorm before the partitioned
SwiGLU computation. This makes the normalization-scale and residual-input
gradients complete within every replica. The reduce map occurs after each
partial attention output projection and partial FFN down projection.

Query-down, query-latent norm, key/value-down, key/value-latent norm, and shared
rotary-key projection parameters remain replicated within each tensor group.
Their gradients must be explicitly summed across tensor ranks because each
local loss path contains only a subset of heads. Attention/FFN RMSNorm scales
outside the copy maps already receive full gradients and must not be summed
again. Embedding, final norm, and output head also already receive complete
gradients on their owning pipeline stages. An extra sum for those parameters
would multiply their updates by the tensor-parallel degree; the full-parameter
dense reference comparison checks this distinction.

Pipeline stage zero owns the embedding and first Transformer block. Stage one
owns the second block, final normalization, and untied vocabulary projection.
Stage zero sends its full residual activation to the matching tensor rank of
stage one. The receiver wraps that activation in a differentiable Tensor leaf.
After local backward, it sends the leaf gradient to the corresponding stage-zero
rank, which continues its own backward with that explicit cotangent. This is a
sequential microbatch schedule. All local parameter gradients are subsequently
averaged over data ranks and updated by their locally owned AdamW instance.

Each rank retains only its pipeline-stage parameters and tensor shards after
initialization. Embedding/head, latent projections, and scales remain replicated
where stated. Data replicas each own full Adam state for their local partition.
Consequently this is real DP/PP/TP composition, while the separately tested
ZeRO Stage 3 primitive is not automatically applied to these stage parameters.

## Rank mapping and schedules

For dimensions `D`, `P`, `T`, the global rank is

\[
r(d,p,t)=(dP+p)T+t,\qquad 0\le d<D,\;0\le p<P,\;0\le t<T.
\]

Data groups vary `d` with `p,t` fixed. Pipeline groups vary `p`; tensor groups
vary `t`. With `(D,P,T)=(2,2,2)`, rank 5 has coordinates `(1,0,1)`, data group
`(1,5)`, pipeline group `(5,7)`, and tensor group `(4,5)`.

The MLP dimensions are `6 -> 8 -> 6 -> 8 -> 3`. Each pipeline stage owns two
matrices. Tensor rank `t` owns four output columns of the first matrix and the
matching four input rows of the second. Forward first computes local column
projections and `tanh`, then sums partial row projections with an all-reduce.
Backward reduces partial input gradients for the column-parallel matrix. Stage
zero applies a further `tanh` and sends its activation to stage one. Stage one
computes mean-square loss and sends the activation gradient back. Every tensor
rank sends to the corresponding tensor rank in the next pipeline stage.

Each data rank consumes six distinct examples, split into two three-example
microbatches. It accumulates local gradients over the two microbatches, then
averages them across its data group. This is a sequential microbatch schedule,
with no claim of pipeline bubble removal or overlap. Its update is SGD so the
three parallel transformations can be checked independently of AdamW.

## Socket and collective interfaces

`CollectiveServer(world_size, host='127.0.0.1', port=0, token=None, timeout=10)`
is a context manager. It creates a fresh 256-bit random token if none is supplied
and exposes the selected `.port` and `.token`. Spawned ranks construct
`ProcessGroup(rank, world_size, host, port, token)`. `subgroup(members)` returns
a view sharing the same rank connection. Peer/root arguments always use global
ranks; shard order follows sorted subgroup membership.

The group exposes `all_reduce(array, reduction='sum')`, `all_gather(array)`,
`scatter(arrays_or_None, root)`, `gather(array, root)`,
`reduce_scatter(flat_array, reduction='sum')`, `broadcast(array_or_None, root)`,
`send(array, dst, tag)`, `recv(src, tag)`, `barrier()`, and `close()`.
Reductions also accept floating-point `mean`. All-gather supports uneven input
lengths. Reduce-scatter splits a flat summed array with `numpy.array_split`.
Collective order and arguments must agree within a group. Errors, missing
participants, or unexpected disconnects abort the session; recovery requires a
new run from checkpoint.

Frames carry a fixed `!IQ` network-order header with JSON-metadata length and
raw-array-payload length, then metadata, bytes, and a 32-byte HMAC-SHA256. The
MAC covers the complete header, metadata, and payload. Each direction maintains
an independent monotonic sequence counter. Array decoding checks a fixed dtype
allowlist, dimensional bounds, contiguous byte extents, and exact payload
coverage. The default metadata limit is 64 KiB, frame payload limit is 64 MiB,
and aggregate outstanding collective input limit is 256 MiB. Point-to-point
queues have separate 256 MiB and 4,096-message limits. These are bounds for a
small local demonstration, not a capacity plan for huge models. Intermediate
copies and collective result buffers add to these bounds.

HMAC provides authentication and integrity, not confidentiality or privilege
separation among ranks sharing the token. The coordinator is trusted and has
access to all gradients and parameters. Do not expose this listener to a public
network. Distributed connections require a separately managed encrypted private
channel. Local multiprocessing queues are used only for parent/worker outcomes;
network messages never use pickle.

## Parameter server

`ParameterServerClient(group)` provides `initialize(name, value, workers=None)`
(rank zero only), `pull(name) -> (version, array)`, and
`push(name, version, gradient, learning_rate) -> (accepted, version, array)`.
Initialization records the fixed worker set. Every accepted worker contributes
once to a version. Once all contributions arrive, the server averages them and
atomically applies a finite FP32 SGD update. Stale pushes return `accepted=False`
and the current version without modifying state. Shape, learning-rate, worker,
duplicate contribution, and finite-value checks precede an update. This is
synchronous versioned aggregation, not asynchronous staleness compensation.

## ZeRO Stage 3 API and memory accounting

Construct `ShardedAdamW(ordered_numpy_parameters, data_group, learning_rate=...)`.
All ranks validate a common parameter-layout/hyperparameter digest. For each
named tensor, the optimizer partitions flattened indices with quotient and
remainder splitting; a tensor smaller than the group yields valid empty shards.
Local slices from all tensors occupy one contiguous FP32 parameter allocation.
FP32 gradients, first moments, and second moments have exactly the same local
layout. No persistent full parameter array is retained by this optimizer.

The explicit lifecycle is `materialize(name)`, use the returned array for a
layer, `accumulate_gradient(name, full_gradient)`, and `release(name)`. The caller
must also discard its own full-array references. Only one tensor may remain
materialized through this API at a time. `accumulate_gradient` reduce-scatters
the full gradient, averages data ranks by default, and accumulates into local
storage. `step(gradient_scale=microbatch_count, max_grad_norm=...)` unscales,
computes global shard norm, clips, checks all-rank finite status, and applies
bias-corrected AdamW. Non-finite gradients skip the update on every rank.

For a rank owning `n_r` parameters, persistent array storage is exactly
`16*n_r` bytes: four FP32 vectors. AdamW arithmetic adds local-sized temporary
arrays. A materialized tensor with `G` elements requires a full `4G`-byte array,
plus its received shard-list buffers and serialization copies; reduce-scatter
also transiently requires a full gradient. Constructor input mappings and
caller-owned model/autograd arrays are outside the optimizer's ownership and
must be counted separately. The centralized coordinator stores collective
inputs/results; it does not share the ranks' Stage 3 memory savings.

`state_dict()` returns local arrays, rank/layout metadata, hyperparameters,
accepted-update count, and gradient-accumulation counters. `load_state_dict()`
validates matching rank, world size, parameter layout, finite FP32 arrays, and
hyperparameters. Resharding checkpoints across a new world size is not
implemented. Three-rank tests compare three AdamW updates, uneven shards,
gradient accumulation, clipping, restored state, and an empty bias shard with a
separate full-vector NumPy reference.

## Automatic Transformer integration

`ZeroTransformer(config, group, seed=0, learning_rate=3e-4)` implements the
same MLA/RoPE/SwiGLU architecture using sharded parameter ownership. Call it
with integer `[batch,time]` token IDs and an optional attention mask; the result
is an ordinary differentiable Tensor of logits. Call `.zero_grad()` before an
accumulation window, perform backward for each appropriately normalized loss,
then call `.step(gradient_scale=1, max_grad_norm=...)` once all ranks finish.
All ranks must execute the same module graph and collective sequence.

Embedding, linear, and weighted normalization operators automatically gather
and release named parameters. Their backward callbacks retain activation inputs,
not gathered weights. A scalar leaf keeps parameter-dependent paths in autograd.
The tied output head and lookup accumulate into one embedding shard. A mutation
generation rejects stale backward graphs after updates or checkpoint restore.
Use `.memory_report()` to inspect actual owned array bytes and materialization
counts; the report lists exclusions from process-wide peak memory.

The default `zero_train.py` command verifies three updates over two ranks against
a dense FP32 combined-batch reference. `--checkpoint DIRECTORY` saves the actual
shards. `--resume DIRECTORY --steps N` performs N additional updates in a new
process group, validating numerical continuity against the dense prefix.

## Collective checkpoint API

Every rank calls `save_sharded_checkpoint(path, optimizer, rng, counters,
run_id=None, max_payload_bytes=...)`. Rank payloads and metadata are immutable.
A leader-held nonblocking filesystem lock excludes concurrent transactions;
persisted run ownership excludes unrelated jobs. Every prepared file is fsynced
and validated before one manifest rename commits the generation. Old generations
remain; orphan files are never mistaken for committed state.

Every rank calls `load_sharded_checkpoint(path, optimizer, rng,
expected_run_id=None, max_payload_bytes=...)`. Loading checks fixed topology,
layout, ownership, scalar types, array shapes/dtypes, writable destinations,
finite values, nonnegative moments, hashes, RNG, and counters. Only after all
ranks validate does any rank apply state. Returned counters identify the next
training input. Loading restores the run ownership for subsequent saves.
A communication failure after publication can make a caller uncertain about
commit success; consult the manifest before retrying. Tests include real worker
termination during save and exact continuation in entirely new worker processes.
The directory must be shared and trusted; these locks and rename guarantees are
not a claim about arbitrary network filesystems or object stores.

## Actual corpus and SFT entry point

`zero_corpus.py pretrain --output RUN --steps 2` trains the sharded model on
packed source text; `sft` uses response masks from prompt/response JSONL. The
source is staged with a 16 MiB bound, and byte BPE learns from at most 64 KiB.
Use `--data PATH` for another source and `--resume --steps N` for N total
completed updates. World size is bounded to 1..4 local processes, accumulation
to 1..4 microbatches, and context to 4..64 positions.

Every rank traverses the same deterministic cyclic stream and selects its own
slot among each group of data-degree samples. If local summed token loss is
L_r, total supervised target mass is M, and data degree is D, backward uses
L_r*D/M. The sharded optimizer's mean reduction then yields sum_r(dL_r)/M.
This avoids weighting a short response equally with a long response unless the
objective explicitly requests pair weighting. Exact-resume tests compare every
parameter, gradient, and Adam state shard. An unequal-mask SFT case compares the
distributed update with a dense token-weighted reference.

Run ownership, staged-file hashes, config/tokenizer consistency, rank counters,
global sample position, and checkpoint presence are validated before restart.
The CLI's SIGTERM handler unwinds spawned-process ownership and preserves the
last manifest, rather than leaving detached workers. Numerical traces and real
packed-corpus reports are separate evidence. Serving loads the dense local
training format; these corpus checkpoints are currently sharded-only artifacts.
