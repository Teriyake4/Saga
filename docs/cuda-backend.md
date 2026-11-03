# CUDA projection substrate and backend/session adapter

The implementation tracked by [CUDA-01 / #16](https://github.com/Teriyake4/Saga/issues/16)
connects the CUDA executor to `IGpuBackend`, `IExecutionSession` and startup
planning. It is deliberately **not automatically registered** in the server.
The available device math is still BF16 projection only, not a model runtime,
Q3/Q4 execution, attention/KV, GDN, MTP, graph execution or HTTP generation.
Implementation and host tests do not establish native CUDA qualification.

## Startup binding

`include/saga/backend/cuda_session.h` provides `CudaBackend` and a startup-only
`SessionBinder`. The binder receives the immutable selected plan and a resource
table containing canonical granted allocation IDs. It returns owning metadata
and prepared warmup/prefill/decode callbacks; absent or incompatible callbacks
fail before use. Borrowed plan/resource-table references must not escape the
binder. Bringup-buffer copies must not be retained by callbacks.

Callbacks receive a noncopyable, nonmovable `SessionExecution` capability, valid
only on the owner thread during the callback. It exposes checked copies,
synchronization and launches of the exact startup-selected preparations; it has
no allocation, registry or raw executor access. Copies of a selected
`PreparedKernel` retain preparation identity; resolving another same-named
kernel does not authorize it. There is no token-loop registry resolution or
kernel ranking. Model-specific equations and request-to-tensor mapping belong
to the eventual model implementation, not this adapter.

The backend permits one live session, including state retained by step consumers.
Except for `request_cancel()`, callers externally serialize all lifecycle,
execution, health/report and teardown operations on the creating thread. Steps
are currently synchronous: returned success proves physical settlement, not
merely submission. Caller leases, request data and session resources remain
retained by pending-result consumers after settlement. This is not a generation
scheduler or a claim of prefill/decode model semantics.

`request_cancel()` is a cross-thread stop request for the active step. Requests
while idle are ignored. Cancellation recorded before terminal publication wins;
callbacks can observe the stop token, but cancellation settles only after a
physical fence. Stop callbacks run without the cancellation mutex held. Graph
capture and verify are unsupported and fail explicitly.

## Allocations and approval

Instantiation checks the exact candidate metadata, aliases, identity and graph
mode against the grant, and validates that the executor matches the selected
device facts/profile. Unique project records are allocated once with their exact
padded extents, locations and alignment. Zero-byte declarations are measured
without physical allocation. The table uses canonical IDs, not alias spellings.
Bringup resources are released after successful warmup and physical drain;
persistent/execution resources stay with the session and consumers.

An explicit bounded device runtime-overhead reservation is required. The
executor's memory baseline is sampled **after context/device initialization but
before its stream/events are created**. Device availability is reported at that
same boundary: pre-existing context memory is already excluded from available
memory, while subsequent stream/event/runtime growth is included in the observed
residual. Do not combine this with a pre-context availability budget without
separately reserving the excluded footprint. The residual subtracts tracked
live device buffers and can include allocator caches or unrelated process
activity; it is not exact driver attribution, a complete host-allocator audit,
or a model-fit prediction.

The coordinator remains the sole creator of startup approval, after warmup,
physical settlement, overhead observation and reconciliation. A bringup grant
alone never establishes readiness. The adapter does not turn on `/readyz` or
claim a supported model. Unsupported/unknown resources and graph profiles fail
closed rather than silently substituting a context, codec or execution path.

## Failed physical settlement

`ExecutorOptions::fence_timeout` bounds polling of a recorded CUDA event (default
30 seconds). CUDA API calls themselves are still subject to driver/OS behavior;
the evidence runner separately bounds the entire process. Query/record errors,
timeout or inability to select the device poison the executor. New work is
rejected; there is no retry-until-success teardown or in-process device recovery.

Potentially referenced buffers/ledger leases are retained in process-lifetime
quarantine. A failed session fence retains its complete state, active request and
caller lease too. `failure_report()` reports executor pending-allocation counts,
bytes and reason; it is not a complete inventory of all additional session-owned
host objects. `physically_settled()` is false after failure, preventing startup
approval. Failed teardown must not release or reuse these owners.

Span-based copy overloads borrow host storage. If their fence cannot be proved,
the executor terminates the process before borrowed storage can unwind. Use
owned `Buffer` copy overloads when recoverable quarantine is required. Buffer
release errors that cannot safely be recovered also fail-stop. Quarantine is
intentional retained ownership, not a successful free or a normal memory leak
recovery mechanism. Process termination may be the only reclamation boundary.

## Projection and validation scope

`register_bf16_projection_kernels()` supplies baseline and WMMA SM89 candidates
for `Y[T,N] = X[T,K] * W[N,K]^T`, with BF16 inputs and FP32 accumulation/output.
Prefill/decode/verify suffixes are phase labels for this same stateless operation,
not implementations of model phases. Baseline supports guarded tails; WMMA
requires N/K/T multiples of 16 and its declared stride/alignment multiples.
Actual extents, device/domain, ownership, overlap and arithmetic are checked at
launch. Existing event timings use different fixtures and do not rank variants.

`cuda_executor` and `cuda_session` CTests compile production executor/session
sources against an isolated fake runtime. They are host contract tests, not CUDA
execution. The separate physical projection and session targets use real CUDA;
see [native device testing](m2-device-testing.md). Measured tuning belongs to
[PERF-01 / #26](https://github.com/Teriyake4/Saga/issues/26), not this bridge.
