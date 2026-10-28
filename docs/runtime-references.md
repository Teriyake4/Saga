# Neutral runtime contracts and design references

## Local contract

`PROPOSAL.md`, section "Uniform asynchronous lifetime and failure contract", specifies backend-neutral completion, successful-only result readability, retention until execution and consumer release, cancellation without premature buffer reclamation, and deferred physical reclamation. `include/saga/memory/ledger.h` defines `AllocationLease` as independently copyable/releasable across threads and `OwnedBacking` as exclusive ownership with no raw-owner escape. These contracts inform the host completion primitives; this API does not perform backend synchronization.

## Architecture reference

- NInfer maintainer engine architecture, pinned revision `64d10fbf9f8e6bfc79ab0570b94674c5b9997929`: https://raw.githubusercontent.com/Neroued/ninfer/64d10fbf9f8e6bfc79ab0570b94674c5b9997929/docs/maintainer/engine-architecture.md
- Retrieved content observed: the document describes separate ownership boundaries for EngineCore, Scheduler, ResourceManager, and Program; states that Program owns physical resources and device allocations; and describes cancellation/cleanup as waiting for in-flight operations to reach stable completion before releasing dependent resources.
- SHA-256: `775a47e2a663c20c9de9e2a748f702f289c9eae74a2da3ce05a41dfd223176ff`, recorded for this exact URL/path/revision in `bench/reference/source-lock.json` (32,163 bytes). This is source-lock retrieval evidence; it is not a hash recomputed from the rendered fetch response in this task.

No source code was imported from the reference. Resource deleters must be nonthrowing. Abandonment is a host-level terminal notification, NOT proof of a device fence: a backend must retain its producer until physical work has completed. Cancellation is only a terminal outcome after contributing work completes; it must never be used to reclaim buffers while work may still access them.

## Implemented neutral library

Public headers live in `include/saga/runtime/`; implementations live in
`src/runtime/`. CMake builds them as `saga_runtime`, separate from `saga_host`.

| Component | Contract | Caller obligation |
|---|---|---|
| `Completion` / `CompletionSource` / `ResourceLease` | Shared observation, exactly-once settlement, retained RAII resources; only successful results readable | Keep producer/resources alive through physical work; do not use abandonment as a fence; no throwing deleters |
| `ExecutionWorker` | One worker, bounded task queue, separately bounded pre-reserved cleanup; stop/capture destruction outside queue locks; shutdown drains cleanup | Each callback must finish/fence external work before returning; reserve cleanup before beginning work that may require rollback |
| `SequenceSlots` | Pool/generation-tagged logical handles, immediate invalidation on retirement, successful-fence-only reuse; otherwise quarantine | Owner-thread operations; logical reservation is not physical resource ownership; retain and finish retirement tokens |
| `AdmissionCoordinator` | Single active request, bounded queue/outstanding payload counts and bytes; deadline/cancellation tracking | Owner-thread coordinator operations; poll regularly, forward cancellation, supply a truthful work fence and retain payload resources through execution |
| `OutputBuffer` | Fixed byte/chunk rings, successful-commit gate, nonblocking backpressure, owning chunk reads, drain before terminal read | Supply the correct commit fence; define committed model results elsewhere; stop producing/retry under an external bounded policy |

### Ownership and shutdown details

- `CompletionSource` is move-only; source moves/destruction need external
  serialization even though competing settlement calls are supported.
- `ExecutionWorker::shutdown()` returns false without blocking in its own worker
  or stop-notification context. An external owner must join it. Destroying the
  worker facade from one of its callbacks is unsupported. Shutdown is cooperative,
  not a deadline guarantee for callbacks that fail to return.
- Destroying/dropping a retirement token does not recycle a sequence slot.
  `SequenceSlots` owns logical state only, not physical execution allocations.
- Admission tickets permit concurrent cancellation and observation. Their stop
  callbacks may reenter ticket APIs, not coordinator owner operations. Active
  cancellation settles only after the attached work fence becomes terminal; failed
  or abandoned fences fail-stop admission. Controller destruction is notification,
  not synchronization with external execution.
- Admission quota tracks normalized payload bytes, not arbitrary caller resource
  sizes. Tickets, copied completions and execution resource leases retain the
  payload and its quota. Queue capacity N allows one ready entry plus N waiters
  while idle, or N waiters while active; `queued()` includes the ready entry.
- Output close is first-valid-close-wins. Already queued bytes remain readable;
  `try_read()` reports closed after they drain. `terminal()` becomes available at
  close, even before the drain completes. Chunk copying failure consumes nothing.
- Worker callbacks, resource owners and deferred cleanup must be composed in the
  right destruction order. A ready completion does not imply all producer wrapper
  objects have already been destroyed; join where a test needs final-release proof.

These components are not connected to HTTP generation, do not establish model
commit semantics, and contain no CUDA synchronization, model math, SSE encoding or
production model/backend registries. Host-only tests do not qualify Windows/CUDA.
See `docs/HANDOFF.md` for validation and the next integration boundaries.
