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
production model/backend registrations or session implementations. Host-only tests
do not qualify Windows/CUDA. See `docs/HANDOFF.md` for validation and the next
integration boundaries.

## Neutral model/backend contract slice

`include/saga/runtime/model_contracts.h` adds a minimal M2.1/M2.2 boundary, not a
complete model schema or generation runtime:

- `ArtifactMetadata` owns declared component identities and sizes. Existing
  `backend::LogicalWeightBinding::component_identity` selects the source component;
  `physical_identity` separately identifies a tensor's backing. Multiple logical
  names may share that physical identity only when component, offset, extent,
  encoding and shape agree exactly. Different physical identities may occupy
  disjoint ranges in one component, but not overlapping ranges.
- `validate_model_contract()` checks structural identities, positive dimensions,
  bounds, context, declared capability availability and matching program phases.
  This slice requires nonempty weight bindings and program calls; an empty/no-op
  program is not a model-contract convention. Kernel requests reuse the existing
  guards through `backend::validate_request()`. None of these checks verifies actual
  payload bytes, represented values, model compatibility, fit or qualification.
- `ModelRegistry` and `BackendRegistry` are independently populated, explicit,
  startup-only and externally serialized/non-reentrant. They reject null, empty
  and duplicate registrations and unknown selection; selected shared owners survive
  registry destruction. Backend selections are mutable so session instantiation is
  callable. No production model/backend or silent fallback is registered.
- `IModelDefinition` describes a model and creates an `IModelRuntime` whose current
  contract covers sequence creation/release, not prefill/decode or typed text results.
  `IGpuBackend::instantiate()` consumes the startup description and granted resources;
  `IExecutionSession::execute()` runs only its prebound immutable program. Needed
  borrowed factory inputs must be copied before return; no per-call program lookup
  or generic model-program interpretation is established.
- A bring-up grant permits bounded preparation, not serving. Session execution and
  sequence release obey the existing physical-fence and resource-retention contracts.
  Operations/destruction must be serialized, physical work drained, and runtime
  destroyed before its borrowed session. Completion consumers may outlive both if
  the implementation retains every needed backing owner in the completion.

`tests/model_contracts_test.cpp` supplies only synthetic implementations. It checks
metadata rejection/alias boundaries, registry ownership, grant-backed instantiation,
constructor-failure unwinding and retained storage through the last copied completion.
It does not implement context planning, inspect-to-approval orchestration, a controller,
HTTP/SSE integration, real model math or device synchronization.
