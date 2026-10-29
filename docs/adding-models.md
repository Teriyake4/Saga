# Adding a model family

Saga model families implement the existing contracts in
`include/saga/runtime/model_contracts.h`. Generate a deliberately incomplete,
fail-closed starting point:

```sh
python3 tools/new_model.py --name AcmeText --id acme-text --output src/models
```

This creates `src/models/acme-text/` with a definition, runtime, registration
entrypoint, local CMake integration snippet, and README. The output directory is
created if needed; an existing family path is never overwritten. A symlink used
as or within the output-parent path is followed normally: generated files land
at its resolved destination. The final family directory itself is claimed with
an exclusive directory creation; if another process wins that race, generation
fails without replacing that process's files. Names must be PascalCase C++
identifiers and IDs lowercase kebab-case. IDs are normalized with a `model_`
namespace prefix, so IDs that are C++ keywords (such as `class`) still generate
valid namespace and CMake identifiers. The generator is
stdlib-only and deterministic. Generated methods throw `std::logic_error` until
implemented; the scaffold is not a usable runtime and makes no readiness claim.
The CMake file is a local snippet, not a change to Saga's build or a plugin
loader. Register definitions explicitly during startup via `ModelRegistry`.

## Definition responsibilities

The generated class derives from `runtime::IModelDefinition`; preserve these
actual override signatures:

```cpp
runtime::ModelInspection inspect(const runtime::ArtifactMetadata&) const override;
std::vector<backend::ModelProgram> describe_programs(
    const runtime::ModelInspection&, const runtime::RuntimeOptions&) const override;
memory::Profile describe_resources(
    const runtime::ModelInspection&, const runtime::RuntimeOptions&) const override;
std::unique_ptr<runtime::IModelRuntime> create_runtime(
    const runtime::ModelDescription&, runtime::IExecutionSession&) const override;
```

`inspect` must validate the artifact and return explicit `ModelSpec`, frontend,
state, and capabilities. State is declared with `StateSchema` (kind, dtype,
dimensions, location, lifetime, resource ID). Resource declarations belong in
`memory::Profile::terms`: each term needs justified identity, owner, category,
location, lifetime, size or per-context size, alignment, and physical identity
where applicable. Do not guess geometry or allocation sizes; unknown evidence is
not a zero-byte allocation. Use the planner/validators and reject unsupported or
ambiguous artifacts.

`describe_programs` supplies finite, bounded ordered `backend::ModelProgram`
call lists using existing `KernelRequest` contracts. Alternatives are explicit
program alternatives, not arbitrary graph IR. The selected backend/kernel
registries remain responsible for compatibility and binding; a model must not
fork the scheduler, backend API, or per-model kernel registry. Never claim
mathematical equivalence for alternatives without evidence.

Declare optional MTP/vision capability only when implemented and inspected.
Respect `RuntimeOptions` requests and fail clearly when a requested capability
is not supported. Keep required artifact capabilities distinct from optional
ones. Runtime methods must implement sequence lifecycle and prefill/decode
semantics, preserve resource ownership through physical settlement, and honor
session drain/fence rules. Unsupported `verify` can remain the contract's
`nullopt` default; do not advertise MTP merely because it exists in the API.

## Weights, formats, and registration

Describe logical weights via `backend::LogicalWeightBinding`, including logical
name, physical identity, encoding, shape, byte offset/extent, and artifact
component identity. Shared storage should use intentional identical physical
identity and exact alias ranges; do not conflate artifact component identity
with backing identity. Validate actual payload metadata and packing against the
supported encoding. A model architecture or encoding implementation does not
imply support for arbitrary GGUF files: GGUF is a container with variant
metadata, tensor naming, quantization, and layout conventions. Add only explicit
format/version/encoding combinations whose parser and physical contracts are
implemented and tested.

The generated `register_model(runtime::ModelRegistry&)` is an explicit startup
entrypoint. Call it from the owning application integration and link its target
using the local CMake snippet as appropriate. Do not add dynamic loading.

## Validation checklist

1. Add artifact inspection tests for accepted and rejected metadata, including
   aliasing and payload-layout boundaries.
2. Test program descriptors against supported kernel registry behavior and
   device facts; test resource profiles through the planner, including unknown
   sizes and budget rejection.
3. Test optional capabilities requested, unavailable, and not requested.
4. Test runtime sequence creation/release, prefill/decode results, error paths,
   cancellation, and physical drain/lifetime guarantees using test sessions.
5. Compile the generated/implemented translation unit against current Saga
   headers, then run focused tests and the normal test suite.
6. Establish numerical and device qualification separately. Structural contract
   validation is not proof of payload correctness, allocation fit, performance,
   or production readiness.
