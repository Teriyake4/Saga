# Platform helpers

The `saga_platform` CMake target owns OS behavior separately from model/backend
math. `src/platform/CMakeLists.txt` selects common `shared/` sources and exactly
one native backend. Windows is active; `linux/CMakeLists.txt` retains a deferred
Linux selection reusing the `posix/` implementation without duplicated bodies.
macOS selects `posix/` only for development/regression tests. No Linux or Windows
runtime qualification is inferred from a macOS pass.

## Read-only file mapping

Build exactly one native backend with the shared mapping source:

- Windows: `src/platform/shared/file_mapping.cpp` + `src/platform/windows/file_mapping.cpp`.
- POSIX (macOS host development and Linux): `src/platform/shared/file_mapping.cpp` + `src/platform/posix/file_mapping.cpp`.

The public API remains in `include/saga/platform/file_mapping.hpp`. The POSIX implementation was exercised on macOS; this does not qualify Linux. Windows/MSVC has not been tested here.

## Process helpers

Build the common UTF-8 validation source with exactly one native process backend:

- Windows: `src/platform/shared/process.cpp` + `src/platform/windows/process.cpp`.
- POSIX: `src/platform/shared/process.cpp` + `src/platform/posix/process.cpp`.

Public API in `include/saga/platform/process.hpp`: `collect_arguments(int, char* const[])`, `collect_arguments(int, wchar_t* const[])`, `read_environment(std::string_view, std::size_t max_bytes = 4096)`, and noncopyable/nonmovable `ShutdownSignals` with `requested() const noexcept`.

Argument collection excludes argv[0], bounds each argument, and requires valid UTF-8; Windows wide arguments are converted with strict native UTF-16 conversion. Environment names and values are validated as UTF-8, names reject NUL and `=`, and values are bounded by UTF-8 byte length. Environment values are returned only to the caller and are not logged. POSIX environment reads rely on the usual process rule that the environment is not concurrently mutated.

At most one `ShutdownSignals` registration may be active. Handlers only set a static lock-free atomic flag. POSIX installs SIGINT/SIGTERM handlers and restores their exact prior `sigaction` state on destruction; Windows registers Ctrl-C/Ctrl-Break with `SetConsoleCtrlHandler` and unregisters on destruction. Signal/console notification is a shutdown request only: it does not stop sockets or perform application cleanup, which must happen in normal thread context. POSIX behavior was exercised on macOS only; no Windows/MSVC test was run. Native
Windows console-close/logoff/shutdown service handling is not implemented: only
Ctrl-C and Ctrl-Break are graceful requests. Handler removal is best-effort and
callbacks touch only process-lifetime atomic storage, not destroyed caller data. The standalone `tests/platform_process_test.cpp` exercises UTF-8 arguments, environment bounds, and POSIX registration lifecycle without triggering default signal actions.
