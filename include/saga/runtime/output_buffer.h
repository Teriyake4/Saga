#pragma once

#include "saga/runtime/completion.h"

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace saga::runtime {

enum class OutputPublishResult { published, invalid_commit, commit_not_ready, commit_not_success, empty_chunk, chunk_too_large, backpressure, closed };
enum class OutputTerminal { success, error, cancelled };
enum class OutputReadStatus { data, empty, closed, allocation_failure };

struct OutputReadResult {
    OutputReadStatus status{OutputReadStatus::empty};
    std::string bytes;
};

struct OutputSnapshot {
    std::size_t buffered_bytes{};
    std::size_t buffered_chunks{};
    std::size_t byte_capacity{};
    std::size_t chunk_capacity{};
};

// Bounded committed-byte channel. The caller must pass a completion that represents
// the appropriate commit fence; this class cannot establish that execution contract.
// Destruction must be externally serialized against all operations.
class OutputBuffer {
public:
    OutputBuffer(std::size_t byte_capacity, std::size_t chunk_capacity);
    OutputBuffer(const OutputBuffer&) = delete;
    OutputBuffer& operator=(const OutputBuffer&) = delete;
    OutputBuffer(OutputBuffer&&) = delete;
    OutputBuffer& operator=(OutputBuffer&&) = delete;

    OutputPublishResult try_publish_committed(std::string_view bytes, const Completion& commit);
    // Allocation failure while copying a chunk returns allocation_failure without consuming it.
    OutputReadResult try_read();
    bool close(OutputTerminal terminal) noexcept;
    OutputSnapshot snapshot() const noexcept;
    bool is_closed() const noexcept;
    // Empty until the first valid close; the first valid terminal value wins.
    std::optional<OutputTerminal> terminal() const noexcept;

private:
    explicit OutputBuffer(std::pair<std::size_t, std::size_t> capacities);
    struct Chunk { std::size_t offset{}; std::size_t size{}; };
    mutable std::mutex mutex_;
    std::vector<char> storage_;
    std::vector<Chunk> chunks_;
    std::size_t byte_head_{};
    std::size_t byte_tail_{};
    std::size_t buffered_bytes_{};
    std::size_t chunk_head_{};
    std::size_t chunk_tail_{};
    std::size_t buffered_chunks_{};
    bool closed_{};
    std::optional<OutputTerminal> terminal_;
};

} // namespace saga::runtime
