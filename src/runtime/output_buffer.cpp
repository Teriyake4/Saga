#include "saga/runtime/output_buffer.h"

#include <algorithm>
#include <stdexcept>

namespace {
std::pair<std::size_t, std::size_t> checked_capacities(std::size_t byte_capacity,
                                                       std::size_t chunk_capacity) {
    if (byte_capacity == 0 || chunk_capacity == 0)
        throw std::invalid_argument("output buffer capacities must be nonzero");
    return {byte_capacity, chunk_capacity};
}
}

namespace saga::runtime {

OutputBuffer::OutputBuffer(std::size_t byte_capacity, std::size_t chunk_capacity)
    : OutputBuffer(checked_capacities(byte_capacity, chunk_capacity)) {}

OutputBuffer::OutputBuffer(std::pair<std::size_t, std::size_t> capacities)
    : storage_(capacities.first), chunks_(capacities.second) {}

OutputPublishResult OutputBuffer::try_publish_committed(std::string_view bytes, const Completion& commit) {
    {
        std::lock_guard lock(mutex_);
        if (closed_) return OutputPublishResult::closed;
    }
    if (!commit.valid()) return OutputPublishResult::invalid_commit;
    if (!commit.is_ready()) return OutputPublishResult::commit_not_ready;
    if (commit.status() != CompletionStatus::success) return OutputPublishResult::commit_not_success;
    if (bytes.empty()) return OutputPublishResult::empty_chunk;
    if (bytes.size() > storage_.size()) return OutputPublishResult::chunk_too_large;

    std::lock_guard lock(mutex_);
    if (closed_) return OutputPublishResult::closed;
    if (buffered_chunks_ == chunks_.size() || bytes.size() > storage_.size() - buffered_bytes_)
        return OutputPublishResult::backpressure;

    const auto offset = byte_tail_;
    const auto first = std::min(bytes.size(), storage_.size() - byte_tail_);
    std::copy_n(bytes.data(), first, storage_.data() + byte_tail_);
    if (first != bytes.size()) std::copy_n(bytes.data() + first, bytes.size() - first, storage_.data());
    byte_tail_ = (byte_tail_ + bytes.size()) % storage_.size();
    buffered_bytes_ += bytes.size();
    chunks_[chunk_tail_] = Chunk{offset, bytes.size()};
    chunk_tail_ = (chunk_tail_ + 1) % chunks_.size();
    ++buffered_chunks_;
    return OutputPublishResult::published;
}

OutputReadResult OutputBuffer::try_read() {
    std::lock_guard lock(mutex_);
    if (buffered_chunks_ == 0) {
        return OutputReadResult{closed_ ? OutputReadStatus::closed : OutputReadStatus::empty, {}};
    }
    const Chunk chunk = chunks_[chunk_head_];
    std::string result;
    try {
        result.resize(chunk.size);
    } catch (const std::bad_alloc&) {
        return OutputReadResult{OutputReadStatus::allocation_failure, {}};
    } catch (const std::length_error&) {
        return OutputReadResult{OutputReadStatus::allocation_failure, {}};
    }
    const auto first = std::min(chunk.size, storage_.size() - chunk.offset);
    std::copy_n(storage_.data() + chunk.offset, first, result.data());
    if (first != chunk.size) std::copy_n(storage_.data(), chunk.size - first, result.data() + first);
    byte_head_ = (byte_head_ + chunk.size) % storage_.size();
    buffered_bytes_ -= chunk.size;
    chunk_head_ = (chunk_head_ + 1) % chunks_.size();
    --buffered_chunks_;
    return OutputReadResult{OutputReadStatus::data, std::move(result)};
}

bool OutputBuffer::close(OutputTerminal terminal) noexcept {
    if (terminal != OutputTerminal::success && terminal != OutputTerminal::error &&
        terminal != OutputTerminal::cancelled) return false;
    std::lock_guard lock(mutex_);
    if (closed_) return false;
    terminal_ = terminal;
    closed_ = true;
    return true;
}

OutputSnapshot OutputBuffer::snapshot() const noexcept {
    std::lock_guard lock(mutex_);
    return OutputSnapshot{buffered_bytes_, buffered_chunks_, storage_.size(), chunks_.size()};
}

bool OutputBuffer::is_closed() const noexcept {
    std::lock_guard lock(mutex_);
    return closed_;
}

std::optional<OutputTerminal> OutputBuffer::terminal() const noexcept {
    std::lock_guard lock(mutex_);
    return terminal_;
}

} // namespace saga::runtime
