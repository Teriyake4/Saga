#include "saga/runtime/output_buffer.h"

#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <mutex>
#include <string>
#include <thread>

namespace {
using namespace saga::runtime;
void check(bool value, const char* message) {
    if (!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}

void test_commit_checks_and_ring() {
    bool invalid_capacities_threw = false;
    try { OutputBuffer buffer(std::numeric_limits<std::size_t>::max(), 0); }
    catch (const std::invalid_argument&) { invalid_capacities_threw = true; }
    check(invalid_capacities_threw, "both capacities validated before either ring allocates");
    invalid_capacities_threw = false;
    try { OutputBuffer buffer(0, std::numeric_limits<std::size_t>::max()); }
    catch (const std::invalid_argument&) { invalid_capacities_threw = true; }
    check(invalid_capacities_threw, "zero byte capacity rejected before chunk allocation");

    OutputBuffer buffer(5, 2);
    CompletionSource pending_source;
    check(buffer.try_publish_committed("a", pending_source.completion()) == OutputPublishResult::commit_not_ready, "pending commit rejected");
    Completion invalid;
    check(buffer.try_publish_committed("a", invalid) == OutputPublishResult::invalid_commit, "invalid commit rejected");
    for (const auto status : {CompletionStatus::failure, CompletionStatus::cancelled, CompletionStatus::abandoned}) {
        CompletionSource source;
        auto fence = source.completion();
        if (status == CompletionStatus::failure) source.fail();
        else if (status == CompletionStatus::cancelled) source.cancel();
        else source = CompletionSource{};
        check(buffer.try_publish_committed("a", fence) == OutputPublishResult::commit_not_success, "non-success commit rejected");
    }
    CompletionSource source;
    auto fence = source.completion();
    source.succeed();
    check(buffer.try_publish_committed("", fence) == OutputPublishResult::empty_chunk, "empty chunk rejected");
    check(buffer.try_publish_committed("123456", fence) == OutputPublishResult::chunk_too_large, "oversized chunk rejected");

    OutputBuffer byte_limited(4, 4);
    check(byte_limited.try_publish_committed("abc", fence) == OutputPublishResult::published, "byte limited first chunk");
    const auto before_byte_full = byte_limited.snapshot();
    check(byte_limited.try_publish_committed("de", fence) == OutputPublishResult::backpressure, "byte capacity backpressures independently");
    check(byte_limited.snapshot().buffered_bytes == before_byte_full.buffered_bytes && byte_limited.snapshot().buffered_chunks == before_byte_full.buffered_chunks, "byte-full rejection leaves snapshot unchanged");

    OutputBuffer chunk_limited(8, 1);
    check(chunk_limited.try_publish_committed("a", fence) == OutputPublishResult::published, "chunk limited first chunk");
    check(chunk_limited.try_publish_committed("b", fence) == OutputPublishResult::backpressure, "chunk capacity backpressures independently");

    check(buffer.try_publish_committed("abc", fence) == OutputPublishResult::published, "first wrap-test chunk published");
    check(buffer.try_read().bytes == "abc", "first wrap-test chunk read");
    check(buffer.try_publish_committed("xyz", fence) == OutputPublishResult::published, "xyz published with tail at physical offset three");
    check(buffer.try_read().bytes == "xyz", "chunk physically spanning storage end reads correctly");
    check(buffer.snapshot().buffered_bytes == 0 && buffer.snapshot().buffered_chunks == 0, "snapshot drains");
}

void test_close_and_terminal() {
    OutputBuffer buffer(3, 1);
    CompletionSource source;
    auto fence = source.completion();
    source.succeed();
    check(!buffer.terminal().has_value(), "open buffer has no terminal metadata");
    check(!buffer.close(static_cast<OutputTerminal>(99)), "invalid terminal rejected");
    check(!buffer.is_closed() && !buffer.terminal().has_value(), "invalid close leaves buffer open");
    check(buffer.try_publish_committed("abc", fence) == OutputPublishResult::published, "fill before close");
    const auto full = buffer.snapshot();
    check(buffer.close(OutputTerminal::cancelled), "first valid close wins");
    const auto after_close = buffer.snapshot();
    check(full.buffered_bytes == after_close.buffered_bytes && full.buffered_chunks == after_close.buffered_chunks, "close when full preserves queued snapshot");
    check(!buffer.close(OutputTerminal::error), "close is idempotent");
    check(buffer.terminal() == OutputTerminal::cancelled, "first terminal retained");
    check(buffer.try_publish_committed("x", fence) == OutputPublishResult::closed, "post-close publish rejected");
    check(buffer.try_publish_committed("", Completion{}) == OutputPublishResult::closed, "closed takes precedence over invalid commit and empty chunk");
    check(buffer.try_publish_committed("1234", Completion{}) == OutputPublishResult::closed, "closed takes precedence over oversize");
    check(buffer.try_read().bytes == "abc", "queued output drains after close");
    check(buffer.try_read().status == OutputReadStatus::closed, "EOF follows queued chunks");
}

void test_deterministic_slow_consumer_releases_space() {
    OutputBuffer buffer(1, 1);
    CompletionSource source;
    auto fence = source.completion();
    source.succeed();
    std::mutex mutex;
    std::condition_variable changed;
    bool backpressure_observed = false;
    bool consumed = false;
    std::string assembled;

    std::thread producer([&] {
        check(buffer.try_publish_committed("a", fence) == OutputPublishResult::published, "initial publish fills bounded buffer");
        check(buffer.try_publish_committed("b", fence) == OutputPublishResult::backpressure, "full buffer reports backpressure");
        {
            std::lock_guard lock(mutex);
            backpressure_observed = true;
        }
        changed.notify_all();
        {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return consumed; });
        }
        check(buffer.try_publish_committed("b", fence) == OutputPublishResult::published, "consumer read releases bounded space");
        check(buffer.close(OutputTerminal::success), "producer closes after publishing");
    });
    std::thread consumer([&] {
        {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return backpressure_observed; });
        }
        auto result = buffer.try_read();
        check(result.status == OutputReadStatus::data, "consumer obtains first chunk");
        assembled += result.bytes;
        {
            std::lock_guard lock(mutex);
            consumed = true;
        }
        changed.notify_all();
        for (;;) {
            result = buffer.try_read();
            if (result.status == OutputReadStatus::data) assembled += result.bytes;
            else if (result.status == OutputReadStatus::closed) break;
        }
    });
    producer.join();
    consumer.join();
    check(assembled == "ab", "slow consumer assembles exact output");
    const auto snapshot = buffer.snapshot();
    check(snapshot.buffered_bytes == 0 && snapshot.buffered_chunks == 0, "consumer releases all bounded space");
}
}

int main() {
    test_commit_checks_and_ring();
    test_close_and_terminal();
    test_deterministic_slow_consumer_releases_space();
}
