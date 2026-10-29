#include "saga/runtime/model_contracts.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {
void check(bool value, const char* message) {
    if (!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
struct Sentinel {
    explicit Sentinel(int& count) : count(count) {}
    ~Sentinel() noexcept { ++count; }
    int& count;
};
}

int main() {
    using namespace saga::runtime;
    int released = 0;
    auto source = std::make_unique<StepSource>(ResourceLease(std::make_unique<Sentinel>(released)),
                                                   StepResultBounds{8, 2});
    auto pending = source->pending();
    auto last_reader = pending;
    bool threw = false;
    try { (void)pending.result(); } catch (const std::logic_error&) { threw = true; }
    check(threw, "pending result is unavailable before success");
    OwningStepResult value{{StepRowResult{7, {1.0f, 2.0f}, {std::byte{3}}}}};
    check(source->succeed(std::move(value)), "success settles once");
    check(!source->cancel() && !source->fail(), "subsequent settlements are rejected");
    source.reset();
    auto result = last_reader.result();
    check(result.rows.size() == 1 && result.rows[0].consumed_frontier == 7 &&
          result.rows[0].logits.size() == 2 && result.rows[0].data.size() == 1,
          "successful result is copied out as owning bounded data");
    check(released == 0, "consumer retains request resources after producer destruction");
    pending = {};
    check(released == 0, "other consumer retains request resources");
    last_reader = {};
    check(released == 1, "last consumer releases retained resources");

    StepSource failed({}, StepResultBounds{0, std::nullopt});
    auto failure = failed.pending();
    check(failed.fail(std::make_exception_ptr(std::runtime_error("step failed"))), "failure settles");
    threw = false;
    try { (void)failure.result(); } catch (const std::runtime_error& e) { threw = std::string(e.what()) == "step failed"; }
    check(threw, "failure never exposes result and rethrows stored exception");

    StepSource cancelled({}, StepResultBounds{0, std::nullopt});
    auto cancellation = cancelled.pending();
    check(cancelled.cancel(), "cancellation settles");
    threw = false;
    try { (void)cancellation.result(); } catch (const std::runtime_error&) { threw = true; }
    check(threw, "cancelled step has no result");

    PendingStep abandoned;
    {
        StepSource producer({}, StepResultBounds{0, std::nullopt});
        abandoned = producer.pending();
    }
    check(abandoned.status() == CompletionStatus::abandoned, "producer abandonment is observable");

    FrontendContract frontend{"frontend", FrontendPolicy::caller_token_ids, 16, {1}};
    PrefillRequest request{{1, 0, 4}, {{{2, 3}, {0, 1}, 8, 2}}};
    validate_step_request(request, frontend, 16);
    request.rows.front().token_ids.front() = 16;
    threw = false;
    try { validate_step_request(request, frontend, 16); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "out-of-range token IDs are rejected");
    request.rows.front().token_ids.front() = 2;
    request.rows.front().positions.front() = 8;
    threw = false;
    try { validate_step_request(request, frontend, 16); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "out-of-range positions are rejected");
    request.rows.front().positions.front() = 0;
    request.rows.push_back(request.rows.front());
    threw = false;
    try { validate_step_request(request, frontend, 16); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "multi-row requests are rejected at the initial boundary");
    request.rows.resize(1);
    request.rows.front().context_tokens = 17;
    threw = false;
    try { validate_step_request(request, frontend, 16); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "request context beyond the model maximum is rejected");

    StepSource bounded({}, StepResultBounds{4, 1});
    auto bounded_pending = bounded.pending();
    check(!bounded.succeed({}) && !bounded_pending.is_ready(),
          "result row count must match the typed request boundary");
    check(!bounded.succeed({{{4, {}, {}}}}) && !bounded_pending.is_ready(),
          "logit row must match the expected vocabulary shape");
    OwningStepResult oversized;
    oversized.rows.push_back({std::numeric_limits<std::uint64_t>::max(), {1.0f}, {}});
    check(!bounded.succeed(std::move(oversized)) && !bounded_pending.is_ready(),
          "frontier beyond the context limit is rejected without settling success");
    check(bounded.succeed({{{4, {1.0f}, {}}}}) && bounded_pending.result().rows.size() == 1,
          "bounded producer publishes one row with the expected vocabulary shape");

    StepSource raced({}, StepResultBounds{2, std::nullopt});
    auto raced_pending = raced.pending();
    std::atomic<int> settlements{0};
    std::thread success([&] { if (raced.succeed({{{1, {}, {}}}})) ++settlements; });
    std::thread cancellation_race([&] { if (raced.cancel()) ++settlements; });
    success.join();
    cancellation_race.join();
    check(settlements == 1 && raced_pending.is_ready(), "racing settlements have exactly one winner");
    if (raced_pending.status() == CompletionStatus::success)
        check(raced_pending.result().rows.size() == 1, "winning success publishes its result");
    else {
        threw = false;
        try { (void)raced_pending.result(); } catch (const std::runtime_error&) { threw = true; }
        check(threw, "winning cancellation does not publish a result");
    }
    return 0;
}
