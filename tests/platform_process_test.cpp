#include "saga/platform/process.hpp"

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cstring>
#endif

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void throws(Function function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

#ifdef _WIN32
void set_test_environment(const wchar_t* name, const wchar_t* value) {
    check(SetEnvironmentVariableW(name, value) != 0, "set Windows environment");
}
#endif
}

int main() {
    try {
        char program[] = "test";
        char empty[] = "";
        char unicode[] = "\u6A21\u578B-\u00E9";
        char invalid[] = {static_cast<char>(0xc0), static_cast<char>(0xaf), 0};
        char* argv[] = {program, empty, unicode};
        const auto arguments = saga::platform::collect_arguments(3, argv);
        check(arguments.size() == 2 && arguments[0].empty() && arguments[1] == "\u6A21\u578B-\u00E9",
              "UTF-8 argv");
        char* bad_argv[] = {program, invalid};
        throws<std::invalid_argument>([&] { (void)saga::platform::collect_arguments(2, bad_argv); },
                                      "invalid argv encoding");
        throws<std::invalid_argument>([] {
            (void)saga::platform::collect_arguments(-1, static_cast<char* const*>(nullptr));
        }, "negative argc");
        throws<std::invalid_argument>([] {
            (void)saga::platform::collect_arguments(4097, static_cast<char* const*>(nullptr));
        }, "argument count bound");
        std::string too_long(32768, 'x');
        char* long_argv[] = {program, too_long.data()};
        throws<std::length_error>([&] { (void)saga::platform::collect_arguments(2, long_argv); },
                                  "per-argument bound");
        char* null_argv[] = {program, nullptr};
        throws<std::invalid_argument>([&] { (void)saga::platform::collect_arguments(2, null_argv); },
                                      "null argv");

#ifdef _WIN32
        wchar_t wide_program[] = L"test";
        wchar_t wide_unicode[] = L"\u6A21\u578B-\u00E9";
        wchar_t* wide_argv[] = {wide_program, wide_unicode};
        const auto wide_arguments = saga::platform::collect_arguments(2, wide_argv);
        check(wide_arguments.size() == 1 && wide_arguments[0] == "\u6A21\u578B-\u00E9", "wide UTF-16 argv");
        wchar_t invalid_surrogate[] = {static_cast<wchar_t>(0xd800), L'\0'};
        wchar_t* invalid_wide_argv[] = {wide_program, invalid_surrogate};
        throws<std::invalid_argument>([&] {
            (void)saga::platform::collect_arguments(2, invalid_wide_argv);
        }, "invalid UTF-16 argv");
#else
        throws<std::invalid_argument>([] {
            (void)saga::platform::collect_arguments(0, static_cast<wchar_t* const*>(nullptr));
        }, "wide argv rejected on POSIX");
#endif

#ifdef _WIN32
        const wchar_t missing[] = L"SAGA_PLATFORM_PROCESS_TEST_MISSING_97231";
        SetEnvironmentVariableW(missing, nullptr);
        check(!saga::platform::read_environment("SAGA_PLATFORM_PROCESS_TEST_MISSING_97231"),
              "missing environment");
        set_test_environment(L"SAGA_PLATFORM_PROCESS_TEST_EMPTY", L"");
        set_test_environment(L"SAGA_PLATFORM_PROCESS_TEST_VALUE", L"12345");
        set_test_environment(L"SAGA_PLATFORM_PROCESS_TEST_\u6A21\u578B", L"\u6771\u4EAC-\u00E9");
#else
        const char* missing = "SAGA_PLATFORM_PROCESS_TEST_MISSING_97231";
        unsetenv(missing);
        check(!saga::platform::read_environment(missing), "missing environment");
        setenv("SAGA_PLATFORM_PROCESS_TEST_EMPTY", "", 1);
        setenv("SAGA_PLATFORM_PROCESS_TEST_VALUE", "12345", 1);
        setenv("SAGA_PLATFORM_PROCESS_TEST_\u6A21\u578B", "\u6771\u4EAC-\u00E9", 1);
#endif
        check(saga::platform::read_environment("SAGA_PLATFORM_PROCESS_TEST_EMPTY") ==
                  std::optional<std::string>(""), "empty environment");
        check(saga::platform::read_environment("SAGA_PLATFORM_PROCESS_TEST_\u6A21\u578B") ==
                  std::optional<std::string>("\u6771\u4EAC-\u00E9"), "native Unicode environment round trip");
        throws<std::length_error>([] {
            (void)saga::platform::read_environment("SAGA_PLATFORM_PROCESS_TEST_\u6A21\u578B", 3);
        }, "environment bound is UTF-8 bytes");
        throws<std::length_error>([] {
            (void)saga::platform::read_environment("SAGA_PLATFORM_PROCESS_TEST_VALUE", 4);
        }, "environment bound");
        throws<std::invalid_argument>([] { (void)saga::platform::read_environment("BAD=NAME"); },
                                      "invalid name");
        throws<std::invalid_argument>([] {
            (void)saga::platform::read_environment("NAME", 1024 * 1024 + 1);
        }, "environment max bound");

#ifndef _WIN32
        struct sigaction previous {};
        struct sigaction custom {};
        custom.sa_handler = SIG_IGN;
        sigemptyset(&custom.sa_mask);
        check(sigaction(SIGTERM, &custom, &previous) == 0, "install prior handler");
        {
            saga::platform::ShutdownSignals signals;
            throws<std::logic_error>([] { saga::platform::ShutdownSignals second; },
                                     "duplicate registration");
            check(raise(SIGTERM) == 0 && signals.requested(), "registered SIGTERM sets request");
        }
        struct sigaction current {};
        check(sigaction(SIGTERM, nullptr, &current) == 0 && current.sa_handler == SIG_IGN,
              "restore prior SIGTERM handler");
        check(sigaction(SIGTERM, &previous, nullptr) == 0, "restore original process handler");
#else
        {
            saga::platform::ShutdownSignals signals;
            throws<std::logic_error>([] { saga::platform::ShutdownSignals second; },
                                     "duplicate registration");
        }
#endif
        {
            saga::platform::ShutdownSignals again;
            check(!again.requested(), "sequential registration resets flag");
        }

        std::cout << "PASS: process helper tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
