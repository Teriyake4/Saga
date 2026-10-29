#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "saga/platform/process.hpp"
#include <windows.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <string>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void exercise(DWORD event) {
    {
        saga::platform::ShutdownSignals first;
    }
    saga::platform::ShutdownSignals signals;
    require(GenerateConsoleCtrlEvent(event, 0) != 0, "GenerateConsoleCtrlEvent failed");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!signals.requested() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    require(signals.requested(), "console event did not request shutdown before deadline");
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected event name");
        const std::string event = argv[1];
        if (event == "ctrl-c") exercise(CTRL_C_EVENT);
        else if (event == "ctrl-break") exercise(CTRL_BREAK_EVENT);
        else throw std::runtime_error("unknown console event name");
        std::cout << "PASS: isolated Windows " << event << " registration lifecycle\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
