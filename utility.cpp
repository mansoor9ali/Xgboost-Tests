#include "utility.h"
#include <iostream>
#include <cstdarg>
#include <cstring>
#include <cstdio>

namespace xgb_utils {

// Global log file stream
std::ofstream logFile;

// ============================================================================
// LOGGING FUNCTIONS IMPLEMENTATION
// ============================================================================

bool initLogging(const std::string& filename) {
    if (logFile.is_open()) {
        logFile.close();
    }

    logFile.open(filename);
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file: " << filename << std::endl;
        return false;
    }

    log("=== XGBoost Log Started ===");
    return true;
}

void closeLogging() {
    if (logFile.is_open()) {
        log("=== XGBoost Log Ended ===");
        logFile.close();
    }
}

void log(const std::string& message) {
    // Print to console
    std::cout << message << std::endl;

    // Write to log file
    if (logFile.is_open()) {
        logFile << message << std::endl;
        logFile.flush();
    }
}

void logf(const char* format, ...) {
    char buffer[1024];
    va_list args;

    // Format the string
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // Print to console
    std::cout << buffer << std::endl;

    // Write to log file
    if (logFile.is_open()) {
        logFile << buffer << std::endl;
        logFile.flush();
    }
}

// ============================================================================
// ARRAY INTERFACE HELPERS IMPLEMENTATION
// ============================================================================

void makeArrayInterface(size_t data, size_t n, const char* typestr, size_t length, char* out) {
    static const char kTemplate[] =
        "{\"data\": [%zu, true], \"shape\": [%zu, %zu], \"typestr\": \"%s\", \"version\": 3}";

    std::memset(out, '\0', length);
    std::snprintf(out, length, kTemplate, data, n, 1ul, typestr);
}

void makeConfig(int n_threads, size_t length, char* out) {
    static const char kTemplate[] = "{\"missing\": NaN, \"nthread\": %d}";

    std::memset(out, '\0', length);
    std::snprintf(out, length, kTemplate, n_threads);
}

} // namespace xgb_utils

