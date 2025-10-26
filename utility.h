#ifndef XGBOOST_TESTS_UTILITY_H
#define XGBOOST_TESTS_UTILITY_H

#include <string>
#include <fstream>
#include <stdexcept>
#include <xgboost/c_api.h>

// ============================================================================
// XGBOOST UTILITY FUNCTIONS
// Common utilities for XGBoost C++ programs
// ============================================================================

namespace xgb_utils {

// Global log file stream
extern std::ofstream logFile;

// ============================================================================
// LOGGING FUNCTIONS
// ============================================================================

/**
 * @brief Initialize logging system
 * @param filename Name of the log file to create
 * @return true if successful, false otherwise
 */
bool initLogging(const std::string& filename = "xgboost_log.txt");

/**
 * @brief Close the log file
 */
void closeLogging();

/**
 * @brief Log a message to both console and log file
 * @param message Message to log
 */
void log(const std::string& message);

/**
 * @brief Log a formatted message (printf-style) to both console and log file
 * @param format Format string (printf-style)
 * @param ... Variable arguments
 */
void logf(const char* format, ...);

// ============================================================================
// ERROR HANDLING MACRO
// ============================================================================

/**
 * @brief Macro to check XGBoost C API return codes and throw exception on error
 *
 * Usage:
 *   SAFE_XGBOOST(XGDMatrixCreateFromMat(data, rows, cols, -1, &dmat));
 */
#define SAFE_XGBOOST(call) {  \
    int err = (call); \
    if (err != 0) { \
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
        ": error in " + #call + ": " + XGBGetLastError());  \
    } \
}

// For backwards compatibility with existing code
#define safe_xgboost SAFE_XGBOOST

// ============================================================================
// ARRAY INTERFACE HELPERS (for sparse matrix creation)
// ============================================================================

/**
 * @brief Create JSON-encoded array interface for sparse matrix data
 * @param data Pointer to data (as size_t address)
 * @param n Number of elements
 * @param typestr NumPy dtype string (e.g., "<u8", "<u4", "<f4")
 * @param length Maximum length of output buffer
 * @param out Output buffer (must be pre-allocated)
 */
void makeArrayInterface(size_t data, size_t n, const char* typestr, size_t length, char* out);

/**
 * @brief Create JSON-encoded DMatrix configuration
 * @param n_threads Number of threads to use
 * @param length Maximum length of output buffer
 * @param out Output buffer (must be pre-allocated)
 */
void makeConfig(int n_threads, size_t length, char* out);

} // namespace xgb_utils

#endif // XGBOOST_TESTS_UTILITY_H

