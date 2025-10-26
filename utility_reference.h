#ifndef XGBOOST_TESTS_UTILITY_H
#define XGBOOST_TESTS_UTILITY_H

/*
 * ============================================================================
 * XGBOOST UTILITY LIBRARY - QUICK REFERENCE
 * ============================================================================
 *
 * Common utilities for XGBoost C++ programs
 * Include this header in your XGBoost projects to access logging, error
 * handling, and helper functions.
 *
 * USAGE:
 *   #include "utility.h"
 *   using namespace xgb_utils;
 *
 * ============================================================================
 */

#include <string>
#include <fstream>
#include <stdexcept>
#include <xgboost/c_api.h>

namespace xgb_utils {

// ============================================================================
// LOGGING FUNCTIONS
// ============================================================================

/**
 * @brief Initialize logging system - Call this at the start of your program
 *
 * @param filename Name of the log file (default: "xgboost_log.txt")
 * @return true if successful, false otherwise
 *
 * Example:
 *   initLogging("my_model.log");
 */
bool initLogging(const std::string& filename = "xgboost_log.txt");

/**
 * @brief Close the log file - Call this at the end of your program
 *
 * Example:
 *   closeLogging();
 */
void closeLogging();

/**
 * @brief Log a message to both console and log file
 *
 * @param message Message to log
 *
 * Example:
 *   log("Training started");
 *   log(std::string("Epoch: ") + std::to_string(epoch));
 */
void log(const std::string& message);

/**
 * @brief Log a formatted message (printf-style) to console and file
 *
 * @param format Format string (printf-style)
 * @param ... Variable arguments
 *
 * Example:
 *   logf("Epoch %d: loss = %.4f", epoch, loss);
 *   logf("Training samples: %d, Test samples: %d", n_train, n_test);
 */
void logf(const char* format, ...);

// ============================================================================
// ERROR HANDLING MACRO
// ============================================================================

/**
 * @brief Macro to check XGBoost C API return codes and throw exception on error
 *
 * Wraps XGBoost API calls and automatically checks for errors.
 * Throws std::runtime_error with detailed error information if the call fails.
 *
 * Usage:
 *   SAFE_XGBOOST(XGDMatrixCreateFromMat(data, rows, cols, -1, &dmat));
 *   SAFE_XGBOOST(XGBoosterCreate(cache, 1, &booster));
 *   SAFE_XGBOOST(XGBoosterSetParam(booster, "eta", "0.1"));
 *
 * Note: Also available as lowercase 'safe_xgboost' for backward compatibility
 */
#define SAFE_XGBOOST(call) {  \
    int err = (call); \
    if (err != 0) { \
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
        ": error in " + #call + ": " + XGBGetLastError());  \
    } \
}

// Backward compatibility with existing code
#define safe_xgboost SAFE_XGBOOST

// ============================================================================
// ARRAY INTERFACE HELPERS (for sparse matrix creation)
// ============================================================================

/**
 * @brief Create JSON-encoded array interface for sparse matrix data
 *
 * Used when creating sparse matrices (CSR/CSC format) in XGBoost.
 * Creates the JSON string required by XGDMatrixCreateFromCSR/CSC.
 *
 * @param data Pointer to data array (as size_t address)
 * @param n Number of elements in the array
 * @param typestr NumPy dtype string:
 *                - "<u8" for uint64_t (indptr)
 *                - "<u4" for uint32_t (indices)
 *                - "<f4" for float (data values)
 * @param length Maximum length of output buffer
 * @param out Output buffer (must be pre-allocated)
 *
 * Example:
 *   const float data[] = {1.0, 2.0, 3.0};
 *   char j_data[128];
 *   makeArrayInterface((size_t)data, 3, "<f4", sizeof(j_data), j_data);
 */
void makeArrayInterface(size_t data, size_t n, const char* typestr, size_t length, char* out);

/**
 * @brief Create JSON-encoded DMatrix configuration
 *
 * Creates configuration JSON for sparse matrix creation.
 *
 * @param n_threads Number of threads to use (0 for auto)
 * @param length Maximum length of output buffer
 * @param out Output buffer (must be pre-allocated)
 *
 * Example:
 *   char config[64];
 *   makeConfig(4, sizeof(config), config);
 */
void makeConfig(int n_threads, size_t length, char* out);

} // namespace xgb_utils

// ============================================================================
// TYPICAL PROGRAM STRUCTURE
// ============================================================================
/*

#include "utility.h"
using namespace xgb_utils;

int main() {
    // 1. Initialize logging
    initLogging("my_program.log");

    // 2. Create and prepare data
    DMatrixHandle h_train;
    SAFE_XGBOOST(XGDMatrixCreateFromMat(data, rows, cols, -1, &h_train));
    SAFE_XGBOOST(XGDMatrixSetFloatInfo(h_train, "label", labels, rows));

    // 3. Create and configure booster
    BoosterHandle booster;
    DMatrixHandle cache[] = {h_train};
    SAFE_XGBOOST(XGBoosterCreate(cache, 1, &booster));
    SAFE_XGBOOST(XGBoosterSetParam(booster, "objective", "reg:squarederror"));
    SAFE_XGBOOST(XGBoosterSetParam(booster, "eta", "0.1"));
    SAFE_XGBOOST(XGBoosterSetParam(booster, "device", "cpu"));

    // 4. Train
    log("Training started...");
    for (int iter = 0; iter < 100; iter++) {
        SAFE_XGBOOST(XGBoosterUpdateOneIter(booster, iter, h_train));
        if (iter % 10 == 0) {
            logf("Iteration %d completed", iter);
        }
    }
    log("Training completed!");

    // 5. Predict
    bst_ulong out_len;
    const float* out_result;
    SAFE_XGBOOST(XGBoosterPredict(booster, h_train, 0, 0, 0, &out_len, &out_result));
    logf("Made predictions for %lu samples", out_len);

    // 6. Cleanup
    SAFE_XGBOOST(XGDMatrixFree(h_train));
    SAFE_XGBOOST(XGBoosterFree(booster));

    // 7. Close logging
    closeLogging();
    return 0;
}

*/

#endif // XGBOOST_TESTS_UTILITY_H

