#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <fstream>
#include <string>
#include <filesystem>
#include <xgboost/c_api.h>// Wrapper to check XGBoost C API return codes
// Global log file stream
std::ofstream logFile;

// Custom logging functions
void log(const std::string& message) {
    std::cout << message << std::endl;
    if (logFile.is_open()) {
        logFile << message << std::endl;
        logFile.flush();
    }
}

void logf(const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    std::cout << buffer << std::endl;
    if (logFile.is_open()) {
        logFile << buffer << std::endl;
        logFile.flush();
    }
}

 // static inline void safe_xgboost(int ret) {
 //        if (ret != 0) {
 //            const char* err = XGBGetLastError();
 //            logf("XGBoost API error: %s", err ? err : "unknown");
 //            std::exit(EXIT_FAILURE);
 //        }
 //    }
#define safe_xgboost(call) {  \
int err = (call); \
if (err != 0) { \
throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
": error in " + #call + ":" + XGBGetLastError());  \
} \
}


int main() {


    // 1D matrix (must be float for XGBoost API)
    const float data1[] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };

    // 2D matrix (must be float for XGBoost API)
    const int ROWS = 6, COLS = 3;
    const float data2[ROWS][COLS] = { {1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 6.0f}, {3.0f, -1.0f, 9.0f}, {4.0f, 8.0f, -1.0f}, {2.0f, 5.0f, 1.0f}, {0.0f, 1.0f, 5.0f} };
    DMatrixHandle dmatrix1, dmatrix2;
    // Pass the matrix, no of rows & columns contained in the matrix variable
    // here 0.0f represents the missing value in the matrix dataset
    // dmatrix variable will contain the created DMatrix using it
    safe_xgboost(XGDMatrixCreateFromMat(data1, 1, 50, 0.0f, &dmatrix1));
    // here -1.0f represents the missing value in the matrix dataset
    safe_xgboost(XGDMatrixCreateFromMat((const float*)data2, ROWS, COLS, -1.0f, &dmatrix2));



    // // // Open log file with absolute path
    // // logFile.open(".\\Xgboost-Tests\\xgboost.log", std::ios::out | std::ios::trunc);
    // // Open log file with directory creation (requires C++17+)
    // std::filesystem::path logPath = std::filesystem::path(".") / "Xgboost-Tests" / "xgboost.log";
    // std::error_code ec;
    // std::filesystem::create_directories(logPath.parent_path(), ec);
    // if (ec) {
    //     std::cerr << "Failed to create log directory: " << ec.message() << std::endl;
    // }
    // logFile.open(logPath.string(), std::ios::out | std::ios::trunc);
    //
    // if (!logFile.is_open()) {
    //     std::cerr << "Failed to open log file!" << std::endl;
    //     return 1;
    // }
    //
    // log("=== XGBoost Test Program Starting ===");
    // log("Log file created successfully");
    //
    // // Sample data: 6 samples, 3 features each
    // log("\nCreating training data...");
    // std::vector<float> train_data = {
    //     1.0f, 2.0f, 3.0f,
    //     4.0f, 5.0f, 6.0f,
    //     7.0f, 8.0f, 9.0f,
    //     10.0f, 11.0f, 12.0f,
    //     13.0f, 14.0f, 15.0f,
    //     16.0f, 17.0f, 18.0f
    // };
    //
    // // Labels for binary classification
    // std::vector<float> train_labels = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    // log("Training data created: 6 samples, 3 features");
    //
    // // Create DMatrix for training data
    // log("\nCreating DMatrix...");
    // DMatrixHandle train_matrix;
    // XGDMatrixCreateFromMat(train_data.data(), 6, 3, -1, &train_matrix);
    // XGDMatrixSetFloatInfo(train_matrix, "label", train_labels.data(), 6);
    // log("DMatrix created successfully");
    //
    // // Create booster
    // log("\nCreating booster...");
    // BoosterHandle booster;
    // DMatrixHandle matrices[] = {train_matrix};
    // XGBoosterCreate(matrices, 1, &booster);
    // log("Booster created successfully");
    //
    // // Set parameters
    // log("\nSetting parameters...");
    // XGBoosterSetParam(booster, "objective", "binary:logistic");
    // XGBoosterSetParam(booster, "max_depth", "3");
    // XGBoosterSetParam(booster, "eta", "0.1");
    // log("Parameters set: objective=binary:logistic, max_depth=3, eta=0.1");
    //
    // log("\nStarting training for 10 iterations...");
    //
    // // Train for 10 iterations
    // for (int iter = 0; iter < 10; ++iter) {
    //     XGBoosterUpdateOneIter(booster, iter, train_matrix);
    //     logf("Iteration %d completed", iter);
    // }
    //
    // log("Training loop completed");
    //
    // // Make predictions
    // log("\nMaking predictions...");
    // bst_ulong out_len;
    // const float* out_result;
    // XGBoosterPredict(booster, train_matrix, 0, 0, 0, &out_len, &out_result);
    //
    // log("\n=== Training completed successfully! ===");
    // log("Predictions:");
    //
    // for (bst_ulong i = 0; i < out_len; ++i) {
    //     logf("Sample %llu: %.4f", (unsigned long long)i, out_result[i]);
    // }
    //
    // log("\n=== Cleanup ===");
    //
    // // Cleanup
    // XGBoosterFree(booster);
    // XGDMatrixFree(train_matrix);
    //
    // log("Booster and DMatrix freed");
    // log("\n=== Done! ===");
    // log("All output displayed on console and saved to xgboost.log");
    //
    // // Close log file
    // logFile.close();

    return 0;
}
