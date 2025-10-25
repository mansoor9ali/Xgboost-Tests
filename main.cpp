#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <fstream>
#include <string>
#include <filesystem>
#include <xgboost/c_api.h>

// This is a tutorial for using the XGBoost C API.
// The code demonstrates how to create a simple XGBoost model, train it, and make predictions.

// Wrapper to check XGBoost C API return codes
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


#define safe_xgboost(call) {  \
int err = (call); \
if (err != 0) { \
throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
": error in " + #call + ":" + XGBGetLastError());  \
} \
}


int main() {
    // Open a log file to write the output.
    logFile.open("xgboost_log.txt");

    // This tutorial uses two datasets, data1 and data2, to demonstrate how to create DMatrix objects.
    // A DMatrix is the internal data structure that XGBoost uses.

    // 1D matrix (must be float for XGBoost API)
    const float data1[] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };

    // 2D matrix (must be float for XGBoost API)
    const int ROWS = 6, COLS = 3;
    const float data2[ROWS][COLS] = { {1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 6.0f}, {3.0f, -1.0f, 9.0f}, {4.0f, 8.0f, -1.0f}, {2.0f, 5.0f, 1.0f}, {0.0f, 1.0f, 5.0f} };

    // DMatrixHandle is a pointer to the DMatrix object.
    DMatrixHandle dmatrix1, dmatrix2;

    // Create a DMatrix from the 1D data.
    // Pass the matrix, number of rows & columns contained in the matrix variable.
    // Here 0.0f represents the missing value in the matrix dataset.
    // dmatrix1 will contain the created DMatrix.
    safe_xgboost(XGDMatrixCreateFromMat(data1, 1, 50, 0.0f, &dmatrix1));
    log("dmatrix1 created from 1D data.");

    // Create a DMatrix from the 2D data.
    // Here -1.0f represents the missing value in the matrix dataset.
    // dmatrix2 will contain the created DMatrix.
    safe_xgboost(XGDMatrixCreateFromMat((const float*)data2, ROWS, COLS, -1.0f, &dmatrix2));
    log("dmatrix2 created from 2D data.");

    // In this tutorial, we will use dmatrix2 for training.
    // A model needs labels for training. Let's create some for our data2.
    const float labels[ROWS] = { 0, 1, 0, 1, 0, 1 };
    safe_xgboost(XGDMatrixSetFloatInfo(dmatrix2, "label", labels, ROWS));
    log("Labels added to dmatrix2.");

    // Create a Booster, which is the model itself.
    // We pass the training DMatrix in an array.
    BoosterHandle booster;
    DMatrixHandle dms[] = { dmatrix2 };
    safe_xgboost(XGBoosterCreate(dms, 1, &booster));
    log("Booster created.");

    // Set parameters for the booster.
    // We'll use a simple binary logistic objective for this example.
    safe_xgboost(XGBoosterSetParam(booster, "objective", "binary:logistic"));
    safe_xgboost(XGBoosterSetParam(booster, "eta", "0.1")); // learning rate
    log("Booster parameters set.");

    // Train the model for a few iterations.
    log("Training the model for 10 iterations...");
    for (int i = 0; i < 10; ++i) {
        safe_xgboost(XGBoosterUpdateOneIter(booster, i, dmatrix2));
    }
    log("Training finished.");

    // Now, let's make predictions on some data.
    // For simplicity, we'll predict on the same data we trained on (dmatrix2).
    // In a real-world scenario, you would use a separate test dataset.
    bst_ulong out_len;
    const float* out_result;

    // Predict the labels for the data in dmatrix2.
    safe_xgboost(XGBoosterPredict(booster, dmatrix2, 0, 0, 0, &out_len, &out_result));
    logf("Predictions made for %lu samples.", out_len);

    // Print the predictions.
    log("Predictions:");
    for (bst_ulong i = 0; i < out_len; ++i) {
        logf("Sample %lu: %f", i, out_result[i]);
    }

    // It's important to free the memory used by the DMatrix and Booster handles
    // to avoid memory leaks.
    safe_xgboost(XGDMatrixFree(dmatrix1));
    safe_xgboost(XGDMatrixFree(dmatrix2));
    safe_xgboost(XGBoosterFree(booster));
    log("Cleaned up resources.");

    logFile.close();
    return 0;
}
