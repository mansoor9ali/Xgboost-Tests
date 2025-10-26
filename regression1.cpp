#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdarg>
#include <fstream>
#include <string>
#include <filesystem>
#include <xgboost/c_api.h>

// This is a tutorial for using the XGBoost C API.
// The code demonstrates how to create a simple XGBoost model, train it, and make predictions.

// Wrapper to check XGBoost C API return codes
// Global log file stream

/*
*This program demonstrates how to use the XGBoost library's C API to train a simple gradient boosting model and make predictions.
*Sets up logging - Creates a custom logging system that writes output to both the console and a log file (xgboost_log.txt)
Creates training data:
Generates a small dataset with 5 rows and 3 columns (features)
Each value is calculated as (row+1) * (col+1)
Creates labels using the formula 1 + i³ (1, 2, 9, 28, 65)
Trains an XGBoost model:
Converts the data into XGBoost's DMatrix format
Creates a gradient boosting tree model with specific hyperparameters:
Objective: reg:squarederror (squared error regression)
Max depth: 5
Learning rate (eta): 0.1
Subsample: 0.5
Trains for 200 iterations
Makes predictions:
Creates test data (identical to training data in this case)
Uses the trained model to predict labels for the test data
Logs both the original labels and predictions
Cleans up - Frees all allocated XGBoost resources
Type of Machine Learning
This is a supervised regression task using gradient boosted decision trees. The model learns the relationship between the input features and the continuous target labels, then predicts values for new data.
The program is essentially a tutorial/example demonstrating the basic workflow of XGBoost in C++.
 *
 *
 */
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

    // create the train data
    const int cols = 3, rows = 5;
    std::vector<float> train(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            train[i * cols + j] = (i + 1) * (j + 1);
        }
    }

    std::vector<float> train_labels(rows);
    for (int i = 0; i < rows; i++) {
        train_labels[i] = 1 + i * i * i;
    }


    // convert to DMatrix
    DMatrixHandle h_train[1];
    safe_xgboost(XGDMatrixCreateFromMat(train.data(), rows, cols, -1, &h_train[0]));

    // load the labels
    safe_xgboost(XGDMatrixSetFloatInfo(h_train[0], "label", train_labels.data(), rows));

    // read back the labels, just a sanity check
    bst_ulong bst_result;
    const float *out_floats;
    XGDMatrixGetFloatInfo(h_train[0], "label" , &bst_result, &out_floats);
    for (unsigned int i=0;i<bst_result;i++)
        logf("label[%d]=%f", i, out_floats[i]);

    // create the booster and load some parameters
    BoosterHandle h_booster;
    XGBoosterCreate(h_train, 1, &h_booster);
    XGBoosterSetParam(h_booster, "booster", "gbtree");
    XGBoosterSetParam(h_booster, "objective", "reg:squarederror");
    XGBoosterSetParam(h_booster, "max_depth", "5");
    XGBoosterSetParam(h_booster, "eta", "0.1");
    XGBoosterSetParam(h_booster, "min_child_weight", "1");
    XGBoosterSetParam(h_booster, "subsample", "0.5");
    XGBoosterSetParam(h_booster, "colsample_bytree", "1");
    XGBoosterSetParam(h_booster, "num_parallel_tree", "1");
    //param['updater'] = 'grow_gpu'
    safe_xgboost(XGBoosterSetParam(h_booster, "device", "cuda")); // Use GPU for training if available

    // perform 200 learning iterations
    for (int iter=0; iter<200; iter++)
        XGBoosterUpdateOneIter(h_booster, iter, h_train[0]);

    // predict
    const int sample_rows = 5;
    std::vector<float> test(sample_rows * cols);
    for (int i = 0; i < sample_rows; i++) {
        for (int j = 0; j < cols; j++) {
            test[i * cols + j] = (i + 1) * (j + 1);
        }
    }
    DMatrixHandle h_test;
    XGDMatrixCreateFromMat(test.data(), sample_rows, cols, -1, &h_test);
    bst_ulong out_len;
    const float *f;
    XGBoosterPredict(h_booster, h_test, 0, 0, 0, &out_len, &f);

    for (unsigned int i=0;i<out_len;i++)
        logf("prediction[%d]=%f", i, f[i]);


    // free xgboost internal structures
    XGDMatrixFree(h_train[0]);
    XGDMatrixFree(h_test);
    XGBoosterFree(h_booster);


    logFile.close();
    return 0;
}
