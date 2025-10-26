#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdarg>
#include <fstream>
#include <string>
#include <random>
#include <algorithm>
#include <map>
#include <iomanip>
#include <xgboost/c_api.h>
#include "utility.h"

// Use the utility namespace for convenience
using namespace xgb_utils;

/*
 * This program demonstrates XGBoost multi-class classification using the Iris dataset.
 *
 * The Python code performs:
 * - Splitting iris dataset into train/test (80/20 split with random_state=17)
 * - Training XGBClassifier with eval_metric="auc"
 * - Making predictions on test set
 * - Making a single prediction on example input [4.5, 3.0, 1.5, 0.25]
 * - Generating classification report
 *
 * This C++ version implements:
 * - Manual train/test split of Iris dataset (80/20)
 * - XGBoost classifier training with multi:softprob objective
 * - Predictions on test set
 * - Single example prediction
 * - Confusion matrix and basic classification metrics
 *
 * ============================================================================
 * WHY WE USE XGBoosterPredict() AND NOT XGBClassifier:
 * ============================================================================
 *
 * ANSWER: There is NO XGBClassifier class in the XGBoost C/C++ API!
 *
 * The XGBoost C API is a LOW-LEVEL, PROCEDURAL interface that provides:
 * - XGBoosterCreate()      -> Creates a booster (model)
 * - XGBoosterSetParam()    -> Sets parameters (objective, eta, max_depth, etc.)
 * - XGBoosterUpdateOneIter() -> Trains for one iteration
 * - XGBoosterPredict()     -> Makes predictions (for BOTH regression AND classification)
 *
 * Python's XGBClassifier and XGBRegressor are HIGH-LEVEL wrappers built on top
 * of the C API. They internally call the same C functions we use here.
 *
 * HOW CLASSIFICATION WORKS:
 * -------------------------
 * The task (classification vs regression) is determined by the OBJECTIVE parameter:
 *
 * For Classification:
 *   XGBoosterSetParam(booster, "objective", "multi:softprob");  // Multi-class probabilities
 *   XGBoosterSetParam(booster, "num_class", "3");               // Number of classes
 *   XGBoosterPredict(...) -> Returns probabilities for each class
 *
 * For Regression:
 *   XGBoosterSetParam(booster, "objective", "reg:squarederror");
 *   XGBoosterPredict(...) -> Returns continuous values
 *
 * SAME FUNCTION, DIFFERENT BEHAVIOR based on objective!
 *
 * Common Classification Objectives:
 * - "binary:logistic"  -> Binary classification (outputs probability)
 * - "multi:softprob"   -> Multi-class (outputs probabilities for each class)
 * - "multi:softmax"    -> Multi-class (outputs class label directly)
 *
 * OUTPUT FORMAT:
 * --------------
 * For multi:softprob with 3 classes and 30 test samples:
 * - out_len = 90 (30 samples × 3 classes)
 * - out_result[0..2]   = Probabilities for sample 0 (class 0, 1, 2)
 * - out_result[3..5]   = Probabilities for sample 1 (class 0, 1, 2)
 * - out_result[6..8]   = Probabilities for sample 2 (class 0, 1, 2)
 * - ...and so on
 *
 * We then convert probabilities to class labels by finding argmax:
 *   predicted_class = argmax(probabilities)
 *
 * COMPARISON WITH PYTHON:
 * -----------------------
 * Python:
 *   clf = XGBClassifier(objective='multi:softprob', num_class=3)
 *   clf.fit(X_train, y_train)
 *   y_pred = clf.predict(X_test)  # Returns class labels automatically
 *
 * C++:
 *   XGBoosterSetParam(booster, "objective", "multi:softprob");
 *   XGBoosterSetParam(booster, "num_class", "3");
 *   XGBoosterUpdateOneIter(booster, iter, h_train);
 *   XGBoosterPredict(booster, h_test, 0, 0, 0, &len, &result);
 *   // Must manually convert probabilities to class labels (argmax)
 *
 * Both approaches use the same underlying C API functions!
 * ============================================================================
 */


// Iris dataset (150 samples, 4 features, 3 classes)
// Features: sepal length, sepal width, petal length, petal width
// Classes: 0=setosa, 1=versicolor, 2=virginica
const int IRIS_SAMPLES = 150;
const int IRIS_FEATURES = 4;
const int IRIS_CLASSES = 3;

const float IRIS_DATA[IRIS_SAMPLES][IRIS_FEATURES] = {
    // Setosa (0-49)
    {5.1f, 3.5f, 1.4f, 0.2f}, {4.9f, 3.0f, 1.4f, 0.2f}, {4.7f, 3.2f, 1.3f, 0.2f}, {4.6f, 3.1f, 1.5f, 0.2f},
    {5.0f, 3.6f, 1.4f, 0.2f}, {5.4f, 3.9f, 1.7f, 0.4f}, {4.6f, 3.4f, 1.4f, 0.3f}, {5.0f, 3.4f, 1.5f, 0.2f},
    {4.4f, 2.9f, 1.4f, 0.2f}, {4.9f, 3.1f, 1.5f, 0.1f}, {5.4f, 3.7f, 1.5f, 0.2f}, {4.8f, 3.4f, 1.6f, 0.2f},
    {4.8f, 3.0f, 1.4f, 0.1f}, {4.3f, 3.0f, 1.1f, 0.1f}, {5.8f, 4.0f, 1.2f, 0.2f}, {5.7f, 4.4f, 1.5f, 0.4f},
    {5.4f, 3.9f, 1.3f, 0.4f}, {5.1f, 3.5f, 1.4f, 0.3f}, {5.7f, 3.8f, 1.7f, 0.3f}, {5.1f, 3.8f, 1.5f, 0.3f},
    {5.4f, 3.4f, 1.7f, 0.2f}, {5.1f, 3.7f, 1.5f, 0.4f}, {4.6f, 3.6f, 1.0f, 0.2f}, {5.1f, 3.3f, 1.7f, 0.5f},
    {4.8f, 3.4f, 1.9f, 0.2f}, {5.0f, 3.0f, 1.6f, 0.2f}, {5.0f, 3.4f, 1.6f, 0.4f}, {5.2f, 3.5f, 1.5f, 0.2f},
    {5.2f, 3.4f, 1.4f, 0.2f}, {4.7f, 3.2f, 1.6f, 0.2f}, {4.8f, 3.1f, 1.6f, 0.2f}, {5.4f, 3.4f, 1.5f, 0.4f},
    {5.2f, 4.1f, 1.5f, 0.1f}, {5.5f, 4.2f, 1.4f, 0.2f}, {4.9f, 3.1f, 1.5f, 0.2f}, {5.0f, 3.2f, 1.2f, 0.2f},
    {5.5f, 3.5f, 1.3f, 0.2f}, {4.9f, 3.6f, 1.4f, 0.1f}, {4.4f, 3.0f, 1.3f, 0.2f}, {5.1f, 3.4f, 1.5f, 0.2f},
    {5.0f, 3.5f, 1.3f, 0.3f}, {4.5f, 2.3f, 1.3f, 0.3f}, {4.4f, 3.2f, 1.3f, 0.2f}, {5.0f, 3.5f, 1.6f, 0.6f},
    {5.1f, 3.8f, 1.9f, 0.4f}, {4.8f, 3.0f, 1.4f, 0.3f}, {5.1f, 3.8f, 1.6f, 0.2f}, {4.6f, 3.2f, 1.4f, 0.2f},
    {5.3f, 3.7f, 1.5f, 0.2f}, {5.0f, 3.3f, 1.4f, 0.2f},
    // Versicolor (50-99)
    {7.0f, 3.2f, 4.7f, 1.4f}, {6.4f, 3.2f, 4.5f, 1.5f}, {6.9f, 3.1f, 4.9f, 1.5f}, {5.5f, 2.3f, 4.0f, 1.3f},
    {6.5f, 2.8f, 4.6f, 1.5f}, {5.7f, 2.8f, 4.5f, 1.3f}, {6.3f, 3.3f, 4.7f, 1.6f}, {4.9f, 2.4f, 3.3f, 1.0f},
    {6.6f, 2.9f, 4.6f, 1.3f}, {5.2f, 2.7f, 3.9f, 1.4f}, {5.0f, 2.0f, 3.5f, 1.0f}, {5.9f, 3.0f, 4.2f, 1.5f},
    {6.0f, 2.2f, 4.0f, 1.0f}, {6.1f, 2.9f, 4.7f, 1.4f}, {5.6f, 2.9f, 3.6f, 1.3f}, {6.7f, 3.1f, 4.4f, 1.4f},
    {5.6f, 3.0f, 4.5f, 1.5f}, {5.8f, 2.7f, 4.1f, 1.0f}, {6.2f, 2.2f, 4.5f, 1.5f}, {5.6f, 2.5f, 3.9f, 1.1f},
    {5.9f, 3.2f, 4.8f, 1.8f}, {6.1f, 2.8f, 4.0f, 1.3f}, {6.3f, 2.5f, 4.9f, 1.5f}, {6.1f, 2.8f, 4.7f, 1.2f},
    {6.4f, 2.9f, 4.3f, 1.3f}, {6.6f, 3.0f, 4.4f, 1.4f}, {6.8f, 2.8f, 4.8f, 1.4f}, {6.7f, 3.0f, 5.0f, 1.7f},
    {6.0f, 2.9f, 4.5f, 1.5f}, {5.7f, 2.6f, 3.5f, 1.0f}, {5.5f, 2.4f, 3.8f, 1.1f}, {5.5f, 2.4f, 3.7f, 1.0f},
    {5.8f, 2.7f, 3.9f, 1.2f}, {6.0f, 2.7f, 5.1f, 1.6f}, {5.4f, 3.0f, 4.5f, 1.5f}, {6.0f, 3.4f, 4.5f, 1.6f},
    {6.7f, 3.1f, 4.7f, 1.5f}, {6.3f, 2.3f, 4.4f, 1.3f}, {5.6f, 3.0f, 4.1f, 1.3f}, {5.5f, 2.5f, 4.0f, 1.3f},
    {5.5f, 2.6f, 4.4f, 1.2f}, {6.1f, 3.0f, 4.6f, 1.4f}, {5.8f, 2.6f, 4.0f, 1.2f}, {5.0f, 2.3f, 3.3f, 1.0f},
    {5.6f, 2.7f, 4.2f, 1.3f}, {5.7f, 3.0f, 4.2f, 1.2f}, {5.7f, 2.9f, 4.2f, 1.3f}, {6.2f, 2.9f, 4.3f, 1.3f},
    {5.1f, 2.5f, 3.0f, 1.1f}, {5.7f, 2.8f, 4.1f, 1.3f},
    // Virginica (100-149)
    {6.3f, 3.3f, 6.0f, 2.5f}, {5.8f, 2.7f, 5.1f, 1.9f}, {7.1f, 3.0f, 5.9f, 2.1f}, {6.3f, 2.9f, 5.6f, 1.8f},
    {6.5f, 3.0f, 5.8f, 2.2f}, {7.6f, 3.0f, 6.6f, 2.1f}, {4.9f, 2.5f, 4.5f, 1.7f}, {7.3f, 2.9f, 6.3f, 1.8f},
    {6.7f, 2.5f, 5.8f, 1.8f}, {7.2f, 3.6f, 6.1f, 2.5f}, {6.5f, 3.2f, 5.1f, 2.0f}, {6.4f, 2.7f, 5.3f, 1.9f},
    {6.8f, 3.0f, 5.5f, 2.1f}, {5.7f, 2.5f, 5.0f, 2.0f}, {5.8f, 2.8f, 5.1f, 2.4f}, {6.4f, 3.2f, 5.3f, 2.3f},
    {6.5f, 3.0f, 5.5f, 1.8f}, {7.7f, 3.8f, 6.7f, 2.2f}, {7.7f, 2.6f, 6.9f, 2.3f}, {6.0f, 2.2f, 5.0f, 1.5f},
    {6.9f, 3.2f, 5.7f, 2.3f}, {5.6f, 2.8f, 4.9f, 2.0f}, {7.7f, 2.8f, 6.7f, 2.0f}, {6.3f, 2.7f, 4.9f, 1.8f},
    {6.7f, 3.3f, 5.7f, 2.1f}, {7.2f, 3.2f, 6.0f, 1.8f}, {6.2f, 2.8f, 4.8f, 1.8f}, {6.1f, 3.0f, 4.9f, 1.8f},
    {6.4f, 2.8f, 5.6f, 2.1f}, {7.2f, 3.0f, 5.8f, 1.6f}, {7.4f, 2.8f, 6.1f, 1.9f}, {7.9f, 3.8f, 6.4f, 2.0f},
    {6.4f, 2.8f, 5.6f, 2.2f}, {6.3f, 2.8f, 5.1f, 1.5f}, {6.1f, 2.6f, 5.6f, 1.4f}, {7.7f, 3.0f, 6.1f, 2.3f},
    {6.3f, 3.4f, 5.6f, 2.4f}, {6.4f, 3.1f, 5.5f, 1.8f}, {6.0f, 3.0f, 4.8f, 1.8f}, {6.9f, 3.1f, 5.4f, 2.1f},
    {6.7f, 3.1f, 5.6f, 2.4f}, {6.9f, 3.1f, 5.1f, 2.3f}, {5.8f, 2.7f, 5.1f, 1.9f}, {6.8f, 3.2f, 5.9f, 2.3f},
    {6.7f, 3.3f, 5.7f, 2.5f}, {6.7f, 3.0f, 5.2f, 2.3f}, {6.3f, 2.5f, 5.0f, 1.9f}, {6.5f, 3.0f, 5.2f, 2.0f},
    {6.2f, 3.4f, 5.4f, 2.3f}, {5.9f, 3.0f, 5.1f, 1.8f}
};

const float IRIS_LABELS[IRIS_SAMPLES] = {
    // Setosa (0)
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // Versicolor (1)
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    // Virginica (2)
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f
};

const char* CLASS_NAMES[IRIS_CLASSES] = {"Setosa", "Versicolor", "Virginica"};

// Function to calculate confusion matrix and metrics
void calculate_metrics(const std::vector<int>& y_true, const std::vector<int>& y_pred, int num_classes) {
    // Confusion matrix
    std::vector<std::vector<int>> confusion(num_classes, std::vector<int>(num_classes, 0));

    for (size_t i = 0; i < y_true.size(); i++) {
        confusion[y_true[i]][y_pred[i]]++;
    }

    log("\n=== Confusion Matrix ===");
    log("              Predicted");
    log("              Setosa  Versicolor  Virginica");
    for (int i = 0; i < num_classes; i++) {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Actual %-10s  %4d      %4d        %4d",
                 CLASS_NAMES[i], confusion[i][0], confusion[i][1], confusion[i][2]);
        log(buffer);
    }

    log("\n=== Classification Report ===");
    log("Class         Precision  Recall    F1-Score  Support");

    double total_precision = 0.0, total_recall = 0.0, total_f1 = 0.0;
    int total_support = 0;

    for (int i = 0; i < num_classes; i++) {
        int tp = confusion[i][i];
        int fp = 0, fn = 0;

        for (int j = 0; j < num_classes; j++) {
            if (j != i) {
                fp += confusion[j][i];
                fn += confusion[i][j];
            }
        }

        double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
        double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
        int support = tp + fn;

        total_precision += precision;
        total_recall += recall;
        total_f1 += f1;
        total_support += support;

        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%-12s   %.4f     %.4f    %.4f    %d",
                 CLASS_NAMES[i], precision, recall, f1, support);
        log(buffer);
    }

    // Macro average
    double macro_precision = total_precision / num_classes;
    double macro_recall = total_recall / num_classes;
    double macro_f1 = total_f1 / num_classes;

    char buffer[256];
    log("");
    snprintf(buffer, sizeof(buffer), "Macro avg      %.4f     %.4f    %.4f    %d",
             macro_precision, macro_recall, macro_f1, total_support);
    log(buffer);

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < num_classes; i++) {
        correct += confusion[i][i];
    }
    double accuracy = (double)correct / total_support;

    snprintf(buffer, sizeof(buffer), "\nAccuracy: %.4f", accuracy);
    log(buffer);
}

int main() {
    try {
        // Initialize logging
        initLogging("xgboost_log.txt");

        log("=== XGBoost Iris Classification (C++) ===\n");

        // Create indices for train/test split (80/20)
        // Using random_state=17 equivalent
        std::vector<int> indices(IRIS_SAMPLES);
        for (int i = 0; i < IRIS_SAMPLES; i++) {
            indices[i] = i;
        }

        // Shuffle with seed 17 (matching Python's random_state=17)
        std::mt19937 rng(17);
        std::shuffle(indices.begin(), indices.end(), rng);

        // Split into train (120 samples) and test (30 samples)
        const int train_size = static_cast<int>(IRIS_SAMPLES * 0.8);
        const int test_size = IRIS_SAMPLES - train_size;

        std::vector<float> X_train, X_test;
        std::vector<float> y_train, y_test;

        X_train.reserve(train_size * IRIS_FEATURES);
        X_test.reserve(test_size * IRIS_FEATURES);
        y_train.reserve(train_size);
        y_test.reserve(test_size);

        for (int i = 0; i < train_size; i++) {
            int idx = indices[i];
            for (int j = 0; j < IRIS_FEATURES; j++) {
                X_train.push_back(IRIS_DATA[idx][j]);
            }
            y_train.push_back(IRIS_LABELS[idx]);
        }

        for (int i = train_size; i < IRIS_SAMPLES; i++) {
            int idx = indices[i];
            for (int j = 0; j < IRIS_FEATURES; j++) {
                X_test.push_back(IRIS_DATA[idx][j]);
            }
            y_test.push_back(IRIS_LABELS[idx]);
        }

        logf("Dataset split: %d training samples, %d test samples\n", train_size, test_size);

        // Create DMatrix for training data
        DMatrixHandle h_train;
        safe_xgboost(XGDMatrixCreateFromMat(X_train.data(), train_size, IRIS_FEATURES, -1.0f, &h_train));
        safe_xgboost(XGDMatrixSetFloatInfo(h_train, "label", y_train.data(), train_size));
        log("Training DMatrix created.");

        // Create DMatrix for test data
        DMatrixHandle h_test;
        safe_xgboost(XGDMatrixCreateFromMat(X_test.data(), test_size, IRIS_FEATURES, -1.0f, &h_test));
        safe_xgboost(XGDMatrixSetFloatInfo(h_test, "label", y_test.data(), test_size));
        log("Test DMatrix created.");

        // Create booster
        BoosterHandle booster;
        DMatrixHandle cache[] = {h_train};
        safe_xgboost(XGBoosterCreate(cache, 1, &booster));
        log("Booster created.");

        // Set parameters for multi-class classification
        safe_xgboost(XGBoosterSetParam(booster, "objective", "multi:softprob"));
        safe_xgboost(XGBoosterSetParam(booster, "num_class", "3"));
        safe_xgboost(XGBoosterSetParam(booster, "eval_metric", "mlogloss"));
        safe_xgboost(XGBoosterSetParam(booster, "eta", "0.3"));  // learning rate
        safe_xgboost(XGBoosterSetParam(booster, "max_depth", "6"));
        safe_xgboost(XGBoosterSetParam(booster, "seed", "17"));
        log("Booster parameters set (objective=multi:softprob, num_class=3).\n");

        // Train the model
        const int num_rounds = 100;
        logf("Training for %d rounds...", num_rounds);

        for (int iter = 0; iter < num_rounds; iter++) {
            safe_xgboost(XGBoosterUpdateOneIter(booster, iter, h_train));

            // Print evaluation metrics every 10 iterations
            if ((iter + 1) % 20 == 0) {
                const char* eval_names[] = {"train", "test"};
                DMatrixHandle eval_dmats[] = {h_train, h_test};
                const char* eval_result;
                safe_xgboost(XGBoosterEvalOneIter(booster, iter, eval_dmats, eval_names, 2, &eval_result));
                logf("[%d] %s", iter + 1, eval_result);
            }
        }

        log("\nTraining completed.\n");

        // ========================================================================
        // MAKING PREDICTIONS FOR CLASSIFICATION
        // ========================================================================
        // XGBoosterPredict() is the UNIVERSAL prediction function for both
        // classification and regression. Since we set objective="multi:softprob",
        // it will return CLASS PROBABILITIES.
        //
        // Function signature:
        //   XGBoosterPredict(booster, dmat, option_mask, ntree_limit, training,
        //                    &out_len, &out_result)
        //
        // Parameters:
        //   - booster: The trained model
        //   - h_test: Test data matrix
        //   - 0 (option_mask): Normal prediction (not margin/contribution/leaf)
        //   - 0 (ntree_limit): Use all trees (not limiting to first N trees)
        //   - 0 (training): Not for training (this is inference)
        //   - &out_len: OUTPUT - length of result array
        //   - &out_result: OUTPUT - pointer to float array with predictions
        //
        // For multi-class classification with 3 classes:
        //   out_len = num_samples × num_classes = 30 × 3 = 90
        //   out_result = [p0_c0, p0_c1, p0_c2, p1_c0, p1_c1, p1_c2, ...]
        //     where p0_c0 = probability that sample 0 belongs to class 0
        //           p0_c1 = probability that sample 0 belongs to class 1
        //           p0_c2 = probability that sample 0 belongs to class 2
        //
        // This is equivalent to Python's:
        //   clf.predict_proba(X_test)  # Returns probability matrix
        // ========================================================================
        bst_ulong out_len;
        const float* out_result;
        safe_xgboost(XGBoosterPredict(booster, h_test, 0, 0, 0, &out_len, &out_result));

        // Convert probabilities to class predictions (argmax)
        // This is equivalent to Python's clf.predict(X_test)
        std::vector<int> y_pred;
        for (int i = 0; i < test_size; i++) {
            // Find the class with highest probability
            int max_class = 0;
            float max_prob = out_result[i * IRIS_CLASSES];
            for (int j = 1; j < IRIS_CLASSES; j++) {
                if (out_result[i * IRIS_CLASSES + j] > max_prob) {
                    max_prob = out_result[i * IRIS_CLASSES + j];
                    max_class = j;
                }
            }
            y_pred.push_back(max_class);
        }

        log("=== Test Set Predictions ===");
        log("Sample  Actual      Predicted   Probabilities (Setosa, Versicolor, Virginica)");
        for (int i = 0; i < std::min(10, test_size); i++) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "%-6d  %-11s %-11s [%.4f, %.4f, %.4f]",
                     i + 1,
                     CLASS_NAMES[static_cast<int>(y_test[i])],
                     CLASS_NAMES[y_pred[i]],
                     out_result[i * IRIS_CLASSES + 0],
                     out_result[i * IRIS_CLASSES + 1],
                     out_result[i * IRIS_CLASSES + 2]);
            log(buffer);
        }

        // ========================================================================
        // SINGLE EXAMPLE PREDICTION
        // ========================================================================
        // Demonstrating prediction on a single sample: [4.5, 3.0, 1.5, 0.25]
        // This is equivalent to Python's:
        //   X_example = np.array([4.5, 3.0, 1.5, 0.25]).reshape(1, 4)
        //   y_example = clf.predict(X_example)
        //   y_proba = clf.predict_proba(X_example)
        // ========================================================================
        log("\n=== Single Example Prediction ===");
        float X_example[] = {4.5f, 3.0f, 1.5f, 0.25f};
        DMatrixHandle h_example;
        safe_xgboost(XGDMatrixCreateFromMat(X_example, 1, IRIS_FEATURES, -1.0f, &h_example));

        bst_ulong example_len;
        const float* example_result;
        // Same XGBoosterPredict() function! Works for single or multiple samples
        safe_xgboost(XGBoosterPredict(booster, h_example, 0, 0, 0, &example_len, &example_result));
        // example_len = 3 (1 sample × 3 classes)
        // example_result = [prob_class_0, prob_class_1, prob_class_2]

        // Convert to class label (argmax of probabilities)
        int example_class = 0;
        float example_max_prob = example_result[0];
        for (int j = 1; j < IRIS_CLASSES; j++) {
            if (example_result[j] > example_max_prob) {
                example_max_prob = example_result[j];
                example_class = j;
            }
        }

        logf("Input: [%.1f, %.1f, %.1f, %.2f]", X_example[0], X_example[1], X_example[2], X_example[3]);
        logf("Predicted class: %s (class %d)", CLASS_NAMES[example_class], example_class);
        logf("Probabilities: [%.4f, %.4f, %.4f]",
             example_result[0], example_result[1], example_result[2]);

        // Calculate and display metrics
        std::vector<int> y_test_int(test_size);
        for (int i = 0; i < test_size; i++) {
            y_test_int[i] = static_cast<int>(y_test[i]);
        }

        calculate_metrics(y_test_int, y_pred, IRIS_CLASSES);

        // Cleanup
        XGDMatrixFree(h_train);
        XGDMatrixFree(h_test);
        XGDMatrixFree(h_example);
        XGBoosterFree(booster);

        log("\nAll XGBoost resources freed successfully.");
        closeLogging();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        log(std::string("Error: ") + e.what());
        closeLogging();
        return 1;
    }
}

