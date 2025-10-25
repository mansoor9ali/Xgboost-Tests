# XGBoost C API vs Python API - Quick Reference

## The Key Question: Why No XGBClassifier in C++?

**Answer:** The XGBoost C API is a **low-level, procedural interface** designed to be language-agnostic. It doesn't have separate `XGBClassifier` and `XGBRegressor` classes like Python. Instead, it uses a **unified interface** where the task type is determined by parameters.

---

## API Architecture

```
High Level (Python):  XGBClassifier, XGBRegressor, XGBRanker
                              ↓
Middle Layer:         Python bindings (ctypes)
                              ↓
Low Level (C API):    XGBoosterCreate, XGBoosterPredict, etc.
                              ↓
Core (C++):           Learner class, GradientBooster, etc.
```

---

## Classification vs Regression: How It Works

### The SAME functions handle both tasks!

The difference is in the **objective parameter**:

| Task Type            | Objective Parameter       | Output                          |
|---------------------|---------------------------|----------------------------------|
| Binary Classification| `binary:logistic`         | Probability of class 1          |
| Multi-class (probabilities) | `multi:softprob`   | Probabilities for each class    |
| Multi-class (labels) | `multi:softmax`          | Class label (0, 1, 2, ...)      |
| Regression          | `reg:squarederror`        | Continuous values               |
| Regression          | `reg:linear` (deprecated) | Continuous values               |

---

## Python vs C++ Comparison

### Classification Example

#### Python (High-level):
```python
from xgboost import XGBClassifier

# Create classifier
clf = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    max_depth=6,
    learning_rate=0.3,
    n_estimators=100
)

# Train
clf.fit(X_train, y_train)

# Predict class labels
y_pred = clf.predict(X_test)

# Predict probabilities
y_proba = clf.predict_proba(X_test)
```

#### C++ (Low-level):
```cpp
#include <xgboost/c_api.h>

// Create DMatrix
DMatrixHandle h_train;
XGDMatrixCreateFromMat(X_train, rows, cols, -1.0f, &h_train);
XGDMatrixSetFloatInfo(h_train, "label", y_train, rows);

// Create booster
BoosterHandle booster;
DMatrixHandle cache[] = {h_train};
XGBoosterCreate(cache, 1, &booster);

// Set parameters (THIS determines it's classification!)
XGBoosterSetParam(booster, "objective", "multi:softprob");
XGBoosterSetParam(booster, "num_class", "3");
XGBoosterSetParam(booster, "max_depth", "6");
XGBoosterSetParam(booster, "eta", "0.3");  // learning_rate

// Train for 100 iterations
for (int iter = 0; iter < 100; iter++) {
    XGBoosterUpdateOneIter(booster, iter, h_train);
}

// Predict probabilities
bst_ulong out_len;
const float* out_result;
XGBoosterPredict(booster, h_test, 0, 0, 0, &out_len, &out_result);

// out_len = n_samples × n_classes
// out_result = [p0_c0, p0_c1, p0_c2, p1_c0, p1_c1, p1_c2, ...]

// Convert to class labels (argmax)
for (int i = 0; i < n_samples; i++) {
    int max_class = 0;
    float max_prob = out_result[i * num_classes];
    for (int j = 1; j < num_classes; j++) {
        if (out_result[i * num_classes + j] > max_prob) {
            max_prob = out_result[i * num_classes + j];
            max_class = j;
        }
    }
    // max_class is the predicted label
}

// Cleanup
XGDMatrixFree(h_train);
XGBoosterFree(booster);
```

---

## XGBoosterPredict Function Explained

```cpp
int XGBoosterPredict(
    BoosterHandle handle,      // The trained model
    DMatrixHandle dmat,        // Input data matrix
    int option_mask,           // Prediction type (0=normal, 1=margin, etc.)
    unsigned ntree_limit,      // Number of trees to use (0=all)
    int training,              // Is this for training? (0=no, 1=yes)
    bst_ulong *out_len,        // OUTPUT: Length of result array
    const float **out_result   // OUTPUT: Pointer to prediction array
);
```

### Parameters:

- **handle**: Your trained booster
- **dmat**: Data to predict on (wrapped in DMatrix)
- **option_mask**: 
  - `0` = normal prediction
  - `1` = output margin (raw scores before objective function)
  - `2` = predict contribution (SHAP values)
  - `4` = predict leaf index
- **ntree_limit**: Use first N trees (0 = use all trees)
- **training**: Set to 1 if prediction is for training (e.g., DART dropout), 0 otherwise
- **out_len**: Receives the length of output array
- **out_result**: Receives pointer to output array

### Output Format:

#### For Binary Classification (binary:logistic):
```
out_len = n_samples
out_result = [p1, p2, p3, ...]  // Probability of class 1 for each sample
```

#### For Multi-class (multi:softprob):
```
out_len = n_samples × n_classes
out_result = [
    p0_c0, p0_c1, p0_c2,  // Sample 0: probabilities for class 0, 1, 2
    p1_c0, p1_c1, p1_c2,  // Sample 1: probabilities for class 0, 1, 2
    ...
]
```

#### For Multi-class (multi:softmax):
```
out_len = n_samples
out_result = [0, 2, 1, ...]  // Class labels directly
```

#### For Regression:
```
out_len = n_samples
out_result = [3.14, 2.71, 1.41, ...]  // Continuous values
```

---

## Common Objectives Reference

### Classification:
- `binary:logistic` - Binary classification, outputs probability
- `binary:logitraw` - Binary classification, outputs raw score (before sigmoid)
- `multi:softprob` - Multi-class, outputs probability vector
- `multi:softmax` - Multi-class, outputs class label
- `rank:pairwise` - Learning to rank

### Regression:
- `reg:squarederror` - L2 loss (recommended)
- `reg:squaredlogerror` - L2 loss on log(pred + 1)
- `reg:logistic` - Logistic regression for regression
- `reg:pseudohubererror` - Huber loss
- `reg:absoluteerror` - L1 loss (MAE)
- `reg:quantileerror` - Quantile loss

---

## Why This Design?

### Advantages of C API approach:

✅ **Language-agnostic**: Can be called from ANY language (Python, R, Java, Julia, C#, etc.)
✅ **Unified interface**: Same functions for all tasks
✅ **Minimal API surface**: Fewer functions to learn
✅ **Maximum flexibility**: Full control over all parameters
✅ **Performance**: No abstraction overhead

### Disadvantages:

❌ **More verbose**: Requires more code than Python
❌ **Manual memory management**: Must free resources
❌ **No scikit-learn compatibility**: No `.fit()`, `.predict()` methods
❌ **Manual helper functions**: Must implement your own metrics, plotting, etc.

---

## Summary

| Python API | C API | How It Works |
|-----------|-------|--------------|
| `XGBClassifier()` | No equivalent | Use `XGBoosterCreate()` + set `objective="multi:softprob"` |
| `XGBRegressor()` | No equivalent | Use `XGBoosterCreate()` + set `objective="reg:squarederror"` |
| `clf.fit()` | `XGBoosterUpdateOneIter()` | Call in a loop for each iteration |
| `clf.predict()` | `XGBoosterPredict()` | Same function for classification & regression |
| `clf.predict_proba()` | `XGBoosterPredict()` | With `objective="multi:softprob"` |

**The key insight:** Python's `XGBClassifier` and `XGBRegressor` are just convenient wrappers around the same C API functions you use in C++!

---

## Further Reading

- [XGBoost C API Documentation](https://xgboost.readthedocs.io/en/stable/c.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- Source code: `xgboost/include/xgboost/c_api.h`

