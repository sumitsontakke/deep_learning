# 🧠 ANN Deep Learning Cheat Sheet (Code + Justification)

---

### 1. **Load Dataset**
```python
import pandas as pd
df = pd.read_csv("file.csv")  # For CSV tabular data
```
or
```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
> ✅ *Use Keras datasets for quick experimentation; CSV when using real-world or tabular data.*

---

### 2. **Normalize and Scale**
**Image Classification:**
```python
X = X.astype('float32') / 255
```
**Tabular Regression:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
```
> ✅ *Normalize image pixel values [0,255] → [0,1]; scale regression features to speed up convergence.*

---

### 3. **Split Dataset**
```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```
or inside `model.fit()`:
```python
model.fit(X_train, y_train, validation_split=0.2)
```
> ✅ *Split data to evaluate performance on unseen data (validation and test sets).*

---

### 4. **Build ANN Model**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, LeakyReLU
```

#### 4.1 **Flatten Layer**
```python
model.add(Flatten(input_shape=(28, 28)))
```
> ✅ *Converts multi-dimensional input into 1D vector (needed before Dense layers).*

#### 4.2 **Dense + Activation Functions**
```python
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
```

| Activation     | When to Use? |
|----------------|--------------|
| `ReLU`         | Default choice for hidden layers |
| `LeakyReLU`    | If you suspect dying ReLU problem (neurons stuck with 0) |
| `PReLU`        | Like LeakyReLU but slope is learnable |
| `Sigmoid`      | Binary output neuron |
| `Softmax`      | Multi-class classification output |

#### 4.3 **BatchNormalization**
```python
model.add(BatchNormalization())
```
> ✅ *Normalizes activations layer-wise → stabilizes learning & accelerates training.*

#### 4.4 **Dropout**
```python
model.add(Dropout(0.3))
```
> ✅ *Randomly disables neurons during training → combats overfitting.*

#### 4.5 **Output Layer**
```python
# Binary classification
model.add(Dense(1, activation='sigmoid'))

# Multi-class classification (10 classes)
model.add(Dense(10, activation='softmax'))
```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, PReLU

model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flattens 2D image to 1D vector for Dense layer
    
    Dense(128),
    # Activation Functions
    # ReLU for general use — avoids vanishing gradient
    # LeakyReLU if some neurons die (sparse activations or many zeroes)
    # PReLU if LeakyReLU improves performance further
    LeakyReLU(alpha=0.1),

    BatchNormalization(),  # Stabilizes and accelerates training
    Dropout(0.3),          # Prevents overfitting
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')  # Binary classification (change for multiclass)
])
```
---

### 5. **Compile the Model**
```python
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
```

| Optimizer  | When to Use?              |
|------------|---------------------------|
| `adam`     | Default, fast convergence |
| `sgd`      | Good with momentum/decay  |
| `rmsprop`  | For RNNs or noisy data    |

| Loss Function                | Use Case                        |
|-----------------------------|----------------------------------|
| `sparse_categorical_crossentropy` | Multi-class (labels not one-hot) |
| `categorical_crossentropy`       | Multi-class (one-hot labels)     |
| `binary_crossentropy`            | Binary classification            |
| `mse`                            | Regression                       |

---

### 6. **Train the Model**
```python
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
```

---

### 7. **Evaluate the Model**
```python
model.evaluate(X_test, y_test)
```

Sure! Here's a **brief Evaluate section** with code snippets and justifications for each evaluation metric:

---

### **6. Evaluate Model**

#### ✅ **Code**
```python
# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

#### 📌 **Justification**
- `test_loss`: Indicates how well the model is performing on unseen data using the same loss function used during training.
  - Lower value → Better fit.
  - Used to check for **overfitting/underfitting** (e.g., low train loss but high test loss = overfit).
  
- `test_accuracy`: Fraction of correct predictions over total predictions on test data.
  - Good for **classification problems**.
  - Doesn't tell **which class** model is struggling with — pair it with **confusion matrix** for clarity.

---

### 🔍 Other Useful Metrics
```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test).argmax(axis=1)  # For softmax outputs
print(classification_report(y_test, y_pred))
```

#### 📌 **Justification**
- `precision`: Out of predicted positives, how many are correct.
  - High precision = fewer false positives.
  - Important when **false positives are costly** (e.g., spam detection).

- `recall`: Out of actual positives, how many are predicted correctly.
  - High recall = fewer false negatives.
  - Crucial for **medical diagnoses**, fraud detection.

- `f1-score`: Harmonic mean of precision and recall.
  - Balances false positives and false negatives.
  - Useful when there's **class imbalance**.

- `support`: Number of true instances for each class.
  - Shows class **distribution** in the test set.


---

### 8. **Hyperparameter Tuning**
**GridSearchCV** (Scikit-learn):
```python
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
model = KerasClassifier(model=build_model)
grid = GridSearchCV(model, param_grid, cv=3)
```

**KerasTuner**:
```python
from keras_tuner import RandomSearch
tuner = RandomSearch(build_model, objective='val_accuracy', ...)
```
```python
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(hp.Int('units', 64, 256, step=64), activation='relu'))
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(hp.Choice('lr', [1e-3, 1e-4])), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=5, directory='my_dir', project_name='ann_tune')
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)
best_model = tuner.get_best_models(num_models=1)[0]
```
---

### 9. **Train + Evaluate Best Model**
```python
best_model = tuner.get_best_models(num_models=1)[0]
best_model.evaluate(X_test, y_test)
```

---

### 10. **Plot Results**
```python
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend(); plt.show()
```

### Predictions
```python
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int")  # For binary classification
```

A **brief Predictions section** tailored for both **classification and regression** use cases, with code and explanations:

---

### **9. Predictions**

#### ✅ **Classification Predictions**
```python
# Predict class probabilities
pred_probs = model.predict(X_test)

# Convert softmax probabilities to class labels
y_pred = pred_probs.argmax(axis=1)
```

#### 📌 **Justification**
- `model.predict(X_test)` returns **probability distributions** (softmax output) for each class.
- `argmax(axis=1)` picks the **index (class label)** with the highest probability per sample.
- ✅ Use `argmax` when:
  - Final layer is **`Dense(units=n_classes, activation='softmax')`**
  - You're solving a **multi-class classification problem**
  - You want the **predicted class**, not the probabilities.

---





#### ✅ **Binary Classification**
```python
# Sigmoid output → returns probabilities between 0 and 1
y_pred = (model.predict(X_test) > 0.5).astype("int32")
```

#### 📌 **Justification**
- Threshold at `0.5` is common; you can adjust it based on **precision-recall tradeoff**.
- Final layer should be: `Dense(1, activation='sigmoid')`.

---

#### ✅ **Regression Predictions**
```python
# Predict continuous output
y_pred = model.predict(X_test)
```

#### 📌 **Justification**
- No need for `argmax` or thresholding.
- Output is a **real number** (or vector).
- Final layer typically: `Dense(1)` with **no activation** or sometimes `relu` (if outputs must be non-negative).


---













### 11. **Confusion Matrix**
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
```

A **brief Confusion Matrix section** with code, justifications, and guidance on how to read and interpret it:

---

### **10. Confusion Matrix**

#### ✅ **Generate Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_test and y_pred are available
cm = confusion_matrix(y_test, y_pred)

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---

#### ✅ **Classification Report (Optional but Recommended)**
```python
print(classification_report(y_test, y_pred))
```

---

### 📌 **How to Read the Confusion Matrix**

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)     | False Negative (FN)    |
| **Actual Negative** | False Positive (FP)    | True Negative (TN)     |

---

### 📌 **Key Metrics & Justification**
- **True Positive (TP)**: Correctly predicted as positive (actual = positive, predicted = positive)
- **True Negative (TN)**: Correctly predicted as negative
- **False Positive (FP)**: Incorrectly predicted as positive (Type I Error)
- **False Negative (FN)**: Incorrectly predicted as negative (Type II Error)

---

### 📌 **Why It Matters**
- **Precision** = TP / (TP + FP) → Measures **exactness**, important when **false positives** are costly (e.g., spam detection).
- **Recall** = TP / (TP + FN) → Measures **completeness**, critical when **false negatives** are dangerous (e.g., cancer detection).
- **F1-score** = Harmonic mean of precision and recall → Use when you need a **balance** between precision and recall.
- **Accuracy** = (TP + TN) / Total → Useful when classes are **balanced**.

---

### 12. Callbacks
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, callbacks=[early_stop, checkpoint])
```
---
