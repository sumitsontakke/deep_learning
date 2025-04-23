# deep_learning
I'm collecting all my practice and learning code in this repository.

Working with AI tools: ChatGPT & Microsoft Copilot


---

# 🧠 Deep Learning Cheat Sheet – Classification Focus

---

Layer/Step | Purpose
--- | ---
Conv2D | Detect spatial features (edges, textures)
BatchNormalization | Normalize activations → faster convergence and stability
MaxPooling2D | Downsample → reduce dimension, preserve features
Dropout | Prevent overfitting
Dense(10) | Final classification into 10 CIFAR-10 classes with softmax
Adam Optimizer | Efficient, adaptive optimizer with great performance

## ✅ Classification Loss Functions

| Problem Type           | Loss Function                    | Use When… |
|------------------------|----------------------------------|-----------|
| **Binary**             | `binary_crossentropy`           | 0/1 labels |
|                        | `hinge`, `squared_hinge`        | For SVM-like models |
|                        | `focal_loss`                    | Class imbalance |
| **Multiclass (sparse)**| `sparse_categorical_crossentropy` | Class index (e.g., `3`) |
| **Multiclass (one-hot)**| `categorical_crossentropy`      | One-hot labels |
| **Multi-label**        | `binary_crossentropy`           | Multi-hot vector (e.g., `[0, 1, 0]`) |

---

## ⚙️ Optimizers

| Optimizer    | Best For                           | Notes |
|--------------|------------------------------------|-------|
| `SGD`        | Simple tasks, low resource usage   | + momentum helps |
| `RMSprop`    | RNNs, noisy data                   | Adaptive learning |
| `Adam`       | General purpose (default)          | Fast convergence |
| `AdamW`      | LLMs, transformers                 | Adam + weight decay |
| `Adagrad`    | Sparse gradients (NLP)             | Learning rate shrinks over time |
| `Nadam`      | Like Adam + Nesterov momentum      | Stable |

---

## 🧪 `model.compile()` – Key Parameters

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

| Argument         | Purpose |
|------------------|---------|
| `optimizer`       | Update rule (e.g. `'adam'`, `'sgd'`, `Adam(learning_rate=0.001)`) |
| `loss`            | Training objective (e.g. `'binary_crossentropy'`) |
| `metrics`         | Trackable metrics (e.g. `'accuracy'`, `'precision'`, `'AUC'`) |
| `loss_weights`    | For multi-output models |
| `run_eagerly`     | For debugging, disables tf graph |

---

## 🎯 Real-World Applications (w/ Loss + Optimizer)

| Task                        | Loss Function                      | Optimizer |
|-----------------------------|------------------------------------|-----------|
| Image classification (MNIST)| `sparse_categorical_crossentropy` | `adam`    |
| Text sentiment (binary)     | `binary_crossentropy`             | `adam`    |
| Object detection            | `focal_loss`                      | `adam`    |
| Multi-label disease tagging | `binary_crossentropy`             | `adam` or `rmsprop` |
| Class imbalance (fraud)     | `weighted crossentropy` or `focal_loss` | `adam` |

---

## 🔧 Metrics You Can Use

```python
metrics=['accuracy', 'precision', 'recall', 'AUC']
```

---

## 🧑‍💻 Next Steps: Practice Ideas

We’ll tackle 1–2 problems from each of these:
- ✅ MNIST (already done)
- 📦 [Kaggle] Fashion MNIST
- 📦 [Kaggle] Toxic Comment (multi-label)
- 📦 [Kaggle] Chest X-ray (imbalanced, binary)
- 📦 [Kaggle] Leaf Classification (multiclass)

---

Metric | Formula | Meaning
--- | --- | ---
Precision | TP / (TP + FP) | Out of all predicted as class X, how many were correct? (Low FP desired)
Recall | TP / (TP + FN) | Out of all actual class X, how many did the model catch? (Low FN desired)
F1-Score | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean → balances Precision and Recall
Support | Count of true samples per class | Helps interpret metrics better for imbalanced datasets


Metric | What it Tells You | Usefulness
--- | --- | ---
Precision | Out of all predicted for a class, how many were correct? | High precision = low false positives. Useful when false positives are costly.
Recall | Out of all actual items of a class, how many did we catch? | High recall = low false negatives. Useful when missing a positive case is risky.
F1-Score | Harmonic mean of precision and recall | Balances both; useful when data is imbalanced.
Support | Number of actual instances of each class | Shows how many test examples belong to that class.
Accuracy | Overall ratio of correctly predicted items | Good general metric, but not enough for imbalanced data.
Macro Avg | Average of metrics treating all classes equally | Ignores class imbalance. Good to compare per-class performance.
Weighted Avg | Average of metrics weighted by support | Reflects class imbalance, better summary than macro in real-world use.

---
# CNN
Parameter | Guideline
--- | ---
filters | Start small: 32 → 64 → 128 → (256)*
kernel_size | (3,3) is standard; rarely need bigger
activation | Always use 'relu'
padding | Use 'same' to maintain image size
input_shape | Only for the first Conv layer


Decision Point | Guideline
--- | ---
🔢 Number of hidden layers | 2–4 layers for starting ANN model
💡 Neurons per layer | Start with 512 → 256 → 128 pattern (powers of 2)
🧂 Add BatchNorm? | Yes, especially after Dense+ReLU
💧 Dropout? | Yes, e.g., Dropout(0.3–0.5) to avoid overfitting
🧠 Activation functions | ReLU works well; softmax for final
🏁 Final layer | Dense(100, activation='softmax')


# CNN Experimentations
Option | Ideas
--- | ---
✨ Add More Conv Blocks | Add one more Conv2D(256) block if you're confident
📉 Add EarlyStopping | Avoid overfitting
🧪 Batch size | Try 32, 64
🎯 Learning Rate | Use Adam(lr=0.0005) or a scheduler
🔍 Data Augmentation | Huge for CIFAR-100 — will boost accuracy
📈 Try more epochs | Go 50–100 with patience & EarlyStopping

---
