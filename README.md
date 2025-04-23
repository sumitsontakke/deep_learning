# deep_learning
I'm collecting all my practice and learning code in this repository.

Working with AI tools: ChatGPT & Microsoft Copilot


---

# 🧠 Deep Learning Cheat Sheet – Classification Focus

---

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
# CNN Experimentations
Option | Ideas
--- | ---
✨ Add More Conv Blocks | Add one more Conv2D(256) block if you're confident
📉 Add EarlyStopping | Avoid overfitting
🧪 Batch size | Try 32, 64
🎯 Learning Rate | Use Adam(lr=0.0005) or a scheduler
🔍 Data Augmentation | Huge for CIFAR-100 — will boost accuracy
📈 Try more epochs | Go 50–100 with patience & EarlyStopping
