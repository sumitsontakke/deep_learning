Great question! Whether or not you need to convert your labels using `to_categorical(y, num_classes=10)` depends on the **loss function** and the **output layer activation** you are using.

---

### ✅ When You **Need** `to_categorical()`:

Use `to_categorical(y)` **only if**:
- Your labels are integers (e.g., `y = [0, 1, 2, ..., 9]`)
- Your model ends with `softmax` **and**
- You use **`categorical_crossentropy`** as the loss function

This is because `categorical_crossentropy` expects **one-hot encoded vectors** for labels, like:
```
Class 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

---

### ✅ When You **Do Not Need** `to_categorical()`:

You can skip `to_categorical()` **if**:
- Your labels are still integers (e.g., `y = [0, 1, 2, ..., 9]`)
- You use **`sparse_categorical_crossentropy`** as the loss
- Your model still ends with a `softmax` layer

This is what we did in your code:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

🧠 So, **`sparse_categorical_crossentropy` handles integer labels directly** — no need to convert to one-hot.

---

### 💡 Summary Table:

| Label Format | Loss Function                 | Use `to_categorical()`? |
|--------------|-------------------------------|--------------------------|
| Integers     | `sparse_categorical_crossentropy` | ❌ No                    |
| One-hot      | `categorical_crossentropy`     | ✅ Yes                   |

---
