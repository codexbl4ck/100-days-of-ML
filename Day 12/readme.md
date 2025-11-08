# 🌿 Day 12 – Feature Scaling

### 🎯 Goal
Learn how to make all features contribute equally to a machine learning model by putting them on the same scale.

---

### 🧠 What Is Feature Scaling?
In real datasets, features have different units or ranges (like age in years vs. income in rupees).  
Without scaling, models give more importance to features with larger numbers.  
Feature scaling fixes this by making all features fair and comparable.

---

### ⚙️ Types of Scaling
| Type | Range | Use Case |
|------|--------|-----------|
| **Normalization** | 0 → 1 | For KNN, SVM, Neural Networks |
| **Standardization** | Around 0 (mean=0, std=1) | For Linear Models, Logistic Regression, PCA |

---

### 🧩 Key Takeaways
- Always scale **after train-test split** to prevent data leakage.  
- Scaling improves **accuracy**, **speed**, and **stability** of models.  
- It’s a small preprocessing step with a **big impact**.

---

👨‍💻 *Author:* [@codexbl4ck](https://github.com/codexbl4ck)  
📅 *100 Days of ML – Day 12*
