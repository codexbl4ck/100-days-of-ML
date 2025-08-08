# Day 7 – K-Nearest Neighbors (KNN)

## 📌 Objective
Predict whether a fruit is **Apple** or **Orange** based on its **weight** and **texture** using the KNN algorithm.

---

## 🛠 Steps
1. **Load Dataset** using `pandas`.
2. **Separate Features (X)** → Weight, Texture  
   **Target (y)** → Fruit label.
3. **Split** into training & test sets (80% / 20%).
4. **Create Model** → `KNeighborsClassifier(n_neighbors=3)`.
5. **Train** the model on the training set.
6. **Predict** on the test set & evaluate with accuracy.
7. **Test** with a new fruit (e.g., 155g Smooth).