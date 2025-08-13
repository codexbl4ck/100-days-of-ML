# Day 10 – Random Forest Classifier 🌳🌳🌳
Goal: Learn and implement Random Forest to classify flowers from the Iris dataset.

## What I did today

• Loaded the Iris dataset (built into scikit-learn).

• Split the data into training (80%) and testing (20%) sets.

• Created a RandomForestClassifier with 100 decision trees.

• Trained the model and checked accuracy.

• Predicted the species of a new flower.

## Key Notes

• Random Forest = Many Decision Trees working together (majority vote).

• test_size=0.2 → 20% data for testing.

• random_state=42 → Makes results repeatable.

• Very high accuracy because Iris is a simple dataset.

# Example Prediction
For flower:
[5.0, 1.4, 1.7, 0.5] → Predicted species: Setosa ✅