import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz

# 1. Load Dataset
df = pd.read_csv("heart.csv")
df.columns = df.columns.str.strip()  # Remove spaces from column names
print("Columns in dataset:", df.columns.tolist())

# 2. Preprocessing
X = df.drop("target", axis=1)
y = df["target"]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Decision Tree
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)

# 5. Evaluate Decision Tree
y_pred_dt = dtree.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# 6. Visualize Decision Tree
export_graphviz(dtree, out_file="tree_visualization.dot", 
                feature_names=X.columns, class_names=['No Disease', 'Disease'],
                filled=True, rounded=True)

with open("tree_visualization.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph).render("tree_visualization", format='png', cleanup=True)

# 7. Train Random Forest
rforest = RandomForestClassifier(n_estimators=100, random_state=42)
rforest.fit(X_train, y_train)

# 8. Evaluate Random Forest
y_pred_rf = rforest.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# 9. Feature Importance
importances = rforest.feature_importances_
features = pd.Series(importances, index=X.columns)
features.sort_values().plot(kind='barh', title='Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

# 10. Cross-validation
scores = cross_val_score(rforest, X, y, cv=5)
print("\nCross-validation scores (Random Forest):", scores)
print("Average CV score:", scores.mean())
