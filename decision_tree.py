import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_text

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

#LOAD DATASET

df = pd.read_csv("animal_disease_dataset.csv")

print(df.head())

df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

df = df.drop_duplicates()

print("\nMissing Values:")
print(df.isnull().sum())

#FEATURES AND TARGET

X = df.drop("Disease", axis=1)
y = df["Disease"]

#ONE HOT ENCODING

categorical_cols = X.select_dtypes(exclude=np.number).columns

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

encoded_data = encoder.fit_transform(X[categorical_cols])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(categorical_cols)
)

X = X.drop(categorical_cols, axis=1)

X = pd.concat(
    [
        X.reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ],
    axis=1
)

print("\nEncoded Dataset:")
print(X.head())

#FEATURE SCALING

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

print("\nFeature Scaling Complete")
print("Shape:", X.shape)

#TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    shuffle=True
)

#DECISION TREE

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

#PREDICTIONS

y_pred = clf.predict(X_test)

#CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

print("\nConfusion Matrix:")
print(cm)

#EVALUATION

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nPrecision:")
print(
    precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )
)

print("\nRecall:")
print(
    recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )
)

print("\nF1 Score:")
print(
    f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#PLOT TREE

plt.figure(figsize=(20,10))

plot_tree(
    clf,
    filled=True,
    max_depth=3,
    fontsize=8
)

plt.show()

#TREE RULES

feature_names = []

feature_names.extend(["Age", "Temperature"])

feature_names.extend(
    encoder.get_feature_names_out(categorical_cols)
)

tree_rules = export_text(
    clf,
    feature_names=list(feature_names)
)

print("\nDecision Tree Rules:")
print(tree_rules)

#SAMPLE PREDICTION

sample_data = pd.DataFrame({
    'Animal': ['Buffalo'],
    'Age': [5],
    'Temperature': [103.5],
    'Symptom 1': ['fever'],
    'Symptom 2': ['cough'],
    'Symptom 3': ['weakness']
})

encoded_sample = encoder.transform(
    sample_data[categorical_cols]
)

encoded_sample_df = pd.DataFrame(
    encoded_sample,
    columns=encoder.get_feature_names_out(categorical_cols)
)

sample_data = sample_data.drop(
    categorical_cols,
    axis=1
)

sample_data = pd.concat(
    [
        sample_data.reset_index(drop=True),
        encoded_sample_df.reset_index(drop=True)
    ],
    axis=1
)

sample_data = scaler.transform(sample_data)

prediction = clf.predict(sample_data)

print("\nPredicted Disease:")
print(prediction)