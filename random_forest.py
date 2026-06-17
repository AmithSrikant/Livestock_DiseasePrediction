import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# STEP 1 : LOAD DATASET

df = pd.read_csv("animal_disease_dataset.csv")

print(df.head())

df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

df = df.drop_duplicates()

print("\nMissing Values:")
print(df.isnull().sum())

# STEP 2 : FEATURES AND TARGET

X = df.drop("Disease", axis=1)
y = df["Disease"]

# STEP 3 : ONE HOT ENCODING

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

# STEP 4 : FEATURE SCALING

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

print("\nFeature Scaling Complete")
print("Shape:", X.shape)

# STEP 5 : TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    shuffle=True,
    stratify=y
)

# STEP 6 : RANDOM FOREST

model = RandomForestClassifier(
    n_estimators=100,
    random_state=1
)

model.fit(X_train, y_train)

print("\nTraining Accuracy:")
print(model.score(X_train, y_train))

print("\nTesting Accuracy:")
print(model.score(X_test, y_test))

# STEP 7 : PREDICTIONS

y_pred = model.predict(X_test)

# STEP 8 : CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

print("\nConfusion Matrix:")
print(cm)

# STEP 9 : EVALUATION

print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)

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
print(
    classification_report(
        y_test,
        y_pred
    )
)

# STEP 10 : FEATURE IMPORTANCE

feature_names = ["Age", "Temperature"]

feature_names.extend(
    encoder.get_feature_names_out(categorical_cols)
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(
    importance_df.head(10)
)

# STEP 11 : SAMPLE PREDICTION

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

prediction = model.predict(sample_data)

print("\nPredicted Disease:")
print(prediction)

# STEP 12 : SAVE MODEL

joblib.dump(model, "livestock_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel Saved Successfully")