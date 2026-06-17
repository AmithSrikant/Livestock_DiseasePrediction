import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC #unit 3
from sklearn.model_selection import GridSearchCV #Unit 3
from sklearn.metrics import classification_report

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
'''step 1: '''
df = pd.read_csv("animal_disease_dataset.csv")

print(df.head())

# Remove unwanted column if present
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

# Remove duplicate rows
df = df.drop_duplicates()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

'''step 2: '''
#Encoding
#LabelEncoder
# le= LabelEncoder()
# categorical_cols= df.select_dtypes(exclude=np.number).columns
#
# for col in categorical_cols:
#     df[col] = le.fit_transform(df[col])
#
# print("\n LabelEncoder Dataset:")
# print(df.head())

#OneHotEncoder

X= df.drop("Disease", axis=1)
y= df["Disease"]

categorical_cols= X.select_dtypes(exclude=np.number).columns

encoder= OneHotEncoder( sparse_output= False, handle_unknown='ignore')

encoded_data= encoder.fit_transform(X[categorical_cols])

encoded_df= pd.DataFrame(encoded_data,
                         columns= encoder.get_feature_names_out(categorical_cols)
                         )
X= X.drop(categorical_cols, axis= 1)

X= pd.concat(
    [X.reset_index(drop= True), encoded_df.reset_index(drop= True)], axis= 1
)

print("\nEncoded Dataset:")
print(X.head())

#MinMaxScaler
scaler = MinMaxScaler()

X = scaler.fit_transform(X)

print("\nFeature Scaling Complete")
print("Shape of Dataset:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    shuffle=True,
    stratify=y
)

# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred = gnb.predict(X_test)

'''SVM + hyperparameter tuning'''#unit 3
params = {
    'kernel': ['linear', 'rbf'],
    'C': [0.01, 0.1, 1.0, 10],
    'gamma': ['auto', 'scale']
}

model = GridSearchCV(
    SVC(),
    param_grid=params,
    cv=2,
    scoring='accuracy'
)

model.fit(X_train, y_train)

best_model = model.best_estimator_

print("\nBest Model:")
print(best_model)

print("\nTraining Accuracy:")
print(model.score(X_train, y_train))

print("\nTesting Accuracy:")
print(model.score(X_test, y_test))

'''PREDICTIONS'''

y_pred = best_model.predict(X_test)

'''confusion matrix'''
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#a=(correct prediction/total prediction)
print("\nAccuracy:",
      accuracy_score(y_test, y_pred))

#p= (tp/tp+fp)
print("Precision:",
      precision_score(
          y_test,
          y_pred,
          average='weighted',
          zero_division=0
      ))

#r= (tp/tp+fn)
print("Recall:",
      recall_score(
          y_test,
          y_pred,
          average='weighted',
          zero_division=0
      ))

#F1= 2*(p*r/p+r)
print("F1 Score:",
      f1_score(
          y_test,
          y_pred,
          average='weighted',
          zero_division=0
      ))

# print("\nPrediction Probabilities:")
# print(gnb.predict_proba(X_test))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sample_data = pd.DataFrame({
    'Age': [5],
    'Temperature': [103.5],
    'Animal': ['Buffalo'],
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

sample_data = sample_data.drop(categorical_cols, axis=1)

sample_data = pd.concat(
    [
        sample_data.reset_index(drop=True),
        encoded_sample_df.reset_index(drop=True)
    ],
    axis=1
)

sample_data = scaler.transform(sample_data)

# prediction = gnb.predict(sample_data)

# probability = gnb.predict_proba(sample_data)
prediction = best_model.predict(sample_data)# unit 3
print("\nPredicted Disease:")
print(prediction)

# print("\nPrediction Probability:")
# print(probability)

