import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("livestock_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🐄Livestock Disease Prediction System")

st.write("Enter animal details and symptoms to predict disease.")

# Inputs

animal = st.selectbox(
    "Select Animal",
    ["Buffalo", "Cow", "Goat", "Sheep"]
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=30,
    value=5
)

temperature = st.number_input(
    "Temperature",
    min_value=90.0,
    max_value=110.0,
    value=103.5
)

symptom1 = st.text_input("Symptom 1")
symptom2 = st.text_input("Symptom 2")
symptom3 = st.text_input("Symptom 3")

# Prediction Button

if st.button("Predict Disease"):

    sample_data = pd.DataFrame({
        'Animal': [animal],
        'Age': [age],
        'Temperature': [temperature],
        'Symptom 1': [symptom1],
        'Symptom 2': [symptom2],
        'Symptom 3': [symptom3]
    })

    categorical_cols = [
        'Animal',
        'Symptom 1',
        'Symptom 2',
        'Symptom 3'
    ]

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

    st.markdown("---")

    st.success("✅ Disease Prediction Completed")

    st.markdown(
        f"""
        ## 🩺 Predicted Disease

        ### **{prediction[0].upper()}**
        """
    )

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(sample_data)
        confidence = max(probability[0]) * 100


    st.markdown("---")

    st.markdown(
        """
        <div style="
            background-color:#1E293B;
            padding:15px;
            border-radius:10px;
            text-align:center;
            color:white;
        ">
            <h4>👨‍💻 Developed By</h4>
            <h3>Amith S</h3>
            <p><b>SRN:</b> PES1PG25CA022</p>
            <p><b>PES University, Bangalore</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )