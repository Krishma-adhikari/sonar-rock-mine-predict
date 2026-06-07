import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('sonar.pkl')

st.title('Sonar Rock vs Mine Prediction')
st.write("Upload a CSV file with 60 features per row")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # validate feature count
    if df.shape[1] != 60:
        st.error(f"Model expects 60 features, but got {df.shape[1]}")
        st.stop()

    # keep only first 60 columns
    df = df.iloc[:, :60]

    st.write("Preview of data:")
    st.dataframe(df.head())

    if st.button("Predict"):

        prediction = model.predict(df)

        df["Prediction"] = prediction

        df["Result"] = df["Prediction"].apply(
            lambda x: "Rock 🪨" if x == "R" else "Mine 💣"
        )

        st.success("Prediction Complete!")
        st.dataframe(df)

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv_data,
            "predictions.csv",
            "text/csv"
        )
