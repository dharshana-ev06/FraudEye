
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="FraudEye",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("fraudeye_model.pkl")

model = load_model()

st.title("🛡️ FraudEye")
st.subheader("Suspicious Transaction Detection")
st.write("Machine learning based transaction analysis.")

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data = df.drop(columns=["Class"], errors="ignore")

    expected = list(model.feature_names_in_)

    if set(data.columns) != set(expected):
        st.error(
            "CSV columns do not match the trained model. "
            "Upload data with the same feature columns."
        )
    elif data.isnull().any().any():
        st.error("The CSV contains missing values.")
    else:
        data = data[expected]
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        results = data.copy()
        results["Prediction"] = [
            "Suspicious" if p == 1 else "Legitimate"
            for p in predictions
        ]
        results["Fraud Probability (%)"] = (
            probabilities * 100
        ).round(2)

        st.metric(
            "Transactions Analyzed",
            len(results)
        )
        st.metric(
            "Flagged as Suspicious",
            int((predictions == 1).sum())
        )

        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download Results",
            results.to_csv(index=False),
            file_name="fraudeye_results.csv",
            mime="text/csv"
        )