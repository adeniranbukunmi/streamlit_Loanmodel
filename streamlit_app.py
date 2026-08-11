import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os



st.set_page_config(
    page_title="Model Deployment",
    layout = "wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_approval_model2.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders

model, encoders = load_artifacts()

st.title("Loan Approval Prediction Demo")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    married = st.selectbox("Married", encoders['Married'].classes_)
    dependents = st.selectbox("Dependents", encoders['Dependents'].classes_)
    education = st.selectbox("Education", encoders['Education'].classes_)
    self_employed = st.selectbox("Self Employed", encoders['Self_Employed'].classes_)
    property_area = st.selectbox("Property Area", encoders['Property_Area'].classes_)

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=150)
    loan_amount_term = st.selectbox("Loan Amount Term", [360, 180, 120, 240, 60, 300, 84, 36, 12])
    credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good" if x==1 else "Bad")

if st.button("Predict"):
    # 1. Build raw input row — same column names as training
    row = {
        'Gender': gender, 'Married': married, 'Dependents': dependents,
        'Education': education, 'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income, 'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount, 'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history, 'Property_Area': property_area
    }
    input_df = pd.DataFrame([row])

    # 2. Feature engineering — SAME steps as training, just .transform-style, no fitting
    input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    input_df['ApplicantIncome'] = np.log1p(input_df['ApplicantIncome'])
    input_df['CoapplicantIncome'] = np.log1p(input_df['CoapplicantIncome'])
    input_df['LoanAmount'] = np.log1p(input_df['LoanAmount'])
    input_df['TotalIncome'] = np.log1p(input_df['TotalIncome'])

    # 3. Encode categoricals using the SAVED encoders (transform, never fit_transform)
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # 4. Match column order to X_train exactly
    input_df = input_df[model.feature_names_in_]  # sklearn stores this automatically after .fit()


    #  DEBUGIING — (this print out what the model expects vs what we are sending)
    st.write("Feature order model expects:", list(model.feature_names_in_))
    st.write("Row being sent to model:", input_df)
    st.write("Model's class labels:", model.classes_)

    # 5. Predict
    prediction = model.predict(input_df)[0]
    # result = " Loan Approved" if prediction == 1 else " Loan Rejected"
    result = "✅ Loan Approved" if prediction == 'Y' else "❌ Loan Rejected"
    st.subheader(result)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        approved_idx = classes.index('Y')
        st.write(f"Confidence: {max(proba)*100:.1f}%") 

    # --- Feature importance chart ---
    if hasattr(model, "feature_importances_"):
        st.subheader("What influenced this decision")
        importance_df = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)  # ascending so biggest bar ends up on top in the chart

        st.bar_chart(importance_df.set_index('Feature'))

        st.caption(
            "This shows how much each feature influenced the model's decisions overall "
            "(based on all applications it learned from) — not just this one prediction."
        )     










# st.write("# My first app")
# st.write("Hello, *World!* :sunglasses:")




# @st.cache_data
# def get_un_data() -> pd.DataFrame:
#     aws_bucket_url = "https://streamlit-demo-data.s3-us-west-2.amazonaws.com"
#     df = pd.read_csv(aws_bucket_url + "/agri.csv.gz")
#     return df.set_index("Region")  # type: ignore[no-any-return, unused-ignore]

# try:
#     df = get_un_data()
#     countries = st.multiselect(
#         "Choose countries", list(df.index), ["China", "United States of America"]
#     )
#     if not countries:
#         st.error("Please select at least one country.")
#     else:
#         data = df.loc[countries]
#         data /= 1000000.0
#         st.subheader("Gross agricultural production ($B)")
#         st.dataframe(data.sort_index())

#         data = data.T.reset_index()
#         data = pd.melt(data, id_vars=["index"]).rename(
#             columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
#         )
#         chart = (
#             alt.Chart(data)
#             .mark_area(opacity=0.3)
#             .encode(
#                 x="year:T",
#                 y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
#                 color="Region:N",
#             )
#         )
#         st.altair_chart(chart, width="stretch")
# except URLError as e:
#     st.error(f"This demo requires internet access. Connection error: {e.reason}")