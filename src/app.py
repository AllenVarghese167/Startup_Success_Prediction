import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

st.title("StartUp Success Prediction")

def load_model():
    model = joblib.load('models/model.pkl')
    encoder = joblib.load('models/encoder.pkl')
    columns = joblib.load('models/feature_columns.pkl')
    return model, encoder, columns

model, encoder, feature_columns = load_model()

def gauge(p, label):
    color = "#16a34a" if label == "success" else "#dc2626"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 50], "color": "rgba(128,128,128,.12)"},
                         {"range": [50, 100], "color": "rgba(128,128,128,.25)"}]},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=0))
    return fig
 
st.sidebar.header("Enter Startup Details")
 
inputs = {
    "funding_rounds": st.sidebar.slider("Funding Rounds", 0, 8, 2),
    "founder_experience_years": st.sidebar.slider("Founder Experience (years)", 0, 24, 10),
    "team_size": st.sidebar.slider("Team Size", 2, 299, 100),
    "market_size_billion": st.sidebar.number_input("Market Size (Billion $)", 0.0, 1100.0, 20.0),
    "product_traction_users": st.sidebar.number_input("Product Traction (users)", 0, 1000000, 250000),
    "burn_rate_million": st.sidebar.number_input("Burn Rate (Million $)", 0.0, 400.0, 12.0),
    "revenue_million": st.sidebar.number_input("Revenue", 0.0, 4500000.0, 600000.0),
    "investor_type": st.sidebar.selectbox("Investor Type", ['angel', 'none', 'tier1_vc', 'tier2_vc']),
    "sector": st.sidebar.selectbox("Sector", ['AI', 'Climate', 'Crypto', 'Ecommerce', 'Fintech', 'Health', 'SaaS']),
    "founder_background": st.sidebar.selectbox("Founder Background", ['academic', 'ex_bigtech', 'first_time', 'serial_founder']),
}
 
if st.sidebar.button("Predict Startup Success"):
    input_data = pd.DataFrame([inputs])
 
    categories = {
        "investor_type": ['angel', 'none', 'tier1_vc', 'tier2_vc'],
        "sector": ['AI', 'Climate', 'Crypto', 'Ecommerce', 'Fintech', 'Health', 'SaaS'],
        "founder_background": ['academic', 'ex_bigtech', 'first_time', 'serial_founder']
    }
 
    for col, cats in categories.items():
        input_data[col] = pd.Categorical(
            input_data[col], categories=cats
        )
 
    input_encoded = pd.get_dummies(
        input_data,
        columns=list(categories.keys()),
        dtype=int
    )
 
    input_encoded = input_encoded.reindex(
        columns=feature_columns, fill_value=0
    )
 
    probabilities = model.predict_proba(input_encoded)[0]
    prediction = int(np.argmax(probabilities))
    label = encoder.inverse_transform([prediction])[0]
    confidence = float(probabilities[prediction])
 
    st.plotly_chart(gauge(confidence, label), width="stretch")
    st.metric(f"Probability of {label}", f"{confidence * 100:.1f}%")
 
    if label == "success":
        st.success("Startup is predicted to SUCCEED (IPO or Acquisition).")
    else:
        st.error("Startup is predicted to FAIL.")
else:
    st.info("Enter the startup details in the sidebar and click Predict.")