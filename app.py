import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model_aug.pkl")

# Page config
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")

#  UI CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

.legit {
    background-color: rgba(34,197,94,0.2);
    color: #22c55e;
}

.fraud {
    background-color: rgba(239,68,68,0.2);
    color: #ef4444;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>💳 Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered transaction risk analyzer</div>", unsafe_allow_html=True)

# Input Card
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📥 Enter Transaction Details")

col1, col2 = st.columns(2)

scaled_amount = col1.number_input("Scaled Amount", value=0.0)
v1 = col1.number_input("V1", value=0.0)

v2 = col2.number_input("V2", value=0.0)
v3 = col2.number_input("V3", value=0.0)

st.markdown("</div>", unsafe_allow_html=True)

# Predict Button
if st.button("🔍 Analyze Transaction", use_container_width=True):

    data = np.array([[scaled_amount, v1, v2, v3]])

    prediction = model.predict(data)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(data)[0][1]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Confidence Score")
        st.progress(float(prob))
        st.write(f"Fraud Probability: **{prob:.2f}**")
        st.markdown("</div>", unsafe_allow_html=True)

    # Result Card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧾 Result")

    if prediction == 1:
        st.markdown("<div class='result-box fraud'>🚨 Fraud Detected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box legit'>✅ Legit Transaction</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("🚀 Built with Streamlit | Fraud Detection ML Project")
